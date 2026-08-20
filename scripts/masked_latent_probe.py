#!/usr/bin/env python3
"""**물체를 가린 latent** 로 장소를 찾을 수 있는가 — 부재 편향을 걷어낸 앵커.

    $P scripts/masked_latent_probe.py --root <benchmark/data>

문제(사용자 지적): latent 로 장소를 찾으면 **물건이 사라진 프레임일수록 앵커와 덜
닮게 되어** 부재가 가려진다. 원래 설계가 질의에서 키워드를 뺐던 것과 같은 이유다.

해법: 앵커를 만들 때 **그 물체 영역을 빼고** 임베딩한다 = "물건 없는 그 장소".
SceneDiff 가 프레임별 인스턴스 마스크를 주므로 그대로 할 수 있다.

⚠️ **마스크는 앵커에만 필요하다.** 검색 대상 프레임 전부에 seg 를 돌릴 필요가 없다 —
앵커는 씬그래프가 아는 "마지막으로 본 몇 장" 뿐이다. 전 프레임 seg 는 불필요하다.

구현: 검은 칠은 **분포 밖 이미지**를 만들어 임베딩을 망친다. 대신 CLIP 패치 토큰을
써서 **물체 밖 패치만 평균**한다(ViT-B/16 · 224px → 14×14 격자).
⚠️ CLIP 패치 토큰은 CLS 보다 약하다고 알려져 있다 — 그래서 마스크 없는 패치평균도
같이 재서 **마스킹 효과와 패치평균 자체의 손해를 분리**한다.

세 앵커를 비교한다:
    CLS      기존 방식(전체 프레임)          ← 부재 편향 있음
    패치평균  마스크 없이 패치만 평균          ← 패치평균의 손해만
    **마스크** 물체 밖 패치만 평균            ← 편향 제거판
"""
import argparse, glob, json, os, pickle, re, subprocess, sys, tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.scenediff_absence import base_label, frames_of, video_of   # noqa: E402


def grab_idx(mp4, idxs, tmp, tag):
    """프레임 **번호**로 뽑는다(마스크가 프레임 번호로 색인돼 있다)."""
    out = {}
    sel = "+".join("eq(n\\,%d)" % i for i in idxs)
    pat = os.path.join(tmp, "%s_%%03d.jpg" % tag)
    # ⚠️ `-vsync` 는 이 ffmpeg 빌드에서 제거됐다("Unrecognized option 'vsync'").
    # `-fps_mode passthrough` 가 대체다 — 없으면 프레임이 하나도 안 나온다.
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf",
                    "select='%s'" % sel, "-fps_mode", "passthrough", "-y", pat],
                   check=False)
    got = sorted(glob.glob(os.path.join(tmp, "%s_*.jpg" % tag)))
    for i, p in zip(sorted(idxs), got):
        out[i] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--nanchor", type=int, default=4, help="앵커 프레임 수")
    ap.add_argument("--nframes", type=int, default=8, help="대상 클립당 프레임")
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    G = 14                                    # 224/16

    def encode(imgs, masks=None):
        """CLS · 패치평균 · 마스크패치평균 세 가지를 한 번에."""
        inp = cp(images=imgs, return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = cn.vision_model(**inp)
            h = cn.vision_model.post_layernorm(o.last_hidden_state)
            cls = cn.visual_projection(h[:, 0])
            pat = h[:, 1:]                                     # [N,196,D]
            allp = cn.visual_projection(pat.mean(1))
            if masks is None:
                mk = None
            else:
                M = torch.tensor(np.stack(masks), device=args.device)   # [N,14,14] bool
                w = (~M).reshape(len(imgs), -1).float()
                w = w / (w.sum(1, keepdim=True) + 1e-6)
                mk = cn.visual_projection((pat * w[..., None]).sum(1))
        def nz(x):
            x = x.cpu().numpy().astype(np.float32)
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        return nz(cls), nz(allp), (nz(mk) if mk is not None else None)

    tmp = tempfile.mkdtemp()
    # ── 대상 풀: 모든 쌍의 v1·v2 프레임(마스크 없이) CLS·패치평균
    pairs = []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if len(pairs) >= args.limit or not os.path.isdir(d):
            continue
        f = os.path.join(d, "segments.pkl")
        v1, v2 = video_of(d, 1), video_of(d, 2)
        if not (os.path.exists(f) and v1 and v2):
            continue
        pairs.append((os.path.basename(d), d, v1, v2))
    print("쌍 %d" % len(pairs))

    pool_cls, pool_pat, owner = [], [], []
    for k, (name, d, v1, v2) in enumerate(pairs):
        for wv, v in (("v1", v1), ("v2", v2)):
            fr = frames_of(v, args.nframes, tmp, "p%03d%s" % (k, wv))
            if len(fr) < 3:
                continue
            ims = [Image.open(p).convert("RGB") for p in fr]
            c, a, _ = encode(ims)
            pool_cls.append(c); pool_pat.append(a)
            owner += [(name, wv)] * len(ims)
        if (k + 1) % 10 == 0:
            print("  풀 %d/%d" % (k + 1, len(pairs)))
    P_cls = np.concatenate(pool_cls); P_pat = np.concatenate(pool_pat)
    owner = np.array(owner, dtype=object)
    print("대상 프레임 %d" % len(owner))

    # ── 앵커: Removed 물체의 마스크 프레임에서 세 가지 앵커
    rows = []
    for k, (name, d, v1, v2) in enumerate(pairs):
        seg = pickle.load(open(os.path.join(d, "segments.pkl"), "rb"))
        v1o = seg.get("video1_objects") or {}
        for o in (seg.get("objects") or []):
            if not (o.get("in_video1") and not o.get("in_video2")):
                continue
            oi = o.get("original_obj_idx")
            fr = v1o.get(oi) or v1o.get(str(oi)) or {}
            if len(fr) < args.nanchor:
                continue
            keys = sorted(fr, key=lambda x: int(x))
            pick = [keys[i] for i in np.linspace(0, len(keys) - 1, args.nanchor).astype(int)]
            got = grab_idx(v1, [int(x) for x in pick], tmp, "a%03d_%s" % (k, oi))
            ims, mks = [], []
            for key in pick:
                p = got.get(int(key))
                if not p:
                    continue
                im = Image.open(p).convert("RGB")
                m = mask_utils.decode(fr[key]).astype(bool)
                # 14×14 격자로 축소 — 패치 하나라도 물체가 닿으면 제외
                hh, ww = m.shape
                g = np.zeros((G, G), bool)
                for a_ in range(G):
                    for b_ in range(G):
                        g[a_, b_] = m[a_ * hh // G:(a_ + 1) * hh // G,
                                      b_ * ww // G:(b_ + 1) * ww // G].any()
                if g.all():
                    continue                       # 물체가 화면을 다 덮으면 못 쓴다
                ims.append(im); mks.append(g)
            if len(ims) < 2:
                continue
            c, a, mk = encode(ims, mks)
            frac = float(np.mean([g.mean() for g in mks]))
            rec = dict(pair=name, label=o["label"], n=len(ims), mask_frac=frac)
            for tag, A, P in (("cls", c, P_cls), ("patch", a, P_pat), ("mask", mk, P_pat)):
                v = A.mean(0); v /= np.linalg.norm(v) + 1e-9
                sim = P @ v
                same2 = sim[(owner[:, 0] == name) & (owner[:, 1] == "v2")]
                other = sim[owner[:, 0] != name]
                if not len(same2) or not len(other):
                    continue
                rec[tag] = dict(same_v2=float(np.median(same2)),
                                other_max=float(np.max(other)),
                                margin=float(np.median(same2) - np.max(other)),
                                rank=int((other > np.median(same2)).sum()))
            rows.append(rec)
            print("  %-40s %-18s 마스크비 %.2f" % (name[:40], o["label"][:18], frac))
    report(rows)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


def report(rows):
    print("\n앵커 %d개 (Removed 물체) · 마스크가 가린 패치 비율 중앙 %.2f"
          % (len(rows), np.median([r["mask_frac"] for r in rows]) if rows else float("nan")))
    if not rows:
        return
    print("\n%-8s %-11s %-11s %-11s %s"
          % ("앵커", "같은장소v2", "남의장소최고", "여백", "장소 1등 비율"))
    for tag, nm in (("cls", "CLS"), ("patch", "패치평균"), ("mask", "**마스크**")):
        s = [r[tag] for r in rows if tag in r]
        if len(s) < 3:
            print("%-8s 표본 부족" % nm)
            continue
        m = np.array([x["margin"] for x in s])
        top1 = np.mean([x["rank"] == 0 for x in s])
        print("%-8s %-11.3f %-11.3f %-11.3f %.0f%%  (여백>0 %.0f%%)"
              % (nm, np.median([x["same_v2"] for x in s]),
                 np.median([x["other_max"] for x in s]), np.median(m),
                 100 * top1, 100 * (m > 0).mean()))
    print("\n→ 여백 = (같은 장소 변화후) − (가장 닮은 남의 장소). 양수여야 장소를 찾는다.")
    print("  마스크가 CLS 보다 여백이 크면 **부재 편향을 실제로 걷어낸 것**이다.")
    print("  패치평균과 비교해야 순수 마스킹 효과가 보인다(패치평균 자체가 CLS 보다 약함).")


if __name__ == "__main__":
    main()
