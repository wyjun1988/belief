#!/usr/bin/env python3
"""**증거 검색**과 **최신성**을 잰다 — 부재가 아니라 "그 물건 어디서 봤나".

    $P scripts/evidence_retrieval.py --dataset scenediff --root <...>

부재 판정(㉗)은 "장소를 찾아 지금 없는지" 를 본다. 그 앞단에 **증거 검색**이 있다:
질의를 받아 그 물건이 **실제로 보인 프레임**을 올려야 한다. 두 가지를 나눠 잰다.

  **Ⓐ 증거 검색** — 물체 이름으로 질의했을 때 GT 관측 프레임이 상위에 오는가
      지표: hit@1 / hit@5 / hit@10 · 우연 = k/전체프레임

  **Ⓑ 최신성** — 같은 물체가 여러 번 보였을 때 **가장 최근 관측**이 올라오는가
      "그 물건 어디 뒀더라" 는 **지금 상태**를 묻는 것이므로 과거 관측을 올리면
      틀린 답이 된다. 지표: 상위 1개가 마지막 관측 구간에 드는 비율,
      그리고 최근성 가중(exp(−Δt/τ))을 켰을 때의 변화.

우리 저장소 실측 대조군: SuperMemory 물체·위치 **hit@5 0.75**(질문 원문 · 최근성
τ=12h 기본). 그때는 질의가 사람이 쓴 문장이었고 여기서는 물체 이름이라 조건이 다르다.
"""
import argparse, glob, json, os, pickle, re, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FPS = 30.0


def load_clip(device):
    import torch
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              CLIPTextModelWithProjection, CLIPTokenizer)
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(device).eval()
    tk = CLIPTokenizer.from_pretrained(cm)
    tn = CLIPTextModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(device).eval()

    def emb_img(paths, bs=32):
        from PIL import Image
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            with torch.no_grad():
                e = cn(**cp(images=ims, return_tensors="pt").to(device)).image_embeds
            out.append(e.cpu().numpy().astype(np.float32))
        E = np.concatenate(out)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def emb_txt(words):
        with torch.no_grad():
            t = tk(["a photo of a " + w for w in words], padding=True,
                   truncation=True, return_tensors="pt").to(device)
            e = tn(**t).text_embeds.cpu().numpy().astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    return emb_img, emb_txt


def frames_at(mp4, secs, tmp, tag):
    import cv2
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for i, s in enumerate(secs):
        f = int(round(s * fps))
        if total and not (0 <= f < total):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        p = os.path.join(tmp, "%s_%05d.jpg" % (tag, i))
        cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out.append((float(s), p))
    cap.release()
    return out


def eval_one(ts, E, queries, gt_times, tol, taus):
    """queries: [(이름, [관측시각...])]. GT 관측 ±tol 초 안의 프레임을 정답으로."""
    res = {}
    for tau in taus:
        hits = {1: [], 5: [], 10: []}
        latest = []
        for (nm, qv), obs in zip(queries, gt_times):
            if not obs:
                continue
            sim = E @ qv
            if tau is not None:
                # **최근성 가중** — 질의 시각을 타임라인 끝으로 본다
                dt = (ts[-1] - ts) / 3600.0
                sim = sim + np.log(np.exp(-dt / tau) + 1e-12) * 0.0 + (-dt / tau)
            order = np.argsort(-sim)
            good = np.zeros(len(ts), bool)
            for o in obs:
                good |= np.abs(ts - o) <= tol
            if not good.any():
                continue
            for k in hits:
                hits[k].append(bool(good[order[:k]].any()))
            # 최신성 — 상위 1개가 **마지막 관측** 근처인가
            last = max(obs)
            latest.append(bool(abs(ts[order[0]] - last) <= tol))
        res[tau] = (
            {k: float(np.mean(v)) if v else float("nan") for k, v in hits.items()},
            float(np.mean(latest)) if latest else float("nan"),
            len(latest))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hdepic", choices=["hdepic", "scenediff"])
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/hdepic")
    ap.add_argument("--part", default="P03")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=3.0, help="정답 허용 오차(초)")
    ap.add_argument("--limit-vid", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb_img, emb_txt = load_clip(args.device)
    tmp = tempfile.mkdtemp()
    TAUS = [None, 4.0, 1.0, 0.25]        # None = 최근성 미적용, 나머지는 시간(h)
    agg = {t: [] for t in TAUS}
    n_q = 0

    assoc = json.load(open(os.path.join(args.root, "assoc_info.json")))
    mask = json.load(open(os.path.join(args.root, "mask_info.json")))
    vdir = os.path.join(args.root, "Videos", args.part)
    vids = sorted(v for v in assoc if v.startswith(args.part)
                  and os.path.exists(os.path.join(vdir, v + ".mp4")))
    print("녹화 %d개 사용 가능" % len(vids))
    for vi_, vid in enumerate(vids[:args.limit_vid]):
        mp4 = os.path.join(vdir, vid + ".mp4")
        mi = mask.get(vid, {})
        obs_by = {}
        for oid, o in assoc[vid].items():
            nm = o.get("name")
            if not nm:
                continue
            tt = [mi[m]["frame_number"] / FPS
                  for t in o.get("tracks", []) for m in t.get("masks", []) if m in mi]
            if tt:
                obs_by.setdefault(nm, []).extend(tt)
        if not obs_by:
            continue
        import cv2
        cap = cv2.VideoCapture(mp4)
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or FPS)
        cap.release()
        if dur < 60:
            continue
        got = frames_at(mp4, np.arange(0, dur, args.every), tmp, "q%02d" % vi_)
        if len(got) < 30:
            continue
        ts = np.array([g[0] for g in got])
        E = emb_img([g[1] for g in got])
        names = sorted(obs_by)
        Q = emb_txt(names)
        r = eval_one(ts, E, list(zip(names, Q)), [sorted(obs_by[n]) for n in names],
                     args.tol, TAUS)
        for t in TAUS:
            agg[t].append(r[t])
        n_q += r[TAUS[0]][2]
        print("  %-22s 프레임 %-4d · 질의 %-3d · hit@5 %.2f · 최신성 %.2f"
              % (vid[4:], len(ts), r[None][2], r[None][0][5], r[None][1]), flush=True)

    print("\n질의 %d건 · 프레임 표집 간격 %.0f초 · 정답 허용 ±%.0f초"
          % (n_q, args.every, args.tol))
    print("\n%-10s %-8s %-8s %-8s %s" % ("최근성", "hit@1", "hit@5", "hit@10", "최신성"))
    for t in TAUS:
        rs = [x for x in agg[t] if x[2]]
        if not rs:
            continue
        h = {k: float(np.mean([x[0][k] for x in rs])) for k in (1, 5, 10)}
        lt = float(np.mean([x[1] for x in rs]))
        print("%-10s %-8.3f %-8.3f %-8.3f %.3f"
              % ("없음" if t is None else "τ=%.2fh" % t, h[1], h[5], h[10], lt))
    print("\nⒶ 증거 검색 = hit@k · Ⓑ 최신성 = 1등이 **마지막 관측**에 드는 비율")
    print("  (대조: SuperMemory 물체·위치 hit@5 **0.75** — 질의가 사람 문장이었다)")
    if args.out:
        json.dump({str(k): [list(x[0].items()) + [x[1], x[2]] for x in v]
                   for k, v in agg.items()}, open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
