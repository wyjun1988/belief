#!/usr/bin/env python3
"""HD-EPIC 에 **같은 판정 규칙**을 얹는다 — `fixture` 라벨이 있는 첫 데이터.

    $P scripts/hdepic_timeline.py --root /Volumes/External_SSD/khronos/hdepic

세 데이터셋의 성격이 다르다:

| | 부재 라벨 | 인스턴스 신원 | 시간 간격 |
|---|---|---|---|
| SceneDiff | `Removed` 명시 | 있음 | 없음(3~19초) |
| ADT | **없음**(변위에서 유도 — ㉘ 에서 무리로 판명) | 있음 | 2분 |
| **HD-EPIC** | **`fixture` 명시** | 녹화 **안**만 | 최대 56분 |

`fixture` 는 "어느 가구 위에 있었나" 라 우리 과제와 정확히 일치한다. 변위 0.7 m 라도
`counter.006 → sink.001` 이면 **자리를 뜬 것이 확실하다** — ADT 에서 변위로 추측하다
실패한 부분이 여기서는 라벨로 해결된다.

⚠️ **녹화 간에는 물체 ID 가 유지되지 않는다**(8,382개 중 재등장 0). 그래서 세션 간
사례(이름 기준 460종)는 쓸 수 없다 — `plate` 가 18개 녹화에 나오는데 같은 접시인지
다른 접시인지 못 가른다. **녹화 안 사례만** 쓴다.

과제 구성:
    ① fixture 가 바뀐 물체 — 바뀌기 **전** 관측을 앵커로. 정답 = **없음**
    ② fixture 가 하나인 물체 — 앞쪽 관측을 앵커로. 정답 = **있음**

판정 규칙은 SceneDiff·ADT 와 동일하다(㉗):
    앵커 자기유사도 분위 문턱 → 앵커 이후에서 그 장소 찾기 → 마지막 방문에서 키워드
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.scenediff_timeline import visits_of                 # noqa: E402

FPS = 30.0


def frames_at(mp4, secs, tmp, tag):
    """지정한 초 위치의 프레임들을 OpenCV 로 뽑는다."""
    import cv2
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for i, s in enumerate(secs):
        f = int(round(s * fps))
        if total and (f < 0 or f >= total):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        p = os.path.join(tmp, "%s_%04d.jpg" % (tag, i))
        cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out.append((s, p))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/hdepic")
    ap.add_argument("--part", default="P03")
    ap.add_argument("--every", type=float, default=2.0, help="타임라인 표집 간격(초)")
    ap.add_argument("--anchor-n", type=int, default=8)
    ap.add_argument("--anchor-q", type=float, default=0.70,
                    help="SceneDiff 실측 최적값(㉗). 앵커는 **구간 전체에서 균등 표집**"
                         " 한다 — 연속 프레임으로 잡으면 자기유사도가 몰려 문턱이"
                         " 못 쓰게 된다(㉘ 에서 76건 중 2건만 채점됐던 원인)")
    ap.add_argument("--vote-k", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--min-static", type=int, default=2)
    ap.add_argument("--limit-vid", type=int, default=99)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              CLIPTextModelWithProjection, CLIPTokenizer)
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    tok = CLIPTokenizer.from_pretrained(cm)
    tn = CLIPTextModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()

    def embed(paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            with torch.no_grad():
                e = cn(**cp(images=ims, return_tensors="pt").to(args.device)).image_embeds
            out.append(e.cpu().numpy().astype(np.float32))
        E = np.concatenate(out)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def text(words):
        with torch.no_grad():
            t = tok(["a photo of a " + w for w in words], padding=True,
                    truncation=True, return_tensors="pt").to(args.device)
            e = tn(**t).text_embeds.cpu().numpy().astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    assoc = json.load(open(os.path.join(args.root, "assoc_info.json")))
    mask = json.load(open(os.path.join(args.root, "mask_info.json")))
    vdir = os.path.join(args.root, "Videos", args.part)
    tmp = tempfile.mkdtemp()

    rows = []
    vids = sorted(v for v in assoc if v.startswith(args.part))
    for vi_, vid in enumerate(vids[:args.limit_vid]):
        mp4 = os.path.join(vdir, vid + ".mp4")
        if not os.path.exists(mp4):
            continue
        mi = mask.get(vid, {})
        # 물체별 관측 (시각초, fixture)
        objs = {}
        for oid, o in assoc[vid].items():
            obs = []
            for t in o.get("tracks", []):
                for m in t.get("masks", []):
                    r = mi.get(m)
                    if r and r.get("fixture"):
                        obs.append((r["frame_number"] / FPS, r["fixture"]))
            if obs:
                objs[oid] = (o.get("name"), sorted(obs))
        if not objs:
            continue
        # 타임라인 프레임 — 영상 전체를 일정 간격으로
        import cv2
        cap = cv2.VideoCapture(mp4); dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / \
            (cap.get(cv2.CAP_PROP_FPS) or FPS); cap.release()
        if dur < 60:
            continue
        secs = np.arange(0, dur, args.every)
        got = frames_at(mp4, secs, tmp, "v%02d" % vi_)
        if len(got) < 30:
            continue
        ts = np.array([g[0] for g in got])
        E = embed([g[1] for g in got])
        names = sorted({n for n, _ in objs.values() if n})
        if not names:
            continue
        T = text(names)
        Z = T @ E.T                                  # [단어, 프레임]
        Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
        ni = {n: i for i, n in enumerate(names)}

        def judge(anchor_t, after_t, ki):
            ai = np.searchsorted(ts, anchor_t)
            ai = ai[(ai >= 0) & (ai < len(ts))]
            if len(ai) < 2:
                return None
            A = E[ai]
            base = (A @ A.T)[np.triu_indices(len(A), 1)]
            thr = float(np.quantile(base, args.anchor_q))
            after = np.nonzero(ts > after_t)[0]
            if len(after) < 3:
                return None
            S = E[after] @ A.T
            k = min(args.vote_k, len(A))
            sim = np.sort(S, 1)[:, -k:].mean(1)
            ok = after[sim >= thr]
            vs = visits_of(ok, args.max_gap)
            if not vs:
                return None
            return float(np.median(Z[ki, vs[-1]]))

        nm_ = ns_ = 0
        for oid, (nm, obs) in objs.items():
            if not nm or nm not in ni:
                continue
            ki = ni[nm]
            fxs = [f for _, f in obs]
            chg = next((i for i in range(1, len(obs)) if fxs[i] != fxs[0]), None)
            if chg is not None:
                pre = [t for t, f in obs[:chg]]
                if len(pre) < 2:
                    continue
                # **앵커는 구간 전체에서 균등 표집** (연속 아님)
                sel = np.array(pre)[np.linspace(0, len(pre) - 1,
                                                min(args.anchor_n, len(pre))).astype(int)]
                v = judge(sel, obs[chg][0], ki)
                if v is not None:
                    rows.append(dict(vid=vid, obj=nm, case=1, val=v,
                                     fx=(fxs[0], fxs[chg])))
                    nm_ += 1
            elif len(set(fxs)) == 1 and len(obs) >= args.min_static:
                tt = [t for t, _ in obs]
                sel = np.array(tt)[np.linspace(0, len(tt) - 1,
                                               min(args.anchor_n, len(tt))).astype(int)]
                v = judge(sel, tt[-1], ki)
                if v is not None:
                    rows.append(dict(vid=vid, obj=nm, case=2, val=v, fx=(fxs[0],)))
                    ns_ += 1
        print("  %-22s 프레임 %-4d · ①이동 %-3d · ②정적 %-3d (누적 %d)"
              % (vid[4:], len(got), nm_, ns_, len(rows)), flush=True)

    report(rows)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


def report(rows):
    c1 = [r["val"] for r in rows if r["case"] == 1]
    c2 = [r["val"] for r in rows if r["case"] == 2]
    print("\n판정 %d건 — ①fixture 바뀜(정답 없음) %d · ②그대로(정답 있음) %d"
          % (len(rows), len(c1), len(c2)))
    if len(c1) < 5 or len(c2) < 5:
        print("표본 부족")
        return
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(c2, c1, alternative="greater")
    print("키워드 중앙: ①없음 %+.3f · ②있음 %+.3f · 분리 AUC **%.3f** (p=%.4f)"
          % (np.median(c1), np.median(c2), u / (len(c1) * len(c2)), p))
    print("\n%-9s %-9s %-9s %s" % ("문턱", "①정답률", "②정답률", "균형정확도"))
    best = None
    for t in np.percentile(c1 + c2, [10, 25, 50, 75, 90]):
        a1 = float(np.mean([v < t for v in c1])); a2 = float(np.mean([v >= t for v in c2]))
        ba = (a1 + a2) / 2
        print("%-9.3f %-9.2f %-9.2f %.3f %s" % (t, a1, a2, ba,
              "**우연 초과**" if ba > 0.5 else ""))
        if best is None or ba > best[0]:
            best = (ba, t, a1, a2)
    print("→ 최고 균형정확도 **%.3f** (문턱 %.3f · ① %.2f · ② %.2f)" % best)
    print("  (대조: SceneDiff 0.807 · ADT 0.592)")


if __name__ == "__main__":
    main()
