#!/usr/bin/env python3
"""**옮김 vs 중복** — re-ID 가 없을 때 부재 증거가 그 역할을 대신하는가.

    $P scripts/hdepic_moveordup.py --root /Volumes/External_SSD/khronos/hdepic

설계 논점(2026-08-21, 사용자 지적):

    씬그래프에 `컵@거실` 이 있는데 `컵@부엌` 이 새로 관측됐다.
    **re-ID 가 없으면 "옮긴 것" 인지 "두 개" 인지 알 수 없다.**
    거실에 컵이 **없으면** 옮긴 것이고, **있으면** 두 개다.
    → **부재 증거가 re-ID 의 역할을 대신한다.** 이것이 없으면 씬그래프에
      컵이 계속 늘어난다.

그래서 부재는 "어디 있어" 의 답(그건 마지막 목격으로 된다)이 아니라
**"옮김이냐 중복이냐" 를 가르는 판정**이다.

⚠️ HD-EPIC 주석은 동명 객체를 `plate2`·`knife2` 로 번호를 붙여 구분해뒀다 —
**re-ID 문제가 인위적으로 제거된 상태**다. 우리 조건(번호 없이 `plate` 만 보임)을
복원하려면 **번호를 지워야** 한다. 지우면 P03 에서 중복이 22건 나온다
(`spoon` 4개 · `knife` 5개 · `fork` 7개 …).

과제:
    ① **옮김** — 한 물체의 fixture 가 A→B (정답: 같은 개체)
    ② **중복** — 같은 이름의 서로 다른 개체가 A 와 B 에 (정답: 다른 개체)

판정: A 에 그 물건이 **아직 있는가**(부재 증거). 없으면 옮김, 있으면 중복.
비교군은 부재 증거 없이 **항상 옮김**(또는 항상 중복)으로 찍는 것이다.
"""
import argparse, json, os, re, sys, tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.scenediff_timeline import visits_of                 # noqa: E402
FPS = 30.0


def base(n):
    return re.sub(r"\d+$", "", str(n)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/hdepic")
    ap.add_argument("--part", default="P03")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--anchor-n", type=int, default=8)
    ap.add_argument("--anchor-q", type=float, default=0.70)
    ap.add_argument("--vote-k", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch, cv2
    from PIL import Image
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              CLIPTextModelWithProjection, CLIPTokenizer)
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    tk = CLIPTokenizer.from_pretrained(cm)
    tn = CLIPTextModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()

    def emb_img(paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            with torch.no_grad():
                e = cn(**cp(images=ims, return_tensors="pt").to(args.device)).image_embeds
            out.append(e.cpu().numpy().astype(np.float32))
        E = np.concatenate(out)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def emb_txt(ws):
        with torch.no_grad():
            t = tk(["a photo of a " + w for w in ws], padding=True,
                   truncation=True, return_tensors="pt").to(args.device)
            e = tn(**t).text_embeds.cpu().numpy().astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    assoc = json.load(open(os.path.join(args.root, "assoc_info.json")))
    mask = json.load(open(os.path.join(args.root, "mask_info.json")))
    vdir = os.path.join(args.root, "Videos", args.part)
    tmp = tempfile.mkdtemp()
    rows = []

    vids = sorted(v for v in assoc if v.startswith(args.part)
                  and os.path.exists(os.path.join(vdir, v + ".mp4")))
    for vi_, vid in enumerate(vids):
        mi = mask.get(vid, {})
        info = {}
        for oid, o in assoc[vid].items():
            nm = o.get("name")
            obs = sorted((mi[m]["frame_number"] / FPS, mi[m]["fixture"])
                         for t in o.get("tracks", []) for m in t.get("masks", [])
                         if m in mi and mi[m].get("fixture"))
            if nm and obs:
                info[oid] = (base(nm), obs)
        # 과제 구성
        cases = []
        for oid, (nm, obs) in info.items():                 # ① 옮김
            fx = [f for _, f in obs]
            ch = next((i for i in range(1, len(obs)) if fx[i] != fx[0]), None)
            if ch is not None and len(obs[:ch]) >= 2:
                cases.append((nm, [t for t, _ in obs[:ch]], obs[ch][0], 1))
        byname = defaultdict(list)
        for oid, (nm, obs) in info.items():
            byname[nm].append(obs)
        for nm, lst in byname.items():                      # ② 중복
            if len(lst) < 2:
                continue
            fx = [{f for _, f in o} for o in lst]
            if len(set().union(*fx)) > 1 and not set.intersection(*fx):
                a, b = lst[0], lst[1]
                if len(a) >= 2:
                    cases.append((nm, [t for t, _ in a], b[-1][0], 2))
        if not cases:
            continue
        cap = cv2.VideoCapture(os.path.join(vdir, vid + ".mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0
        if dur < 60:
            cap.release(); continue
        secs, paths = [], []
        for i, s in enumerate(np.arange(0, dur, args.every)):
            f = int(round(s * fps))
            if not (0 <= f < total):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            p = os.path.join(tmp, "M%03d_%05d.jpg" % (vi_, i))
            cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            secs.append(float(s)); paths.append(p)
        cap.release()
        if len(paths) < 20:
            continue
        ts = np.array(secs); E = emb_img(paths)
        names = [c[0] for c in cases]
        T = emb_txt(names)
        for ci, (nm, anchor_t, after_t, case) in enumerate(cases):
            ai = np.unique(np.searchsorted(ts, np.array(anchor_t)))
            ai = ai[(ai >= 0) & (ai < len(ts))]
            if len(ai) < 2:
                continue
            sel = ai[np.linspace(0, len(ai) - 1, min(args.anchor_n, len(ai))).astype(int)]
            A = E[sel]
            thr = float(np.quantile((A @ A.T)[np.triu_indices(len(A), 1)], args.anchor_q))
            after = np.nonzero(ts > after_t)[0]
            if len(after) < 3:
                continue
            S = E[after] @ A.T
            sim = np.sort(S, 1)[:, -min(args.vote_k, len(A)):].mean(1)
            ok_ = after[sim >= thr]
            vs = visits_of(ok_, args.max_gap)
            if not vs:
                continue
            # 그 자리를 다시 봤을 때 물건이 아직 있는가 = 부재 증거
            val = float(np.median((E[vs[-1]] @ T[ci])))
            rows.append(dict(vid=vid, obj=nm, case=case, val=val))
        print("  %-22s 사례 %-3d (누적 %d)" % (vid[4:], len(cases), len(rows)), flush=True)

    c1 = [r["val"] for r in rows if r["case"] == 1]
    c2 = [r["val"] for r in rows if r["case"] == 2]
    print("\n판정 %d건 — ①옮김(정답: 같은 개체) %d · ②중복(정답: 다른 개체) %d"
          % (len(rows), len(c1), len(c2)))
    if len(c1) < 3 or len(c2) < 3:
        print("표본 부족")
        if args.out:
            json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        return
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(c2, c1, alternative="greater")
    print("그 자리 재관측 시 물건 점수: ①옮김 %+.3f · ②중복 %+.3f · 분리 AUC **%.3f** (p=%.4f)"
          % (np.median(c1), np.median(c2), u / (len(c1) * len(c2)), p))
    print("\n%-9s %-10s %-10s %s" % ("문턱", "①옮김 정답", "②중복 정답", "균형정확도"))
    best = None
    for t in np.percentile(c1 + c2, [10, 25, 50, 75, 90]):
        a1 = float(np.mean([v < t for v in c1]))       # 없으면 옮김
        a2 = float(np.mean([v >= t for v in c2]))      # 있으면 중복
        ba = (a1 + a2) / 2
        print("%-9.3f %-10.2f %-10.2f %.3f %s" % (t, a1, a2, ba,
              "**우연 초과**" if ba > 0.5 else ""))
        if best is None or ba > best[0]:
            best = (ba, t, a1, a2)
    print("→ 최고 균형정확도 **%.3f** (문턱 %.3f · ① %.2f · ② %.2f)" % best)
    maj = max(len(c1), len(c2)) / len(rows)
    print("  대조: 부재 증거 없이 **항상 한쪽으로 찍기** = %.3f" % maj)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
