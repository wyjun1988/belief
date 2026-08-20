#!/usr/bin/env python3
"""ADT 실시퀀스에 **같은 판정 규칙**을 얹는다 — 타임라인을 조립하지 않는다.

    $P scripts/adt_timeline.py --root data/seq_new

SceneDiff 에서 부재 판정이 섰다(균형정확도 **0.807**, 방해 40개에서도 0.690).
다만 그 타임라인은 **인위로 조립한 것**이다 — 영상이 3~19초짜리라 "있음 → 다른방 →
없음" 을 우리가 이어 붙였다. ADT 는 한 시퀀스가 1,370프레임(≈2분) 연속이라
**사람이 실제로 돌아다닌 순서 그대로** 같은 규칙을 시험할 수 있다.

규칙은 그대로다:
    ⓐ 물건을 마지막으로 본 구간을 앵커로
    ⓑ **그 이후** 프레임 중 앵커 자기유사도 문턱을 넘는 것을 그 장소로
    ⓒ 방문 구간으로 쪼개 **마지막 방문**을 보고, 거기서 키워드 검출도로 판정

과제 구성(SceneDiff ①② 와 같은 대조):
    ① 이동한 물체 — 이동 **전** 구간을 앵커로. 정답 = **없음**(자리를 떴다)
    ② 정적 물체  — 앞부분을 앵커로. 정답 = **있음**

⚠️ **앵커는 시간축에 퍼뜨려야 한다.** ADT 는 프레임 간격이 2라 "직전 8프레임" 을
쓰면 **연속 2초**가 되고, 그 8장이 서로 0.97 로 닮는다. 그 분위를 문턱으로 쓰면
0.97~0.98 이 되어 **이후 어떤 프레임도 못 넘는다**(실측: 통과 0프레임 · 16건 중 14건
판정 불가). SceneDiff 에서는 앵커 8장이 영상 전체에 균등 분포해 자기유사도가 넓게
퍼졌고 문턱이 적절히 잡혔다 — **같은 분위가 데이터에 따라 전혀 다른 문턱이 된다.**
그래서 앵커를 "직전 N프레임" 이 아니라 **이동 전 구간에서 균등 표집**한다.

⚠️ ADT 는 **변위가 작은 '이동'이 30%** 다(중앙 2.3 m 지만 1 m 미만이 27~33%).
0.35 m 는 탁자를 민 것이지 자리를 뜬 것이 아니므로 `--min-disp` 로 거른다.
"""
import argparse, glob, json, os, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.absence_evidence import PLACES                      # noqa: E402
from scripts.owl_presence import load_owl, owl_z                 # noqa: E402
from scripts.scenediff_timeline import visits_of                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq_new"))
    ap.add_argument("--owl-dir", default=None, help="OWLv2 검출 JSON 디렉터리(있으면)")
    ap.add_argument("--min-disp", type=float, default=1.0,
                    help="이 변위 미만은 '자리를 뜬 것' 으로 안 본다")
    ap.add_argument("--anchor-n", type=int, default=8, help="앵커 프레임 수")
    ap.add_argument("--anchor-q", type=float, default=0.70,
                    help="앵커 자기유사도 분위 문턱. SceneDiff 실측 최적값")
    ap.add_argument("--vote-k", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for sd in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(sd):
            continue
        gf = os.path.join(sd, "gt", "objects.json")
        cf = os.path.join(sd, "clip_frames.npz")
        if not (os.path.exists(gf) and os.path.exists(cf)):
            continue
        z = np.load(cf)
        E = z["emb"].astype(np.float32); fidx = z["idx"]
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        gt = json.load(open(gf))["instances"]
        cats = sorted({(r.get("category") or "").strip() for r in gt.values()
                       if r.get("category")})
        vocab = sorted(set(c for c in cats if c and len(c) > 2) | set(PLACES))
        # 키워드 존재도 — OWL 이 있으면 그걸, 없으면 CLIP 텍스트
        if args.owl_dir:
            # 기존 2시퀀스는 파일명이 짧은 별칭이다
            alias = {"Apartment_release_decoration_seq137_M1292": "owl_adt_decoration.json",
                     "Apartment_release_multiskeleton_party_seq102_M1292": "owl_adt_party.json"}
            base = os.path.basename(sd)
            f_ = os.path.join(args.owl_dir,
                              alias.get(base, "owl_%s.json" % base))
            if not os.path.exists(f_):
                print("  %-30s OWL 검출 없음 — 건너뜀" % base[:30])
                continue
            raw = json.load(open(f_))
            owl = {base: {int(os.path.splitext(k)[0]): v for k, v in raw.items()}}
            Z, _ = owl_z(owl, [(base, int(f)) for f in fidx],
                         vocab, E=E, device=args.device)
        else:
            from scripts.absence_evidence import presence
            Z, _ = presence(E, vocab, args.device)
        vi = {w: i for i, w in enumerate(vocab)}

        def judge(anchor_idx, after_from, ki):
            """앵커 이후에서 그 장소를 찾아 마지막 방문의 키워드 값."""
            A = E[anchor_idx]
            if len(A) < 2:
                return None
            SS = A @ A.T
            base = SS[np.triu_indices(len(A), 1)]
            thr = float(np.quantile(base, args.anchor_q))
            after = np.arange(after_from, len(E))
            if len(after) < 3:
                return None
            S = E[after] @ A.T
            k = min(args.vote_k, len(A))
            sim = np.sort(S, 1)[:, -k:].mean(1)
            ok = after[sim >= thr]
            vs = visits_of(ok, args.max_gap)
            if not vs:
                return None
            sel = vs[-1]
            return float(np.median(Z[ki, sel]))

        n_mv = n_st = 0
        for k, r in gt.items():
            c = (r.get("category") or "").strip()
            if c not in vi:
                continue
            ki = vi[c]
            mvs = [m for m in (r.get("moves") or [])
                   if m["displacement_m"] >= args.min_disp]
            if mvs:
                m0 = mvs[0]
                pre = np.nonzero(fidx < m0["start_idx"])[0]
                if len(pre) < args.anchor_n:
                    continue
                # 직전 N장이 아니라 **이동 전 구간 전체에서 균등 표집** — 연속
                # 프레임만 쓰면 자기유사도가 0.97 로 몰려 문턱이 못 쓰게 된다
                a = pre[np.linspace(0, len(pre) - 1, args.anchor_n).astype(int)]
                post = int(np.searchsorted(fidx, m0["end_idx"]))
                v = judge(a, post, ki)
                if v is not None:
                    rows.append(dict(seq=os.path.basename(sd)[:28], cat=c,
                                     case=1, truth="없음", val=v))
                    n_mv += 1
            elif not (r.get("moves") or []):
                if len(E) < args.anchor_n * 3:
                    continue
                # 정적 물체도 같은 조건 — 앞 절반에서 균등 표집, 뒤 절반에서 판정
                half = len(E) // 2
                a = np.linspace(0, half - 1, args.anchor_n).astype(int)
                v = judge(a, half, ki)
                if v is not None:
                    rows.append(dict(seq=os.path.basename(sd)[:28], cat=c,
                                     case=2, truth="있음", val=v))
                    n_st += 1
        print("  %-30s 이동 %-3d · 정적 %-4d (누적 %d)"
              % (os.path.basename(sd)[:30], n_mv, n_st, len(rows)))

    report(rows)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


def report(rows):
    c1 = [r["val"] for r in rows if r["case"] == 1]
    c2 = [r["val"] for r in rows if r["case"] == 2]
    print("\n판정 %d건 — ①이동(정답 없음) %d · ②정적(정답 있음) %d"
          % (len(rows), len(c1), len(c2)))
    if len(c1) < 5 or len(c2) < 5:
        print("표본 부족")
        return
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(c2, c1, alternative="greater")
    print("키워드 값 중앙: ①없음 %+.3f · ②있음 %+.3f · 분리 AUC **%.3f** (p=%.4f)"
          % (np.median(c1), np.median(c2), u / (len(c1) * len(c2)), p))
    print("\n%-9s %-9s %-9s %s" % ("문턱", "①정답률", "②정답률", "균형정확도"))
    best = None
    for t in np.percentile(c1 + c2, [10, 25, 50, 75, 90]):
        a1 = float(np.mean([v < t for v in c1]))
        a2 = float(np.mean([v >= t for v in c2]))
        ba = (a1 + a2) / 2
        print("%-9.3f %-9.2f %-9.2f %.3f %s"
              % (t, a1, a2, ba, "**우연 초과**" if ba > 0.5 else ""))
        if best is None or ba > best[0]:
            best = (ba, t, a1, a2)
    print("→ 최고 균형정확도 **%.3f** (문턱 %.3f · ① %.2f · ② %.2f)" % best)
    print("  (대조: SceneDiff 조립 타임라인 0.807 · 방해 40개에서 0.690)")


if __name__ == "__main__":
    main()
