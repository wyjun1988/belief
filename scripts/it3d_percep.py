#!/usr/bin/env python3
"""**지각층 단독 검증** — OWLv2 가 "이 프레임에 그 물체가 있나" 를 맞히는가.

    $P scripts/it3d_percep.py --cache data/it3dego/cache_all --ann …

⚠️ **왜 이걸 따로 재는가.** ㉟ 에서 "병목은 지각" 이라고 결론냈는데, 근거가
기권률 28~37% 와 `s_before` 중앙 0.151 이었다. **둘 다 파이프라인을 통과한 뒤의
숫자**다 — 검색·장소게이트·창 선택이 다 끼어 있다. 검출기 자체를 GT 와 맞대본 적이
없으므로 그 결론은 **측정이 아니라 추론**이었다.

OWLv2 는 LVIS 제로샷에서 검증된 모델이다. 다만 검증 조건은 웹 사진이고 우리가
먹이는 것은 에고센트릭 프레임(모션 블러·작은 물체·`pick tool` 같은 어휘)이다.
IT3DEgo 에는 **2D bbox GT** 가 있으니 직접 가를 수 있다.

    양성 = GT bbox 가 --tol 안에 있는 프레임 (물체가 보인다)
    음성 = 그 밖의 프레임
    점수 = 그 물체 어휘에 대한 OWL 최대 검출점수 (캐시된 것)

이게 **1 근처면 지각은 멀쩡하고 파이프라인이 문제**, 0.5 근처면 지각이 진짜 병목이다.
비교용으로 CLIP 텍스트-이미지 유사도도 같이 잰다.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.it3d_absence import base_label, load_ann              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--tol", type=float, default=2e6, help="bbox 시각 허용오차(100ns)")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.10, help="조건② 문턱 — 이 위 비율도 본다")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from scipy.stats import mannwhitneyu
    from scripts.absence_evidence import clip_text

    rows = []
    for f in sorted(glob.glob(os.path.join(args.cache, "*.all.npz"))):
        vn = os.path.basename(f)[:-len(".all.npz")]
        ad = os.path.join(args.ann, vn)
        if not os.path.isdir(ad):
            continue
        z = np.load(f, allow_pickle=True)
        ts, E, S = z["ts"], z["emb"], z["owl"]
        labs, segs, box = load_ann(ad)
        words = [base_label(l) for l in labs]
        dup = {w for w in words if words.count(w) > 1}
        Q = clip_text(words, args.device)
        for oi, w in enumerate(words):
            bts = box.get(oi, np.array([], np.int64))
            if len(bts) == 0:
                continue
            vis = np.array([bool(np.any(np.abs(bts - t) <= args.tol)) for t in ts])
            if vis.sum() < args.min_pos or (~vis).sum() < args.min_pos:
                continue
            so, sc = S[:, oi], E @ Q[oi]
            u1, p1 = mannwhitneyu(so[vis], so[~vis], alternative="greater")
            u2, _ = mannwhitneyu(sc[vis], sc[~vis], alternative="greater")
            rows.append(dict(video=vn, obj=labs[oi], word=w, dup=w in dup,
                             n_vis=int(vis.sum()), n_not=int((~vis).sum()),
                             auc_owl=float(u1 / (vis.sum() * (~vis).sum())),
                             auc_clip=float(u2 / (vis.sum() * (~vis).sum())),
                             p=float(p1),
                             med_vis=float(np.median(so[vis])),
                             med_not=float(np.median(so[~vis])),
                             hit_vis=float((so[vis] >= args.thr).mean()),
                             hit_not=float((so[~vis] >= args.thr).mean())))
        print("  %-20s 누적 %d" % (vn, len(rows)), flush=True)

    if not rows:
        print("표본 없음"); return
    A = np.array([r["auc_owl"] for r in rows]); C = np.array([r["auc_clip"] for r in rows])
    ok = [r for r in rows if not r["dup"]]                       # 조건①
    print("\n물체 %d개 (조건① 통과 %d) · 영상 %d"
          % (len(rows), len(ok), len({r["video"] for r in rows})))
    print("\n**프레임에 그 물체가 있나 — 검출기 단독 AUC**")
    for nm, v in (("OWLv2", A), ("CLIP 유사도", C)):
        print("  %-12s 중앙 **%.3f**  분위 [%.3f %.3f %.3f]  · 0.7 이상 %.0f%% · 0.5 이하 %.0f%%"
              % (nm, np.median(v), *np.quantile(v, [.25, .5, .75]),
                 100 * (v >= .7).mean(), 100 * (v <= .5).mean()))
    B = np.array([r["auc_owl"] for r in ok])
    print("  조건① 통과분 OWL 중앙 %.3f" % np.median(B))
    print("\n**문턱 %.2f 에서** 보일 때 넘는 비율 %.3f · 안 보일 때 넘는 비율 %.3f"
          % (args.thr, np.mean([r["hit_vis"] for r in rows]),
             np.mean([r["hit_not"] for r in rows])))
    print("  검출도 중앙 — 보일 때 %.3f · 안 보일 때 %.3f"
          % (np.median([r["med_vis"] for r in rows]),
             np.median([r["med_not"] for r in rows])))
    rows.sort(key=lambda r: -r["auc_owl"])
    print("\n%-22s %6s %6s %7s %7s" % ("물체", "AUC", "CLIP", "보일때", "안보일때"))
    for r in rows[:6]:
        print("  %-20s %6.3f %6.3f %7.3f %7.3f"
              % (r["word"][:20], r["auc_owl"], r["auc_clip"], r["med_vis"], r["med_not"]))
    print("  ...")
    for r in rows[-6:]:
        print("  %-20s %6.3f %6.3f %7.3f %7.3f"
              % (r["word"][:20], r["auc_owl"], r["auc_clip"], r["med_vis"], r["med_not"]))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
