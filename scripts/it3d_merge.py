#!/usr/bin/env python3
"""두 대에서 나눠 잰 IT3DEgo 부재 결과를 합쳐 채점한다.

    $P scripts/it3d_merge.py a.json b.json --out absence.json

⚠️ 판정은 **짝지은 부호검정**이다 — 사건마다 같은 물체·같은 자리·같은 검출기로
대조(물체가 아직 있음)와 검정(떠남)을 만들었으므로, 물체 간 절대값을 비교하지
않는다. ㉜ 에서 물린 어휘 편향 함정이 구조적으로 안 생긴다.
"""
import argparse, json

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, binomtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cond2", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.20])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for f in args.files:
        try:
            rows += json.load(open(f))
        except Exception as e:
            print("⚠️ %s 못 읽음: %s" % (f, e))
    if not rows:
        print("결과 없음")
        return
    print("사건 %d · 영상 %d · 물체 %d"
          % (len(rows), len({r["video"] for r in rows}),
             len({(r["video"], r["obj"]) for r in rows})))
    print("\n%-10s %6s %10s %10s %8s %10s %10s"
          % ("조건②", "사건", "대조 하락", "검정 하락", "떠남>있음", "Wilcoxon", "비짝 AUC"))
    for c in args.cond2:
        sel = [r for r in rows if r["s_before"] >= c]
        if len(sel) < 6:
            continue
        dc = np.array([r["drop_ctl"] for r in sel])
        dt = np.array([r["drop_tst"] for r in sel])
        w = int((dt > dc).sum())
        _, pw = wilcoxon(dt, dc, alternative="greater")
        u, _ = mannwhitneyu(dt, dc, alternative="greater")
        print("  ≥%.2f    %6d %+10.4f %+10.4f  %4d/%-4d %10.3g %10.3f"
              % (c, len(sel), np.median(dc), np.median(dt), w, len(sel), pw,
                 u / (len(dt) * len(dc))))
    # ── 진단 ① 새 자리까지의 거리별
    d = [r for r in rows if r.get("dist") == r.get("dist") and r.get("dist") is not None]
    if len(d) > 20:
        q = np.quantile([r["dist"] for r in d], [0, .25, .5, .75, 1.0])
        print("\n거리별 (떠난 자리 → 새 자리의 3D 거리)")
        print("  %-14s %6s %10s %10s %10s" % ("구간(m)", "사건", "대조 하락", "검정 하락", "떠남>있음"))
        for i in range(4):
            sub = [r for r in d if q[i] <= r["dist"] <= q[i + 1]]
            if len(sub) < 5:
                continue
            dc = np.array([r["drop_ctl"] for r in sub]); dt = np.array([r["drop_tst"] for r in sub])
            print("  %5.2f ~ %-6.2f %6d %+10.4f %+10.4f  %4d/%-4d"
                  % (q[i], q[i + 1], len(sub), np.median(dc), np.median(dt),
                     int((dt > dc).sum()), len(sub)))

    # ── 진단 ② 검정 창에서 물체가 **GT 로 안 보이는** 프레임만
    g = [r for r in rows if r.get("s_test_gone") is not None]
    print("\n검정 창을 GT 가시성으로 가르면 (사건 %d 중 %d 에 '안 보이는' 프레임 있음)"
          % (len(rows), len(g)))
    if len(g) >= 6:
        dg = np.array([r["s_before"] - r["s_test_gone"] for r in g])
        dcg = np.array([r["drop_ctl"] for r in g])
        _, pg = wilcoxon(dg, dcg, alternative="greater")
        print("  안 보이는 프레임만 하락 %+.4f vs 대조 %+.4f · 떠남>있음 %d/%d · p=%.3g"
              % (np.median(dg), np.median(dcg), int((dg > dcg).sum()), len(g), pg))
    v = [r for r in rows if r.get("s_test_vis") is not None]
    if len(v) >= 6:
        dv = np.array([r["s_before"] - r["s_test_vis"] for r in v])
        dcv = np.array([r["drop_ctl"] for r in v])
        print("  아직 보이는 프레임만 하락 %+.4f vs 대조 %+.4f · 떠남>있음 %d/%d"
              % (np.median(dv), np.median(dcv), int((dv > dcv).sum()), len(v)))
    vf = np.array([r.get("vis_frac", 0.0) for r in rows])
    print("  검정 창에서 물체가 여전히 보이는 프레임 비율 중앙 **%.0f%%**" % (100 * np.median(vf)))

    sel = [r for r in rows if r["s_before"] >= args.cond2[0]]
    dc = np.array([r["drop_ctl"] for r in sel]); dt = np.array([r["drop_tst"] for r in sel])
    b = binomtest(int((dt > dc).sum()), int((dt != dc).sum()), 0.5, alternative="greater")
    print("\n부호검정(동률 제외) p=%.3g" % b.pvalue)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
