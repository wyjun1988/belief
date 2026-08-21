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
    sel = [r for r in rows if r["s_before"] >= args.cond2[0]]
    dc = np.array([r["drop_ctl"] for r in sel]); dt = np.array([r["drop_tst"] for r in sel])
    b = binomtest(int((dt > dc).sum()), int((dt != dc).sum()), 0.5, alternative="greater")
    print("\n부호검정(동률 제외) p=%.3g" % b.pvalue)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
