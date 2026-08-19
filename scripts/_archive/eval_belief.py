#!/usr/bin/env python3
"""belief 질의 — "지금 이 물체가 어느 표면 위에 있나"를 우리 씬그래프로 답한다.

    $P scripts/eval_belief.py --seq <name> --graph graph_gtdepth.json

home-jepa 의 ADT 규칙(1.6m 최근접 가구 + 히스테리시스 0.10m, 5초 틱)을 그대로 쓴다.
정답은 GT 좌표로 매긴 수용체, 응답은 **우리 그래프의 belief 위치 + 우리가 추정한
가구 위치**로 매긴 수용체다. 즉 GT 를 지각으로 갈아끼웠을 때 답이 유지되는지를 잰다.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.eval.belief import run   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth.json")
    ap.add_argument("--tick", type=int, default=50, help="질의 간격(프레임). 50=5초")
    ap.add_argument("--furn-pos", default="position",
                    choices=["position", "vox_centroid", "vox_bbox_center"],
                    help="가구 위치 추정량")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]

    r = run(g, gt, tick_frames=args.tick, furn_pos=args.furn_pos)
    print("== %s  (%s, 가구위치=%s)" % (g["sequence"], g.get("depth_dir"), args.furn_pos))
    print("   수용체 %d개, 동적 물체 %d개, 질의 %d건 (%d프레임=%.0f초 간격)"
          % (r["n_receptacles"], r["n_objects"], r["n_queries"], r["tick_frames"],
             r["tick_frames"] / 10))
    print()
    print("   %-26s %s" % ("그래프 belief 정확도", _p(r["graph_acc"])))
    print("   %-26s %s  (n=%d)" % ("  ├ 관측 중", _p(r["graph_acc_observed"]), r["n_observed"]))
    print("   %-26s %s  (n=%d)" % ("  └ 미관측(last-known)", _p(r["graph_acc_unobserved"]),
                                   r["n_unobserved"]))
    print("   %-26s %s  ← 가구만 GT (물체는 우리 추정)" % ("  ※ 가구 위치 GT 대체",
          _p(r["graph_gtfurn_acc"])))
    print("   %-26s %s  ← 베이스라인(움직임 무지)" % ("초기 위치 고정", _p(r["initial_acc"])))
    print("   %-26s %.3f m" % ("belief 위치 오차 중앙", r["belief_err_median_m"]))

    wrong = [x for x in r["rows"] if x["graph"] != x["gt"]]
    if wrong:
        print("\n   틀린 질의 %d건 (상위 8):" % len(wrong))
        for x in sorted(wrong, key=lambda y: -y["err_m"])[:8]:
            print("      %-26s t=%-4d obs=%-5s gt=%-4s ours=%-4s err=%.2fm"
                  % (x["obj"], x["t"], x["observed"], x["gt"], x["graph"], x["err_m"]))

    out = os.path.join(seq_dir, args.graph.replace(".json", "_belief.json"))
    json.dump({k: v for k, v in r.items() if k != "rows"} | {"n_rows": len(r["rows"])},
              open(out, "w"), indent=1)
    print("\n→ %s" % out)


def _p(v):
    return "  -  " if v is None else "%.3f" % v


if __name__ == "__main__":
    main()
