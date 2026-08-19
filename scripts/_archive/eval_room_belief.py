#!/usr/bin/env python3
"""belief 평가 (방 단위) — "지금 어느 방에 있나".

    $P scripts/eval_room_belief.py --seq <name> --graphs graph_gtdepth graph_t23 graph_da3lc_aligned

정답은 기준 구역지도(GT 포즈·GT 뎁스)로 매긴다. 각 그래프에 대해 두 수치를 낸다:
  localization  기준 지도로 우리 belief 위치를 매김 → **위치 오차만**
  end_to_end    그 그래프 자신의 구역지도로 매김   → 위치 + 구역분할 오차 (실제 성능)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.eval.room_belief import load_regions, run   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _regions(seq_dir, tag):
    p = os.path.join(seq_dir, "regions_%s.npz" % tag)
    g = os.path.join(seq_dir, "graph_%s.json" % tag)
    if not (os.path.exists(p) and os.path.exists(g)):
        return None
    meta = json.load(open(g))["regions"]
    return load_regions(np.load(p), meta["zone_names"], meta["up"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--ref", default="gtdepth", help="기준 구역지도 태그")
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--tick", type=int, default=50)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    ref = _regions(seq_dir, args.ref)
    if ref is None:
        sys.exit("기준 구역지도가 없다: regions_%s.npz" % args.ref)

    hdr = "%-28s %-11s %-11s %-13s %-12s %-9s" % (
        "그래프", "전체", "방바뀜", "방바뀜(기준)", "베이스라인", "위치오차")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for name in args.graphs:
        p = os.path.join(seq_dir, name if name.endswith(".json") else name + ".json")
        if not os.path.exists(p):
            print("%-30s (없음)" % name)
            continue
        g = json.load(open(p))
        # 자기 구역지도: aligned 그래프는 원본 태그의 지도를 쓴다(좌표가 정합됐으므로
        # 기준 지도와 같은 프레임이다 → end-to-end 는 구역분할 차이만 남는다)
        tag = name.replace("graph_", "").replace("_aligned", "").replace(".json", "")
        own = _regions(seq_dir, tag) if "_aligned" not in name else ref
        r = run(g, gt, ref, own_reg=own, tick=args.tick)
        r["graph"] = name
        out.append(r)
        f = lambda v: "  -  " if v is None else "%.3f" % v      # noqa: E731
        print("%-28s %-11s %-11s %-13s %-12s %-9s"
              % (name, f(r["end_to_end"]), f(r["changed_end_to_end"]),
                 f(r["changed_localization"]), f(r["changed_baseline"]),
                 f(r["belief_err_median_m"])))
    if out:
        print("\n질의 %d개 / 물체 %d개 · 그중 **방이 바뀐 질의 %d개**"
              % (out[0]["n_queries"], out[0]["n_objects"], out[0]["n_changed"]))
        if out[0]["n_changed"] == 0:
            print("⚠️  방이 바뀐 질의가 0개다 — 이 시퀀스로는 방 belief 를 평가할 수 없다.")
    p = os.path.join(seq_dir, "room_belief_eval.json")
    json.dump([{k: v for k, v in r.items() if k != "rows"} for r in out], open(p, "w"), indent=1)
    print("→ %s" % p)


if __name__ == "__main__":
    main()
