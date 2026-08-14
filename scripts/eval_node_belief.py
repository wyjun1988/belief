#!/usr/bin/env python3
"""조각을 합치지 않고 belief 질의 — "그 물건 지금 어느 방에 있어?"

    $P scripts/eval_node_belief.py --seq <name> --graphs graph_sam graph_sam_relinked --ref gtdepth

후보 = 질의와 같은 종류의 모든 노드, 답 = 그중 가장 최근 배치의 구역.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.eval.node_belief import run   # noqa: E402
from kx.eval.room_belief import load_regions   # noqa: E402
from kx.graph.regions import assign   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--ref", default="gtdepth")
    ap.add_argument("--tick", type=int, default=50)
    ap.add_argument("--by", default="category", choices=["category", "gt", "appearance"])
    ap.add_argument("--emb", default="sam_daaam/track_emb.npz")
    ap.add_argument("--sim-min", type=float, default=0.9)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    meta = json.load(open(os.path.join(seq_dir, "graph_%s.json" % args.ref)))["regions"]
    ref = load_regions(np.load(os.path.join(seq_dir, "regions_%s.npz" % args.ref)),
                       meta["zone_names"], meta["up"])

    def zone_of(p):
        return assign(ref, p)[1]

    emb = None
    if args.by == "appearance":
        z = np.load(os.path.join(seq_dir, args.emb))
        emb = {int(i): e for i, e in zip(z["ids"], z["emb"])}
        print("트랙 임베딩 %d개" % len(emb))

    hdr = "%-30s %-10s %-13s %-10s %-9s %s" % (
        "그래프", "정확도", "이동후 정확도", "질의수", "후보수", "위치오차")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for name in args.graphs:
        p = os.path.join(seq_dir, name if name.endswith(".json") else name + ".json")
        if not os.path.exists(p):
            print("%-30s (없음)" % name)
            continue
        g = json.load(open(p))
        r = run(g, gt, ref, zone_of, tick=args.tick, by=args.by, emb=emb,
                sim_min=args.sim_min)
        r["graph"] = name
        out.append(r)
        f = lambda v: "  -  " if v is None else "%.3f" % v      # noqa: E731
        print("%-30s %-10s %-13s %-10d %-9.0f %.3f"
              % (name, f(r["accuracy"]), f(r["accuracy_after_move"]),
                 r["n_queries"], r["median_candidates"] or 0, r["median_err_m"] or 0))
    p = os.path.join(seq_dir, "node_belief_eval.json")
    json.dump([{k: v for k, v in r.items() if k != "rows"} for r in out], open(p, "w"), indent=1)
    print("\n→ %s" % p)


if __name__ == "__main__":
    main()
