#!/usr/bin/env python3
"""이동 이벤트 로그 내보내기 — home-jepa `gt_moves` 스키마.

    $P scripts/export_events.py --seq <name> --graph graph_gtdepth.json

산출: data/seq/<name>/events_<graph>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.graph.events import move_events, summarize   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth.json")
    ap.add_argument("--all-placements", action="store_true",
                    help="운반 중 구간까지 이벤트로 (기본: 정지 배치 간 전이만)")
    ap.add_argument("--min-distance", type=float, default=0.3)
    ap.add_argument("--mover", action="store_true",
                    help="옮긴이 추정을 켠다 (기본 꺼짐 — ADT 에 라벨이 없어 채점 불가)")
    ap.add_argument("--gt-skeletons", action="store_true",
                    help="ADT GT 스켈레톤으로 옮긴이 신원까지 (평가용 상한, 현실엔 없는 정보)")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)

    ev = move_events(g, poses, stable_only=not args.all_placements,
                     min_distance=args.min_distance, enable_mover=args.mover or args.gt_skeletons,
                     use_gt_skeletons=args.gt_skeletons)
    s = summarize(ev)
    print("%s [%s]" % (g["sequence"], g.get("depth_dir")))
    print("  이벤트 %d건, 방 전환 %d건" % (s["n_events"], s["n_room_changes"]))
    if args.mover or args.gt_skeletons:
        print("  옮긴이(추정, 채점 불가): %s" % s["movers"])
    print("  방 전환: %s" % s["room_transitions"])
    print("  미관측 구간 중앙값: %s 프레임" % s["median_unobserved_gap_frames"])
    print()
    for e in ev:
        print("  t=%-5d %-26s %-8s → %-8s   %s → %s   [%s]"
              % (e["t"], (e["obj"] or "")[:26], e["src_room"] or "-", e["dst_room"] or "-",
                 (e["src"] or "-")[:18], (e["dst"] or "-")[:18], e["mover"] or "미상"))

    out = os.path.join(seq_dir, "events_%s.json" % args.graph.replace("graph_", "").replace(".json", ""))
    json.dump({"sequence": g["sequence"], "graph": args.graph,
               "depth_dir": g.get("depth_dir"), "fps": 10.0,
               "summary": s, "events": ev}, open(out, "w"), ensure_ascii=False, indent=1)
    print("\n→ %s" % out)


if __name__ == "__main__":
    main()
