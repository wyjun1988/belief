#!/usr/bin/env python3
"""씬그래프에 자연어에 가까운 질의를 던진다 — "지금 액자 어디 있어?"

    $P scripts/query_graph.py --seq <name> --graph graph_gtdepth.json --obj PictureFrame
    $P scripts/query_graph.py --seq <name> --graph graph_gtdepth.json --obj PictureFrame --t 400

`--t` 를 안 주면 **영상 끝 시점**을 묻는다(= 사용자가 원한 형태).
답은 세 층으로 낸다: 구역(거실/부엌/…) · 지지 가구(무엇 위에) · 3D 좌표.
그리고 같은 시점의 GT 를 나란히 찍어 맞았는지 바로 보이게 한다.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def believe(o, t):
    """t 시점의 믿음 — t 를 덮는 배치, 없으면 마지막으로 본 자리(관측 여부도 함께)."""
    cur, seen = None, False
    for pl in o["placements"]:
        if pl["start_frame"] <= t <= pl["end_frame"]:
            return pl, True
        if pl["end_frame"] < t:
            cur = pl
    return cur, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth.json")
    ap.add_argument("--obj", required=True, help="물체 이름 부분일치")
    ap.add_argument("--t", type=int, default=None, help="질의 프레임 (기본: 영상 끝)")
    ap.add_argument("--stable-only", action="store_true",
                    help="운반 중 배치는 무시하고 마지막 정지 자리로 답한다")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    t = args.t if args.t is not None else g["n_frames"] - 1

    hits = [o for o in g["objects"].values() if args.obj.lower() in (o["name"] or "").lower()]
    if not hits:
        sys.exit("그런 이름의 물체가 그래프에 없다: %s" % args.obj)

    print("질의 시점: 프레임 %d (%.1f초, 전체 %d프레임)\n" % (t, t / 10.0, g["n_frames"]))
    for o in hits:
        pls = [p for p in o["placements"] if p["stable"]] if args.stable_only else o["placements"]
        if not pls:
            continue
        pl, seen = believe({"placements": pls}, t)
        if pl is None:
            print("%-26s : 아직 관측 전" % o["name"])
            continue
        rec = gt.get(str(o["instance_id"]))
        gtp = np.array(rec["positions"][min(t, len(rec["positions"]) - 1)]) if rec else None
        err = float(np.linalg.norm(np.array(pl["position"]) - gtp)) if gtp is not None else None

        print("● %s (%s)" % (o["name"], o["category"]))
        print("   답:   %s / %s 위 / %s"
              % (pl["zone"] or "구역미상", pl["support"] or "지지물 없음",
                 np.round(pl["position"], 2)))
        print("   근거: 프레임 %d–%d 에서 %d회 관측 %s"
              % (pl["start_frame"], pl["end_frame"], pl["n_obs"],
                 "(질의 시점에 보고 있음)" if seen else "(마지막으로 본 자리 유지)"))
        if gtp is not None:
            print("   GT:   %s   → 오차 %.3f m" % (np.round(gtp, 2), err))
        print()


if __name__ == "__main__":
    main()
