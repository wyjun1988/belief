#!/usr/bin/env python3
"""뎁스 애블레이션 표 — 정합이 실제로 값을 하는지 한 화면에서 본다.

    $P scripts/eval_depth.py --seq <name> --dirs depth_none depth_t2 depth_t23

각 행: 정확도(AbsRel/δ1) · 시간 일관성(TAE) · **정적 물체 3D 산포**.
마지막 열이 P3 의 '정적 물체 오탐율'과 직결된다 — 뎁스가 흔들리면 그래프가 흔들린다.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.metrics import depth_errors, dispersion, static_centroids, tae   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_png_m(path):
    return np.array(Image.open(path)).astype(np.float32) / 1000.0


def evaluate(seq_dir, sub, every, K, poses, static_locals, n):
    acc, taes, tracks = defaultdict(list), [], defaultdict(list)
    d_prev, i_prev = None, None
    gt_dir = os.path.join(seq_dir, "gt", "depth")
    seg_dir = os.path.join(seq_dir, "gt", "seg")

    for i in range(0, n, every):
        p = os.path.join(seq_dir, sub, "%06d.png" % i)
        if not os.path.exists(p):
            continue
        d = load_png_m(p)

        g = os.path.join(gt_dir, "%06d.png" % i)
        if os.path.exists(g):
            e = depth_errors(d, load_png_m(g))
            if e:
                for k, v in e.items():
                    acc[k].append(v)

        if d_prev is not None and i - i_prev <= every:
            t = tae(d_prev, d, poses[i_prev], poses[i], K)
            if t is not None:
                taes.append(t)
        d_prev, i_prev = d, i

        s = os.path.join(seg_dir, "%06d.png" % i)
        if os.path.exists(s) and static_locals:
            for lid, c in static_centroids(d, np.array(Image.open(s)), K, poses[i],
                                           static_locals).items():
                tracks[lid].append(c)

    row = {"dir": sub}
    for k in ("absrel", "rmse", "delta1", "bias"):
        row[k] = float(np.mean(acc[k])) if acc[k] else None
    row["tae"] = float(np.mean(taes)) if taes else None
    row["static_disp"] = dispersion(tracks)
    al = os.path.join(seq_dir, sub + "_align.json")
    if os.path.exists(al):
        row["scale_rel_std"] = json.load(open(al)).get("scale_rel_std")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--dirs", nargs="+", default=["depth_none", "depth_t2", "depth_t23"])
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default=None, help="JSON 저장 경로")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"])
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    n = len(poses)

    static_locals = []
    ids_p = os.path.join(seq_dir, "gt", "seg_ids.json")
    if os.path.exists(ids_p):
        ids = json.load(open(ids_p))
        static_locals = [int(k) for k, v in ids.items() if v.get("motion_type") == "STATIC"]

    rows = []
    for sub in args.dirs:
        if not os.path.isdir(os.path.join(seq_dir, sub)):
            print("(없음) %s" % sub)
            continue
        rows.append(evaluate(seq_dir, sub, args.every, K, poses, static_locals, n))

    hdr = "%-14s %8s %8s %8s %7s %8s %10s %10s" % (
        "dir", "AbsRel", "RMSE", "d<1.25", "bias", "TAE", "staticσ(m)", "scaleσ")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        sd = r["static_disp"]
        print("%-14s %8s %8s %8s %7s %8s %10s %10s" % (
            r["dir"],
            "%.4f" % r["absrel"] if r["absrel"] is not None else "-",
            "%.3f" % r["rmse"] if r["rmse"] is not None else "-",
            "%.4f" % r["delta1"] if r["delta1"] is not None else "-",
            "%.3f" % r["bias"] if r["bias"] is not None else "-",
            "%.4f" % r["tae"] if r["tae"] is not None else "-",
            "%.4f" % sd["median_m"] if sd else "-",
            "%.4f" % r["scale_rel_std"] if r.get("scale_rel_std") is not None else "-"))
    print()

    out = args.out or os.path.join(seq_dir, "depth_eval.json")
    json.dump({"sequence": os.path.basename(seq_dir), "every": args.every, "rows": rows},
              open(out, "w"), indent=1)
    print("→ %s" % out)


if __name__ == "__main__":
    main()
