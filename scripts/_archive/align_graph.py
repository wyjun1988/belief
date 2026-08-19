#!/usr/bin/env python3
"""DA3 포즈로 만든 그래프를 GT 좌표계로 sim3 정합한다 — **채점 전용**.

DA3 를 포즈 없이 돌리면 재구성 전체가 임의의 sim3(회전·평행이동·스케일) 만큼 GT 와
다르다. 그건 오차가 아니라 **게이지 자유도**다 — 세계 축이 통째로 돌아가 있어도 방·거리·
관계는 그대로다. 그래서 좌표를 GT 와 비교하려면 궤적으로 sim3 를 풀어 먼저 맞춰야 한다.

정합 파라미터는 **궤적에서만** 뽑는다(물체 위치는 쓰지 않는다). 파이프라인은 이 정합을
쓰지 않는다 — 오직 숫자를 GT 와 같은 자로 재기 위한 것이다.

    $P scripts/align_graph.py --seq <name> --graph graph_da3.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.pose_stitch import robust_umeyama   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_da3.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    est = np.loadtxt(os.path.join(seq_dir, g.get("pose_file", "pose/poses_da3.txt"))).reshape(-1, 4, 4)
    gt = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    n = min(len(est), len(gt))
    m = np.linalg.norm(est[:n, :3, 3], axis=1) > 0
    s, R, t, resid = robust_umeyama(est[:n][m][:, :3, 3], gt[:n][m][:, :3, 3])
    ate = np.linalg.norm((s * (R @ est[:n][m][:, :3, 3].T)).T + t - gt[:n][m][:, :3, 3], axis=1)
    print("sim3: scale %.4f  ATE 중앙 %.3f m  p90 %.3f m" % (s, np.median(ate), np.percentile(ate, 90)))

    def T(p):
        return (s * (R @ np.asarray(p, float))) + t

    for o in list(g["objects"].values()) + list((g.get("agents") or {}).values()):
        for pl in o.get("placements", []):
            for k in ("position", "vox_centroid", "vox_bbox_center"):
                if pl.get(k):
                    pl[k] = np.round(T(pl[k]), 4).tolist()
        for c in o.get("changes", []):
            for k in ("from", "to"):
                c[k] = np.round(T(c[k]), 4).tolist()
        if o.get("trajectory"):
            o["trajectory"] = [[r[0]] + np.round(T(r[1:4]), 3).tolist() for r in o["trajectory"]]

    g["aligned_to_gt"] = {"scale": s, "R": R.tolist(), "t": t.tolist(),
                          "ate_median_m": float(np.median(ate)),
                          "note": "채점 전용 sim3. 파이프라인은 이 정합을 쓰지 않는다."}
    out = args.out or args.graph.replace(".json", "_aligned.json")
    json.dump(g, open(os.path.join(seq_dir, out), "w"))
    print("→ %s" % out)


if __name__ == "__main__":
    main()
