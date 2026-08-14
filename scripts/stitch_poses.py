#!/usr/bin/env python3
"""DA3 윈도우 포즈 → 전역 궤적 + 스케일 보정된 뎁스. **기기 없는 조건의 입구.**

    $P scripts/stitch_poses.py --seq <name>

하는 일:
  1. 윈도우별 sim3 를 겹치는 프레임으로 이어붙여 하나의 궤적으로 (`pose/poses_da3.txt`)
  2. 전역 스케일 하나를 착용자 머리 높이(1.55m) 사전지식으로 고정
  3. 각 프레임 뎁스에 그 윈도우의 누적 스케일을 곱해 `depth_da3/` 에 저장

그리고 GT 궤적이 있으면 **sim3 정합 후 ATE** 를 찍는다 — 채점용이지 파이프라인 입력이 아니다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.pose_stitch import robust_umeyama, run   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--pose-dir", default="poses_raw_np")
    ap.add_argument("--raw", default="depth_raw_np")
    ap.add_argument("--out-depth", default="depth_da3")
    ap.add_argument("--wearer-height", type=float, default=1.55)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    meta = run(seq_dir, pose_dir=args.pose_dir, wearer_height=args.wearer_height)
    print("윈도우 %d개, 프레임 %d/%d 커버" % (meta["n_windows"], meta["frames_covered"], meta["n_frames"]))
    print("윈도우 스케일: 중앙 %.3f  범위 %.3f~%.3f"
          % (np.median(meta["window_scales"]), min(meta["window_scales"]), max(meta["window_scales"])))
    r = [x for x in meta["stitch_residual_m"] if x == x]
    print("이어붙이기 잔차: 중앙 %.4f  최대 %.4f  (겹침 최소 %d프레임)"
          % (np.median(r), max(r), min(meta["overlaps"][1:] or [0])))
    print("전역 스케일 %.4f  (머리높이 사전지식 %.2fm, 높이변동 %.2fm)"
          % (meta["global_scale"], meta["wearer_height_prior_m"], meta["head_height_span_m"]))

    # --- 뎁스에 같은 스케일 적용 ---
    sc = np.array(meta["depth_scale"])
    raw_dir, out_dir = os.path.join(seq_dir, args.raw), os.path.join(seq_dir, args.out_depth)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".npy"):
            continue
        i = int(os.path.splitext(f)[0])
        if i >= len(sc) or sc[i] == 0:
            continue
        d = np.load(os.path.join(raw_dir, f)).astype(np.float32) * sc[i]
        Image.fromarray(np.clip(d * 1000.0, 0, 65535).astype(np.uint16)).save(
            os.path.join(out_dir, "%06d.png" % i))
        n += 1
    print("뎁스 %d장 → %s" % (n, args.out_depth))

    # --- 채점: GT 궤적 대비 ATE (파이프라인은 이걸 쓰지 않는다) ---
    gt_p = os.path.join(seq_dir, "pose", "poses.txt")
    est = np.loadtxt(os.path.join(seq_dir, "pose", "poses_da3.txt")).reshape(-1, 4, 4)
    if os.path.exists(gt_p):
        gt = np.loadtxt(gt_p).reshape(-1, 4, 4)
        m = sc > 0
        s, R, t, resid = robust_umeyama(est[m][:, :3, 3], gt[m][:, :3, 3])
        ate = np.linalg.norm((s * (R @ est[m][:, :3, 3].T)).T + t - gt[m][:, :3, 3], axis=1)
        rot = [np.degrees(np.arccos(np.clip(
            (np.trace((R @ est[i, :3, :3]).T @ gt[i, :3, :3]) - 1) / 2, -1, 1)))
            for i in np.flatnonzero(m)]
        L = np.linalg.norm(np.diff(gt[m][:, :3, 3], axis=0), axis=1).sum()
        print("\n[채점] GT 대비 sim3 정합 후")
        print("   ATE 중앙 %.3f m  p90 %.3f m  (궤적길이 %.1f m 의 %.1f%%)"
              % (np.median(ate), np.percentile(ate, 90), L, 100 * np.median(ate) / L))
        print("   회전오차 중앙 %.2f°  p90 %.2f°" % (np.median(rot), np.percentile(rot, 90)))
        print("   정합 스케일 %.4f  ← 1.0 에 가까울수록 머리높이 사전지식이 맞았다는 뜻" % s)


if __name__ == "__main__":
    main()
