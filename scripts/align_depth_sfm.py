#!/usr/bin/env python3
"""SfM 랜드마크를 뎁스 앵커로 — 픽셀-정확 대응으로 DA3 뎁스를 정합한다.

    $P scripts/align_depth_sfm.py --seq <name> --pose pose/poses_sfm_hybrid.txt --out depth_sfmlm

MPS 앵커 투영 정합은 **픽셀 정확도의 포즈**를 요구한다 — ATE 0.17 m 도 수십 px
오투영이라 짝이 틀어지고, 모델 포즈에서 AbsRel 이 0.37→2.0~2.6 으로 **악화**했다
(2026-08-16 실측, 게이트 통과 32~37/918). 여기서는 대응을 투영으로 만들지 않는다:
SfM 랜드마크의 관측은 (픽셀, 3D)가 **구성상 정확히** 짝이다 — 그 픽셀의 마스크
중심을 삼각측량해 만든 점이기 때문이다. 프레임당 ~7점이지만 역깊이 아핀은
2파라미터라 충분하고, 빈 프레임은 T3 시간 평활이 메운다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.depth.align import apply_affine, smooth_sequence      # noqa: E402
from scripts.incremental_sfm import (MIN_PARALLAX, MIN_TRI_VIEWS,  # noqa: E402
                                     accept_point)
from scripts.refine_bootstrap import load_obs, triangulate    # noqa: E402


def _write(out_dir, i, d):
    mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(os.path.join(out_dir, "%06d.png" % i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--pose", default="pose/poses_sfm_hybrid.txt")
    ap.add_argument("--raw", default="depth_raw")
    ap.add_argument("--out", default="depth_sfmlm")
    ap.add_argument("--seg", default="gt/seg")
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--max-extent", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--min-pts", type=int, default=5)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(sd, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    W, H = cam["width"], cam["height"]
    poses = np.loadtxt(os.path.join(sd, args.pose)).reshape(-1, 4, 4)
    ids = json.load(open(os.path.join(sd, args.seg_ids)))
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
    reg = [i for i in range(len(poses)) if not np.allclose(poses[i], np.eye(4))]

    keep = set()
    for local, m in ids.items():
        rec = gt.get(str(m.get("gt_instance") or m.get("instance_id")))
        if rec and rec["motion_type"] == "static" and not rec.get("moves") \
                and rec.get("extent_m") and max(rec["extent_m"]) <= args.max_extent:
            keep.add(int(local))
    obs = load_obs(sd, args.seg, ids, keep, 1, W, H, 0.0)

    # 랜드마크 재구성 — SfM 과 같은 검증 게이트로 삼각측량
    land = {}
    for lid in keep:
        vf = [f for f in reg if lid in obs.get(f, {})]
        views = [(poses[f], obs[f][lid][:2]) for f in vf]
        if len(views) < MIN_TRI_VIEWS:
            continue
        C = np.array([T[:3, 3] for T, _ in views])
        if np.linalg.norm(C.max(0) - C.min(0)) < MIN_PARALLAX:
            continue
        X = triangulate(views, K)
        if accept_point(X, views, K):
            land[lid] = X
    print("랜드마크 %d개 (등록 프레임 %d)" % (len(land), len(reg)))

    raw_dir = os.path.join(sd, args.raw)
    out_dir = os.path.join(sd, args.out)
    os.makedirs(out_dir, exist_ok=True)
    n = len(poses)
    A, B, NI = np.zeros(n), np.zeros(n), np.zeros(n, int)
    for i in reg:
        pairs = []
        T = poses[i]
        R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
        raw = None
        for lid, (u, v, *_) in obs.get(i, {}).items():
            if lid not in land:
                continue
            Xc = R @ land[lid] + t
            if Xc[2] < 0.2:
                continue
            if raw is None:
                raw = np.load(os.path.join(raw_dir, "%06d.npy" % i)).astype(np.float32)
            # 마스크 중심 주변 3x3 중앙값 — 단일 픽셀 노이즈 방지
            ui, vi = int(round(u)), int(round(v))
            patch = raw[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
            d_raw = float(np.median(patch[patch > 0])) if (patch > 0).any() else 0
            if d_raw <= 0:
                continue
            pairs.append((1.0 / d_raw, 1.0 / Xc[2]))
        if len(pairs) < args.min_pts:
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        # 점이 적으므로 무거운 RANSAC 대신 반복 최소자승 + 잔차 절사
        for _ in range(3):
            a, b = np.polyfit(x, y, 1)
            r = np.abs(a * x + b - y)
            m = r <= max(3 * np.median(r), 1e-6)
            if m.sum() < args.min_pts:
                break
            x, y = x[m], y[m]
        else:
            pass
        if len(x) >= args.min_pts and a > 0:
            A[i], B[i], NI[i] = a, b, len(x)

    fit = int((NI > 0).sum())
    print("아핀 적합 %d/%d 프레임 (중앙 앵커 %d점)"
          % (fit, n, int(np.median(NI[NI > 0])) if fit else 0))
    A, B = smooth_sequence(A, B, NI, lam=args.lam, min_inliers=args.min_pts)

    have = sorted(int(f.split(".")[0]) for f in os.listdir(raw_dir) if f.endswith(".npy"))
    for i in have:
        raw = np.load(os.path.join(raw_dir, "%06d.npy" % i)).astype(np.float32)
        _write(out_dir, i, apply_affine(raw, A[i], B[i]))
    json.dump({"mode": "sfm_landmark", "pose": args.pose, "fit": fit,
               "landmarks": len(land)},
              open(os.path.join(sd, args.out + "_align.json"), "w"))
    print("→ %s" % out_dir)


if __name__ == "__main__":
    main()
