#!/usr/bin/env python3
"""구간별 미터 앵커를 넣은 **전역 BA** 로 지도를 편다 — 루프를 깨지 않고.

    $P scripts/straighten_ba.py --seq <name> --sfm /tmp/poses_inc_e1.txt

경로 적분 방식(straighten_map.py)은 스케일은 폈지만(1.81→1.11) 루프 일관성을
깨서 전역 스티칭 ATE 가 0.76 그대로였다. 여기서는 구간별 바닥 사전 스케일을
**BA 의 거리 구속**으로 넣는다 — 재투영 잔차가 루프를 지키는 동안 다중 앵커가
구간별 스케일 휨을 편다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.depth.pose_stitch import up_from_trajectory                 # noqa: E402
from scripts.incremental_sfm import (MIN_PARALLAX, MIN_TRI_VIEWS,   # noqa: E402
                                     accept_point, ate, local_ba)
from scripts.refine_bootstrap import load_obs, triangulate          # noqa: E402

WEARER_HEIGHT = 1.55


def segment_scales(sd, sfm, reg, K, depth_dir, seg_n=50):
    """구간별 바닥 사전 스케일 — straighten_map 과 같은 추정 + 이상치 보간."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    centers = np.array([sfm[i][:3, 3] for i in reg])
    up = up_from_trajectory(centers)
    segs = [reg[k:k + seg_n] for k in range(0, len(reg), seg_n)]
    if len(segs) > 1 and len(segs[-1]) < seg_n // 2:
        segs[-2] += segs.pop()
    out = []
    for seg in segs:
        pts = []
        for i in seg[::5]:
            p = os.path.join(sd, depth_dir, "%06d.png" % i)
            if not os.path.exists(p):
                continue
            d = np.array(Image.open(p)).astype(np.float32)[::16, ::16] / 1000.0
            v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
            u, v, dd = u.ravel() * 16.0, v.ravel() * 16.0, d.ravel()
            m = (dd > 0.1) & (dd < 12)
            ray = np.stack([(u[m] - cx) / fx, (v[m] - cy) / fy, np.ones(m.sum())], 1)
            T = sfm[i]
            pts.append((ray * dd[m, None]) @ T[:3, :3].T + T[:3, 3])
        if not pts:
            continue
        pts = np.concatenate(pts)
        C = np.array([sfm[i][:3, 3] for i in seg])
        u_ = up if up @ (C.mean(0) - pts.mean(0)) > 0 else -up
        floor = float(np.percentile(pts @ u_, 2.0))
        cam_h = float(np.median(C @ u_) - floor)
        if cam_h < 0.3:
            continue
        out.append((seg, WEARER_HEIGHT / cam_h))
    sc = np.array([x[1] for x in out])
    med = float(np.median(sc))
    bad = np.abs(sc / med - 1) > 0.30
    if bad.any():
        good = ~bad
        mids = np.array([0.5 * (x[0][0] + x[0][-1]) for x in out])
        sc[bad] = np.interp(mids[bad], mids[good], sc[good])
    return [(out[i][0], float(sc[i])) for i in range(len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--sfm", default="/tmp/poses_inc_e1.txt")
    ap.add_argument("--depth", default="depth_sfmlm2")
    ap.add_argument("--seg", default="gt/seg")
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--max-extent", type=float, default=1.0)
    ap.add_argument("--nfev", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(sd, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    W, H = cam["width"], cam["height"]
    gtp = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    sfm = np.loadtxt(args.sfm).reshape(-1, 4, 4)
    reg = sorted(i for i in range(len(sfm)) if not np.allclose(sfm[i], np.eye(4)))
    ids = json.load(open(os.path.join(sd, args.seg_ids)))
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]

    keep = set()
    for local, m in ids.items():
        rec = gt.get(str(m.get("gt_instance") or m.get("instance_id")))
        if rec and rec["motion_type"] == "static" and not rec.get("moves") \
                and rec.get("extent_m") and max(rec["extent_m"]) <= args.max_extent:
            keep.add(int(local))
    obs = load_obs(sd, args.seg, ids, keep, 1, W, H, 0.0)

    poses = {i: sfm[i].copy() for i in reg}
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
    print("등록 %d프레임 · 랜드마크 %d개" % (len(reg), len(land)))

    # 구간별 앵커: 구간 안에서 기준선이 가장 긴 두 카메라, d0 = 지도거리 × s_i
    anchors = []
    for seg, s_i in segment_scales(sd, sfm, reg, K, args.depth):
        C = np.array([sfm[i][:3, 3] for i in seg])
        d = np.linalg.norm(C[:, None] - C[None, :], axis=2)
        a, b = np.unravel_index(np.argmax(d), d.shape)
        if d[a, b] < 0.2:
            continue
        anchors.append((seg[a], seg[b], float(d[a, b] * s_i)))
    print("구간 앵커 %d개 · d0(m): %s"
          % (len(anchors), " ".join("%.2f" % a[2] for a in anchors)))

    m0, p0, s0 = ate(poses, gtp)
    # 워밍업: 균일 스케일은 재투영의 자유 방향인데 huber 에 뭉개진 앵커 기울기로는
    # 최적화가 그 방향을 못 걷는다(합성 실측: 요구 2× → 이동 0). 중앙 스케일을
    # 좌표에 직접 곱해 앵커 잔차를 이차영역으로 가져온 뒤 BA 를 돈다.
    seg_s = segment_scales(sd, sfm, reg, K, args.depth)
    s_med = float(np.median([x[1] for x in seg_s]))
    for f in poses:
        poses[f][:3, 3] *= s_med
    for l in land:
        land[l] = land[l] * s_med
    print("워밍업: 전 좌표 ×%.3f" % s_med)
    poses, land, rms = local_ba(poses, land, obs, K, anchor=anchors,
                                max_nfev=args.nfev)
    med, p9, sc = ate(poses, gtp)
    print("BA RMS %.2f px" % (rms or -1))
    print("펴기 전: ATE 중앙 %.3f · p90 %.3f · 정합스케일 %.3f" % (m0, p0, s0))
    print("**펴기 후: ATE 중앙 %.3f · p90 %.3f · 정합스케일 %.3f**" % (med, p9, sc))

    if args.out:
        out = np.tile(np.eye(4), (len(gtp), 1, 1))
        for i, T in poses.items():
            out[i] = T
        np.savetxt(args.out, out.reshape(len(gtp), 16))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
