#!/usr/bin/env python3
"""루프 클로저 실행 — DA3 윈도우 포즈를 물체 랜드마크로 전역 보정.

    $P scripts/close_loops.py --seq <name>

입력: `poses_raw_np/`(윈도우별 포즈), `depth_raw_np/`, `gt/seg/`
출력: `pose/poses_da3_lc.txt`, `depth_da3_lc/`, `da3_lc_meta.json`

각 윈도우가 **자기 소유 프레임**에서 본 물체 인스턴스의 3D 위치를 그 윈도우 좌표계로
모아 랜드마크 관측으로 쓴다. 같은 인스턴스를 여러 윈도우가 봤으면 그게 루프 제약이다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.global_sfm import run as global_run   # noqa: E402
from kx.depth.loop_close import MIN_OBS, LoopCloser   # noqa: E402
from kx.depth.pose_stitch import (_scene_points, load_windows, metric_scale,   # noqa: E402
                                  robust_umeyama, stitch, up_from_trajectory)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def window_observations(seq_dir, windows, raw_dir, min_area=400, stride=8, max_extent=3.0):
    """윈도우별 {pts: {instance: local xyz}, cam: {frame: c2w}}"""
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"])
    ids = json.load(open(os.path.join(seq_dir, "gt", "seg_ids.json")))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    keep = {}
    for local, m in ids.items():
        rec = gt.get(str(m["instance_id"]))
        ext = rec.get("extent_m") if rec else None
        if ext and max(ext) > max_extent:
            continue                       # 벽·바닥은 랜드마크로 쓰지 않는다
        keep[int(local)] = m["instance_id"]

    out = []
    for w in windows:
        acc, cams = {}, {}
        for k, f in enumerate(w["frames"]):
            cams[int(f)] = w["c2w"][k]
            if not w["owner"][k]:
                continue                   # 뎁스는 소유 윈도우에만 있다
            dp = os.path.join(seq_dir, raw_dir, "%06d.npy" % f)
            sp = os.path.join(seq_dir, "gt", "seg", "%06d.png" % f)
            if not (os.path.exists(dp) and os.path.exists(sp)):
                continue
            d = np.load(dp).astype(np.float32)
            seg = np.array(Image.open(sp))
            Tw = w["c2w"][k]
            for lid in np.unique(seg[::stride, ::stride]):
                if lid == 0 or int(lid) not in keep:
                    continue
                m = (seg == lid) & (d > 0)
                if m.sum() < min_area:
                    continue
                v, u = np.nonzero(m)
                sel = slice(None, None, max(len(v) // 300, 1))
                v, u = v[sel], u[sel]
                z = d[v, u]
                med = np.median(z)
                ok = np.abs(z - med) < 0.6
                if ok.sum() < 30:
                    continue
                u, v, z = u[ok], v[ok], z[ok]
                pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
                p = np.median(pc @ Tw[:3, :3].T + Tw[:3, 3], axis=0)
                acc.setdefault(keep[int(lid)], []).append(p)
        pts = {k: np.median(np.array(v), axis=0) for k, v in acc.items() if len(v) >= MIN_OBS}
        out.append({"pts": pts, "cam": cams})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--pose-dir", default="poses_raw_np")
    ap.add_argument("--raw", default="depth_raw_np")
    ap.add_argument("--out-pose", default="pose/poses_da3_lc.txt")
    ap.add_argument("--out-depth", default="depth_da3_lc")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--method", default="sfm", choices=["sfm", "bcd"],
                    help="sfm=전역 회전평균+선형해(정석) / bcd=블록좌표하강(구판)")
    ap.add_argument("--wearer-height", type=float, default=1.55)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    n = len(open(os.path.join(seq_dir, "pose", "poses.txt")).read().strip().split("\n"))
    W = load_windows(os.path.join(seq_dir, args.pose_dir))
    init = stitch(W)                       # 순차 체이닝을 초기값으로
    print("윈도우 %d개, 초기 체이닝 완료" % len(W), flush=True)

    obs = window_observations(seq_dir, W, args.raw)
    npts = [len(o["pts"]) for o in obs]
    print("윈도우당 랜드마크 관측: 중앙 %d  (최소 %d, 최대 %d)"
          % (np.median(npts), min(npts), max(npts)), flush=True)

    if args.method == "sfm":
        T, info = global_run(obs, init_R=[a["R"] for a in init])
        print("윈도우 쌍 제약 %d개 (인접이 아니라 공통 랜드마크 기준)" % info["edges"])
        print("회전 평균 잔차 %.2f°   선형해 잔차 %.4f m"
              % (info["rot_residual_deg"], info["lsq_residual_m"]))
    else:
        lc = LoopCloser(obs, init)
        T, info = lc.run(iters=args.iters, verbose=True)
        print("랜드마크 %d개, 공유 프레임 %d개" % (info["landmarks"], info["shared_frames"]))
        print("랜드마크 산포: %.4f m → %.4f m"
              % (info["spread_history"][0], info["spread_history"][-1]))

    # --- 전역 포즈/스케일 재구성 ---
    poses = np.tile(np.eye(4), (n, 1, 1))
    scale = np.zeros(n)
    for w, a in zip(W, T):
        for k, fi in enumerate(w["frames"]):
            if not w["owner"][k] or fi >= n:
                continue
            poses[fi, :3, :3] = a["R"] @ w["c2w"][k, :3, :3]
            poses[fi, :3, 3] = a["s"] * (a["R"] @ w["c2w"][k, :3, 3]) + a["t"]
            scale[fi] = a["s"]
    seen = scale > 0
    up = up_from_trajectory(poses[seen][:, :3, 3])
    pts = _scene_points(seq_dir, poses, scale, raw_dir=args.raw)
    if (pts @ up).mean() > (poses[seen][:, :3, 3] @ up).mean():
        up = -up
    gs, span, cam_h = metric_scale(poses[seen][:, :3, 3], pts, up, args.wearer_height)
    poses[:, :3, 3] *= gs
    scale *= gs

    np.savetxt(os.path.join(seq_dir, args.out_pose), poses.reshape(n, 16), fmt="%.9g")
    out_dir = os.path.join(seq_dir, args.out_depth)
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(seq_dir, args.raw))):
        if not f.endswith(".npy"):
            continue
        i = int(os.path.splitext(f)[0])
        if i >= n or scale[i] == 0:
            continue
        d = np.load(os.path.join(seq_dir, args.raw, f)).astype(np.float32) * scale[i]
        Image.fromarray(np.clip(d * 1000.0, 0, 65535).astype(np.uint16)).save(
            os.path.join(out_dir, "%06d.png" % i))
    json.dump({"method": args.method, "info": {k: v for k, v in info.items()},
               "global_scale": gs,
               "up": up.tolist(), "depth_scale": scale.tolist()},
              open(os.path.join(seq_dir, "da3_lc_meta.json"), "w"))

    # --- 채점 ---
    gt = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    from scipy.spatial.transform import Rotation
    s, R, t, _ = robust_umeyama(poses[seen][:, :3, 3], gt[seen][:, :3, 3])
    ate = np.linalg.norm((s * (R @ poses[seen][:, :3, 3].T)).T + t - gt[seen][:, :3, 3], axis=1)
    Rs = np.array([gt[i, :3, :3] @ poses[i, :3, :3].T for i in np.flatnonzero(seen)])
    Rm = Rotation.from_matrix(Rs).mean().as_matrix()
    sp = np.array([np.degrees(np.arccos(np.clip((np.trace(Rm.T @ M) - 1) / 2, -1, 1))) for M in Rs])
    L = np.linalg.norm(np.diff(gt[seen][:, :3, 3], axis=0), axis=1).sum()
    print("\n[채점] ATE 중앙 %.3f m (%.1f%%)  자세산포 중앙 %.2f° p90 %.2f°  스케일 %.4f"
          % (np.median(ate), 100 * np.median(ate) / L, np.median(sp), np.percentile(sp, 90), s))


if __name__ == "__main__":
    main()
