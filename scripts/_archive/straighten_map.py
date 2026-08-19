#!/usr/bin/env python3
"""SfM 궤적의 **불균일 스케일을 편다** — 구간별 바닥 사전 → 스케일 함수 s(t) → 증분 적분.

    $P scripts/straighten_map.py --seq <name> --sfm /tmp/poses_inc_e1.txt \
        --depth depth_sfmlm2 --out /tmp/poses_sfm_straight.txt

문제: SfM 지도는 v1 씨앗에서 스케일을 물려받는데 v1 스케일은 구간별로 1.3~1.9×
불균일하다. 단일 sim3 로는 못 편다 — 전역 ATE 가 ~0.76 m 에서 정체(실측: 직접
앵커/사슬/SfM 등록 프레임의 오차가 모두 같았다 = 스티칭이 아니라 지도가 휜 것).

해법: 시퀀스를 구간으로 나눠 각 구간의 점구름(지도 스케일 뎁스)에서 바닥을 찾고,
카메라 높이 = 1.55 m 사전으로 구간별 스케일 s_i 를 얻는다. 위치 증분에 보간된
s(t) 를 곱해 적분하면 궤적이 균일 미터가 된다. 회전은 그대로 둔다.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.depth.pose_stitch import up_from_trajectory     # noqa: E402
from scripts.incremental_sfm import ate                 # noqa: E402

WEARER_HEIGHT = 1.55


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--sfm", default="/tmp/poses_inc_e1.txt")
    ap.add_argument("--depth", default="depth_sfmlm2", help="지도 스케일로 정합된 뎁스")
    ap.add_argument("--seg-frames", type=int, default=50, help="구간당 등록 프레임 수")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = __import__("json").load(open(os.path.join(sd, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    gtp = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    sfm = np.loadtxt(args.sfm).reshape(-1, 4, 4)
    reg = sorted(i for i in range(len(sfm)) if not np.allclose(sfm[i], np.eye(4)))
    centers = np.array([sfm[i][:3, 3] for i in reg])
    up = up_from_trajectory(centers)

    # ① 구간별 바닥 → 지도 단위 카메라 높이 → 스케일 s_i
    segs = [reg[k:k + args.seg_frames] for k in range(0, len(reg), args.seg_frames)]
    if len(segs) > 1 and len(segs[-1]) < args.seg_frames // 2:
        segs[-2] += segs.pop()
    mids, scales = [], []
    dep_dir = os.path.join(sd, args.depth)
    for seg in segs:
        pts = []
        for i in seg[::5]:
            p = os.path.join(dep_dir, "%06d.png" % i)
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
        mids.append(0.5 * (seg[0] + seg[-1]))
        scales.append(WEARER_HEIGHT / cam_h)
    # 바닥 추정은 구간에 따라 파탄난다(실측: 마지막 구간 0.60 vs 나머지 1.5~1.7 —
    # 바닥이 화면에 안 잡히는 구도). 중앙값에서 30% 넘게 벗어난 구간은 이웃 보간.
    scales = np.array(scales, float)
    med_s = float(np.median(scales))
    bad = np.abs(scales / med_s - 1) > 0.30
    if bad.any():
        good = ~bad
        scales[bad] = np.interp(np.array(mids)[bad], np.array(mids)[good], scales[good])
        print("이상치 구간 %d개를 이웃 보간으로 대체" % int(bad.sum()))
    print("구간 %d개 · 스케일 %s" % (len(mids), " ".join("%.2f" % x for x in scales)))
    if len(mids) < 2:
        sys.exit("구간이 부족하다")

    # ② s(t) 보간 → 증분 적분으로 궤적 재구성 (회전은 유지)
    s_of = lambda f: float(np.interp(f, mids, scales))
    out = np.tile(np.eye(4), (len(gtp), 1, 1))
    prev = reg[0]
    out[prev] = sfm[prev].copy()
    out[prev][:3, 3] = sfm[prev][:3, 3] * s_of(prev)     # 시작점도 미터로
    for f in reg[1:]:
        s = s_of(0.5 * (prev + f))
        out[f] = sfm[f].copy()
        out[f][:3, 3] = out[prev][:3, 3] + s * (sfm[f][:3, 3] - sfm[prev][:3, 3])
        prev = f

    est = {i: out[i] for i in reg}
    med, p9, sc = ate(est, gtp)
    m0, p0, s0 = ate({i: sfm[i] for i in reg}, gtp)
    print("펴기 전: ATE 중앙 %.3f · p90 %.3f · 정합스케일 %.3f" % (m0, p0, s0))
    print("**펴기 후: ATE 중앙 %.3f · p90 %.3f · 정합스케일 %.3f**" % (med, p9, sc))

    if args.out:
        np.savetxt(args.out, out.reshape(len(gtp), 16))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
