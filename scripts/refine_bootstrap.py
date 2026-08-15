#!/usr/bin/env python3
"""부트스트랩 정제 — 물체 검출로 포즈와 랜드맵을 **번갈아** 다시 푼다.

    $P scripts/refine_bootstrap.py --seq <name> --init pose/poses_da3_lc.txt

v1 의 포즈는 전부 **윈도우를 이어붙이는** 방식이라 오차가 누적됐다(최선 ATE 0.97 m).
랜드마크 루프클로저를 이미 썼는데도 그랬다 — 랜드마크가 윈도우 사이 sim3 를 묶는
제약으로만 쓰였기 때문이다.

여기서는 구조를 바꾼다. 물체를 **지도의 앵커**로 놓고 둘을 번갈아 푼다:

    ① PnP        각 프레임을 현재 랜드맵에 **독립적으로** 맞춘다 (누적 없음)
    ② 삼각측량    각 랜드마크를 현재 포즈들로 다시 삼각측량한다 (**뎁스 모델 불필요**)

②가 핵심이다. 한 번 포즈가 서면 랜드마크 3D 는 2D 관측만으로 다시 구할 수 있다.
모노 뎁스의 스케일 편의가 여기서 빠진다.

⚠️ 전역 스케일은 이 루프로 안 잡힌다 — 게이지 자유도라 외부 앵커(바닥 높이 등)가
필요하다. 여기서 좋아지는 것은 **내부 일관성**이고, 씬그래프가 필요로 하는 것도 그것이다.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.depth.pose_stitch import robust_umeyama       # noqa: E402

MIN_PX = 300
BORDER = 4
# ⚠️ 실측: 마스크 중심 vs 3D 중심 투영 오차가 전체 7.2px(p90 77) → 작고(<1m) 안 잘린
# 것만 2.1px(p90 20) 로 3.4배 준다. 관측 모델의 하한이 여기서 정해지므로 세게 조인다.
MAX_EXTENT = 1.0
MIN_VIEWS = 4              # 삼각측량에 필요한 최소 관측 수
MIN_BASELINE = 0.25        # m. 시차가 없으면 삼각측량이 불안정하다


def load_obs(sd, seg_sub, ids, keep, every, W, H):
    """[frame][local_id] = (u, v) — 마스크 중심. 잘린 것은 뺀다."""
    obs = defaultdict(dict)
    seg_dir = os.path.join(sd, seg_sub)
    for f in sorted(os.listdir(seg_dir)):
        i = int(f.split(".")[0])
        if i % every:
            continue
        s = np.array(Image.open(os.path.join(seg_dir, f)))
        u, c = np.unique(s, return_counts=True)
        for a, n in zip(u, c):
            a = int(a)
            if a == 0 or n < MIN_PX or a not in keep:
                continue
            ys, xs = np.nonzero(s == a)
            if xs.min() < BORDER or ys.min() < BORDER \
                    or xs.max() > W - 1 - BORDER or ys.max() > H - 1 - BORDER:
                continue
            obs[i][a] = (float(xs.mean()), float(ys.mean()))
    return obs


def triangulate(views, K):
    """[(T_wc 4x4, (u,v))] → 3D 점. 다시점 DLT. 시차가 모자라면 None."""
    if len(views) < MIN_VIEWS:
        return None
    C = np.array([T[:3, 3] for T, _ in views])
    if np.linalg.norm(C.max(0) - C.min(0)) < MIN_BASELINE:
        return None
    A = []
    Ki = np.linalg.inv(K)
    for T, (u, v) in views:
        R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]      # world→camera
        P = K @ np.hstack([R, t.reshape(3, 1)])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return None
    return X[:3] / X[3]


def pnp(pts3, pts2, K, reproj=25.0):
    if len(pts3) < 4:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.asarray(pts3, np.float64), np.asarray(pts2, np.float64), K, None,
        flags=cv2.SOLVEPNP_SQPNP, reprojectionError=reproj,
        iterationsCount=500, confidence=0.999)
    if not ok or inl is None or len(inl) < 4:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ tvec).ravel()
    return T, len(inl)


def bundle_adjust(poses, land, obs, K, f_scale=2.5, max_nfev=120):
    """**번들조정** — 포즈와 랜드마크를 재투영 오차로 동시에 줄인다.

    번갈아 푸는 방식(PnP ↔ 삼각측량)은 각 단계가 상대를 고정하므로 수렴이 오르내린다
    (실측: 0.743 → 0.770 → 0.681 → 0.560). BA 는 둘을 한 번에 움직인다.

    ⚠️ 게이지 자유도(sim3 7-DoF)를 남겨두면 최적화가 그 궤도를 떠돈다. 첫 카메라를
    고정해 회전·평행이동 6-DoF 를 없앤다. 스케일은 남지만 채점이 sim3 정합이라 무해하다.
    ⚠️ 마스크 중심은 3D 중심의 투영이 아니다 — 잔차가 편측으로 튀므로 Huber 를 쓴다.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    fids = sorted(poses)
    lids = sorted(land)
    fi = {f: i for i, f in enumerate(fids)}
    li = {l: i for i, l in enumerate(lids)}

    rows = [(fi[f], li[l], u, v) for f in fids for l, (u, v) in obs[f].items() if l in li]
    if len(rows) < 50:
        return poses, land, None
    F, L = len(fids), len(lids)

    x0 = np.zeros(F * 6 + L * 3)
    for f, i in fi.items():
        T = poses[f]
        R = T[:3, :3].T                                  # world→camera
        x0[i * 6:i * 6 + 3] = cv2.Rodrigues(R)[0].ravel()
        x0[i * 6 + 3:i * 6 + 6] = -R @ T[:3, 3]
    for l, i in li.items():
        x0[F * 6 + i * 3:F * 6 + i * 3 + 3] = land[l]

    ci = np.array([r[0] for r in rows])
    pi = np.array([r[1] for r in rows])
    uv = np.array([[r[2], r[3]] for r in rows])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def resid(x):
        P = x[F * 6:].reshape(L, 3)[pi]
        out = np.empty((len(rows), 2))
        for c in np.unique(ci):
            m = ci == c
            R = cv2.Rodrigues(x[c * 6:c * 6 + 3])[0]
            X = P[m] @ R.T + x[c * 6 + 3:c * 6 + 6]
            z = np.maximum(X[:, 2], 1e-3)
            out[m, 0] = fx * X[:, 0] / z + cx - uv[m, 0]
            out[m, 1] = fy * X[:, 1] / z + cy - uv[m, 1]
        return out.ravel()

    S = lil_matrix((len(rows) * 2, len(x0)), dtype=int)
    for k, (c, p, _, _) in enumerate(rows):
        S[2 * k:2 * k + 2, c * 6:c * 6 + 6] = 1
        S[2 * k:2 * k + 2, F * 6 + p * 3:F * 6 + p * 3 + 3] = 1
    S[:, :6] = 0                                          # 첫 카메라 고정

    r0 = resid(x0)
    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=f_scale,
                        method="trf", max_nfev=max_nfev, verbose=0)
    x = res.x
    x[:6] = x0[:6]
    newp, newl = {}, {}
    for f, i in fi.items():
        R = cv2.Rodrigues(x[i * 6:i * 6 + 3])[0]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ x[i * 6 + 3:i * 6 + 6]
        newp[f] = T
    for l, i in li.items():
        newl[l] = x[F * 6 + i * 3:F * 6 + i * 3 + 3]
    info = dict(n_obs=len(rows), n_pose=F, n_land=L,
                rms0=float(np.sqrt(np.mean(r0 ** 2))),
                rms1=float(np.sqrt(np.mean(res.fun ** 2))))
    return newp, newl, info


def ate(est, gt, mask):
    """sim3 정합 후 ATE — 게이지 자유도를 빼고 내부 일관성만 본다."""
    A = np.array([est[i][:3, 3] for i in mask])
    B = np.array([gt[i][:3, 3] for i in mask])
    s, R, t, _ = robust_umeyama(A, B)      # 4-튜플 (마지막은 잔차 중앙값)
    d = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    return float(np.median(d)), float(np.percentile(d, 90)), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--init", default="pose/poses_da3_lc.txt")
    ap.add_argument("--graph", default="graph_da3lc.json")
    ap.add_argument("--seg", default="gt/seg")
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--ba", action="store_true", help="번갈아 풀기 뒤 번들조정")
    ap.add_argument("--ba-iters", type=int, default=120)
    ap.add_argument("--max-extent", type=float, default=None, help="랜드마크 최대 크기(m)")
    ap.add_argument("--f-scale", type=float, default=2.5, help="Huber 문턱(px). 관측 하한에 맞춘다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(sd, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    W, H = cam["width"], cam["height"]
    gtp = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    init = np.loadtxt(os.path.join(sd, args.init)).reshape(-1, 4, 4)
    ids = json.load(open(os.path.join(sd, args.seg_ids)))
    g = json.load(open(os.path.join(sd, args.graph)))
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]

    land = {}
    for local, m in ids.items():
        gi = str(m.get("gt_instance") or m.get("instance_id"))
        rec = gt.get(gi)
        if rec is None or rec["motion_type"] != "static" or rec.get("moves"):
            continue
        o = g["objects"].get(str(m.get("instance_id"))) or g["objects"].get(gi)
        if not o or not o.get("placements"):
            continue
        if o.get("extent_m") and max(o["extent_m"]) > (args.max_extent or MAX_EXTENT):
            continue
        land[int(local)] = np.array(o["placements"][0]["position"], float)
    print("초기 랜드마크 %d개 (출처 %s)" % (len(land), args.graph))

    obs = load_obs(sd, args.seg, ids, set(land), args.every, W, H)
    print("관측 프레임 %d · 관측 중앙 %.0f개/프레임"
          % (len(obs), np.median([len(v) for v in obs.values()])))

    poses = {i: init[i].copy() for i in obs if i < len(init)}
    valid = sorted(poses)
    m0, p90, s0 = ate(poses, gtp, valid)
    print("\n반복 0 (초기 %s): 포즈 %d · ATE 중앙 **%.3f m** · p90 %.3f"
          % (args.init, len(valid), m0, p90))

    for it in range(1, args.iters + 1):
        # ① 삼각측량 — 뎁스 모델을 쓰지 않는다
        new_land, n_tri = {}, 0
        for lid in land:
            views = [(poses[i], obs[i][lid]) for i in sorted(obs)
                     if lid in obs[i] and i in poses]
            X = triangulate(views, K)
            if X is not None and np.all(np.isfinite(X)):
                new_land[lid] = X
                n_tri += 1
            else:
                new_land[lid] = land[lid]
        land = new_land

        # ② PnP — 각 프레임을 랜드맵에 독립적으로
        newp, n_ok, inls = {}, 0, []
        for i in sorted(obs):
            pts3 = [land[l] for l in obs[i] if l in land]
            pts2 = [obs[i][l] for l in obs[i] if l in land]
            r = pnp(pts3, pts2, K)
            if r is None:
                continue
            newp[i], nin = r
            n_ok += 1
            inls.append(nin)
        if len(newp) < 10:
            print("  반복 %d: PnP 성공 %d — 중단" % (it, len(newp)))
            break
        poses = newp
        v = sorted(poses)
        med, p9, s = ate(poses, gtp, v)
        print("반복 %d: 삼각측량 %d · PnP 성공 %d/%d (인라이어 중앙 %.0f) · "
              "ATE 중앙 **%.3f m** · p90 %.3f · 정합스케일 %.3f"
              % (it, n_tri, n_ok, len(obs), np.median(inls), med, p9, s))

    if args.ba:
        poses, land, info = bundle_adjust(poses, land, obs, K, f_scale=args.f_scale, max_nfev=args.ba_iters)
        if info:
            v = sorted(poses)
            med, p9, s = ate(poses, gtp, v)
            print("\n**번들조정** 관측 %d · 포즈 %d · 랜드마크 %d · "
                  "재투영 RMS %.2f → **%.2f px**"
                  % (info["n_obs"], info["n_pose"], info["n_land"],
                     info["rms0"], info["rms1"]))
            print("  ATE 중앙 **%.3f m** · p90 %.3f · 정합스케일 %.3f" % (med, p9, s))

    if args.out:
        arr = np.zeros((len(gtp), 4, 4))
        for i in range(len(gtp)):
            arr[i] = poses.get(i, np.eye(4))
        np.savetxt(args.out, arr.reshape(len(gtp), 16))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
