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
# 채움비 = 마스크 면적 / bbox 면적. **가림의 대용 지표**이고 마스크만으로 구해진다.
# 실측(GT 포즈·랜드마크): 필터 없음 2.1px → ≥0.50 1.4px → ≥0.65 **1.1px(p90 7.9)**.
# ⚠️ 마스크 '면적'으로 거르면 반대로 나빠진다(8000px↑ 에서 12.0px) — 크게 보이는 것은
# 가까이 있어 부분만 보이고, 보이는 면의 중심이 3D 중심에서 멀기 때문이다.
MIN_FILL = 0.65


def load_obs(sd, seg_sub, ids, keep, every, W, H, min_fill=MIN_FILL):
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
            bw = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            if n / bw < min_fill:                          # 가려진 것은 중심이 밀린다
                continue                                   # (기본 0 = 필터 대신 가중치)
            obs[i][a] = (float(xs.mean()), float(ys.mean()), float(n / bw))
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


def bundle_adjust(poses, land, obs, K, f_scale=2.5, max_nfev=120, w_pow=1.0):
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

        # ⚠️ 채움비를 **하드 필터 대신 가중치**로 쓴다. 실측: 필터로 자르면 p90 은
    # 1.340 → 0.944 로 좋아지지만 관측이 1468 → 782 로 줄어 랜드마크가 반토막 나고
    # 중앙값이 0.357 → 0.389 로 나빠졌다. 가중치는 관측 수를 유지하면서 편향만 누른다.
    rows = [(fi[f], li[l], o[0], o[1], o[2] if len(o) > 2 else 1.0)
            for f in fids for l, o in obs[f].items() if l in li]
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
    fill = np.array([r[4] for r in rows])
    wt = np.clip(fill / max(fill.max(), 1e-6), 0.15, 1.0) ** w_pow    # 잔차 가중
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def resid(x):
        P = x[F * 6:].reshape(L, 3)[pi]
        out = np.empty((len(rows), 2))
        for c in np.unique(ci):
            m = ci == c
            R = cv2.Rodrigues(x[c * 6:c * 6 + 3])[0]
            X = P[m] @ R.T + x[c * 6 + 3:c * 6 + 6]
            z = np.maximum(X[:, 2], 1e-3)
            out[m, 0] = (fx * X[:, 0] / z + cx - uv[m, 0]) * wt[m]
            out[m, 1] = (fy * X[:, 1] / z + cy - uv[m, 1]) * wt[m]
        return out.ravel()

    S = lil_matrix((len(rows) * 2, len(x0)), dtype=int)
    for k, (c, p, *_rest) in enumerate(rows):
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


def pose_graph(init, odo, land, obs, K, w_rot=30.0, w_tra=3.0, f_scale=2.5,
               max_nfev=150, w_pow=0.0):
    """**포즈그래프 + 랜드마크** — 루프클로저처럼 전역 제약으로 쓴다.

    프레임별 PnP 는 시간적 연속성을 버린다. 실측에서 관측이 4개 미만인 **75/182 프레임이
    그냥 탈락**했고, 그게 남은 오차의 큰 몫이었다. 순차 추정의 **상대 운동은 국소적으로
    정확**하므로(윈도우 내부는 DA3 가 일관되게 푼다) 그것을 간선으로 남긴다:

        오도메트리 간선   이웃 프레임 상대 포즈  — 국소 정확, 전역 드리프트
        랜드마크 간선     재투영                — 전역 정확, 국소 잡음

    둘을 한 번에 최소화하면 관측이 없는 프레임도 포즈를 받고, 랜드마크가 드리프트를 잡는다.

    ⚠️ 초기 포즈의 **전역 스케일이 틀려 있다**(DA3 경로는 약 2배). 오도메트리 평행이동을
    그대로 제약으로 쓰면 틀린 스케일을 강제하게 되므로, 전역 스케일 s 를 **자유 변수**로
    두고 같이 푼다. 회전은 스케일과 무관하므로 강하게 묶는다.

    ⚠️ `init`(초기화)과 `odo`(오도메트리 측정)는 **반드시 분리**한다. 정제된 포즈와 원본
    포즈를 섞어 하나로 쓰면 게이지가 섞여 측정값이 오염된다 — 실제로 그렇게 했다가
    ATE 가 0.365 → 0.937 로 무너졌다. odo 는 항상 **원본 순차 추정**에서 온다.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    fids = sorted(init)
    lids = sorted(land)
    fi = {f: i for i, f in enumerate(fids)}
    li = {l: i for i, l in enumerate(lids)}
    F, L = len(fids), len(lids)

    rows = [(fi[f], li[l], o[0], o[1], o[2] if len(o) > 2 else 1.0)
            for f in fids if f in obs for l, o in obs[f].items() if l in li]
    pairs = [(fi[fids[k]], fi[fids[k + 1]]) for k in range(F - 1)]

    x0 = np.zeros(F * 6 + L * 3 + 1)
    Rm, Cm = {}, {}
    for f, i in fi.items():
        T = init[f]
        R = T[:3, :3].T
        Rm[i], Cm[i] = R, T[:3, 3]
        x0[i * 6:i * 6 + 3] = cv2.Rodrigues(R)[0].ravel()
        x0[i * 6 + 3:i * 6 + 6] = -R @ T[:3, 3]
    for l, i in li.items():
        x0[F * 6 + i * 3:F * 6 + i * 3 + 3] = land[l]
    x0[-1] = 1.0                                    # 오도메트리 전역 스케일

    # 측정된 상대 운동 — **원본 순차 추정**에서만 뽑는다 (게이지 오염 방지)
    Ro = {fi[f]: odo[f][:3, :3].T for f in fids if f in odo}
    Co = {fi[f]: odo[f][:3, 3] for f in fids if f in odo}
    pairs = [(a, b) for a, b in pairs if a in Ro and b in Ro]
    dmeas = np.array([Ro[a] @ (Co[b] - Co[a]) for a, b in pairs])
    Rrel = [Ro[b] @ Ro[a].T for a, b in pairs]

    ci = np.array([r[0] for r in rows])
    pi = np.array([r[1] for r in rows])
    uv = np.array([[r[2], r[3]] for r in rows])
    fill = np.array([r[4] for r in rows])
    wt = np.clip(fill / max(fill.max(), 1e-6), 0.15, 1.0) ** w_pow
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    NR = len(rows) * 2

    def resid(x):
        out = np.empty(NR + len(pairs) * 6)
        P = x[F * 6:F * 6 + L * 3].reshape(L, 3)[pi]
        for c in np.unique(ci):
            m = ci == c
            R = cv2.Rodrigues(x[c * 6:c * 6 + 3])[0]
            X = P[m] @ R.T + x[c * 6 + 3:c * 6 + 6]
            z = np.maximum(X[:, 2], 1e-3)
            idx = np.nonzero(m)[0]
            out[2 * idx] = (fx * X[:, 0] / z + cx - uv[m, 0]) * wt[m]
            out[2 * idx + 1] = (fy * X[:, 1] / z + cy - uv[m, 1]) * wt[m]
        s = x[-1]
        for k, (a, b) in enumerate(pairs):
            Ra = cv2.Rodrigues(x[a * 6:a * 6 + 3])[0]
            Rb = cv2.Rodrigues(x[b * 6:b * 6 + 3])[0]
            Ca = -Ra.T @ x[a * 6 + 3:a * 6 + 6]
            Cb = -Rb.T @ x[b * 6 + 3:b * 6 + 6]
            er = cv2.Rodrigues((Rb @ Ra.T) @ Rrel[k].T)[0].ravel()
            et = Ra @ (Cb - Ca) - s * dmeas[k]
            out[NR + k * 6:NR + k * 6 + 3] = er * w_rot
            out[NR + k * 6 + 3:NR + k * 6 + 6] = et * w_tra
        return out

    S = lil_matrix((NR + len(pairs) * 6, len(x0)), dtype=int)
    for k, (c, p, *_r) in enumerate(rows):
        S[2 * k:2 * k + 2, c * 6:c * 6 + 6] = 1
        S[2 * k:2 * k + 2, F * 6 + p * 3:F * 6 + p * 3 + 3] = 1
    for k, (a, b) in enumerate(pairs):
        S[NR + k * 6:NR + k * 6 + 6, a * 6:a * 6 + 6] = 1
        S[NR + k * 6:NR + k * 6 + 6, b * 6:b * 6 + 6] = 1
        S[NR + k * 6 + 3:NR + k * 6 + 6, -1] = 1
    S[:, :6] = 0

    r0 = resid(x0)
    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=f_scale,
                        method="trf", max_nfev=max_nfev, verbose=0)
    x = res.x
    x[:6] = x0[:6]
    newp = {}
    for f, i in fi.items():
        R = cv2.Rodrigues(x[i * 6:i * 6 + 3])[0]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ x[i * 6 + 3:i * 6 + 6]
        newp[f] = T
    newl = {l: x[F * 6 + i * 3:F * 6 + i * 3 + 3] for l, i in li.items()}
    info = dict(n_obs=len(rows), n_pose=F, n_land=L, n_edge=len(pairs),
                rms0=float(np.sqrt(np.mean(r0[:NR] ** 2))),
                rms1=float(np.sqrt(np.mean(res.fun[:NR] ** 2))), odo_scale=float(x[-1]))
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
    ap.add_argument("--min-fill", type=float, default=0.0, help="채움비 하한 (0=가중치만)")
    ap.add_argument("--w-pow", type=float, default=0.0, help="채움비 가중 지수 (0=가중 없음)")
    ap.add_argument("--pose-graph", action="store_true", help="오도메트리 간선 + 랜드마크 동시 최적화")
    ap.add_argument("--w-rot", type=float, default=30.0, help="오도메트리 회전 가중")
    ap.add_argument("--w-tra", type=float, default=3.0, help="오도메트리 평행이동 가중")
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

    obs = load_obs(sd, args.seg, ids, set(land), args.every, W, H, args.min_fill)
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
            views = [(poses[i], obs[i][lid][:2]) for i in sorted(obs)
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
            pts2 = [obs[i][l][:2] for l in obs[i] if l in land]
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

    if args.pose_graph:
        # 관측이 없는 프레임도 포함해 전체 궤적을 함께 푼다
        # 관측 없는 프레임은 원본 포즈를 **정제 게이지로 sim3 정렬**해 채운다
        raw = {i: init[i].copy() for i in range(len(init)) if i % args.every == 0}
        common = sorted(set(raw) & set(poses))
        A = np.array([raw[i][:3, 3] for i in common])
        B = np.array([poses[i][:3, 3] for i in common])
        sc_, R_, t_, _ = robust_umeyama(A, B)
        allp = {}
        for i, T in raw.items():
            if i in poses:
                allp[i] = poses[i]
                continue
            T2 = np.eye(4)
            T2[:3, :3] = R_ @ T[:3, :3]
            T2[:3, 3] = sc_ * (R_ @ T[:3, 3]) + t_
            allp[i] = T2
        poses, land, info = pose_graph(allp, raw, land, obs, K, w_rot=args.w_rot,
                                       w_tra=args.w_tra, f_scale=args.f_scale,
                                       max_nfev=args.ba_iters, w_pow=args.w_pow)
        v = sorted(poses)
        med, p9, sc = ate(poses, gtp, v)
        print("\n**포즈그래프+랜드마크** 관측 %d · 포즈 %d(전체) · 랜드마크 %d · 간선 %d"
              % (info["n_obs"], info["n_pose"], info["n_land"], info["n_edge"]))
        print("  재투영 RMS %.2f → **%.2f px** · 오도메트리 스케일 %.3f"
              % (info["rms0"], info["rms1"], info["odo_scale"]))
        print("  ATE 중앙 **%.3f m** · p90 %.3f · 정합스케일 %.3f" % (med, p9, sc))

    if args.ba:
        poses, land, info = bundle_adjust(poses, land, obs, K, f_scale=args.f_scale, max_nfev=args.ba_iters,
                                          w_pow=args.w_pow)
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
