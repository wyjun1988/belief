"""DA3 윈도우별 포즈를 하나의 전역 궤적으로 잇는다 — **기기 없는 조건**의 핵심 조각.

DA3 를 포즈 없이 돌리면 윈도우마다 좌표계와 스케일이 제각각이다. 실측(decoration 9윈도우):

    윈도우 내부 위치     ATE 2.9cm (궤적길이의 2.0%)      ← 좋다
    윈도우 내부 회전     원시 9.78° → 고정오프셋 제거 후 **2.90°**  ← 상대자세는 정확
    윈도우 스케일       1.14 ~ 3.05 (평균 2.22)          ← 제각각, 반드시 이어야 함

즉 틀어진 것은 **윈도우의 전역 자세·스케일**이지 내부 기하가 아니다. 겹치는 프레임의
카메라 중심으로 sim3 를 풀어 차례로 이어붙이면 하나의 궤적이 된다.

남는 자유도는 **전역 sim3 하나**(전체 회전·평행이동·스케일). 씬그래프 자체는 스케일이
일정하기만 하면 되지만(임계값을 같이 비례시키면 동일한 그래프가 나온다), 사람이 읽는
숫자와 임계 기본값을 그대로 쓰려면 스케일 하나는 정해야 한다. 기기가 없으므로 **물리
사전지식**으로 고정한다: 머리에 쓴 카메라는 바닥에서 대략 1.55m 위에 있다.

'위' 방향도 GT 없이 정한다 — 걸어다니는 사람의 머리 높이는 거의 일정하므로 **카메라
중심들의 분산이 가장 작은 방향**이 연직이다. (IMU 가 있으면 중력으로 바로 대체할 것.)
"""
import glob
import json
import os

import numpy as np

WEARER_HEIGHT = 1.55      # m. 머리 착용 카메라의 바닥 대비 높이 (전역 스케일을 정하는 유일한 사전지식)
MIN_OVERLAP = 4           # 이어붙이기에 필요한 최소 공통 프레임
HUBER_ITERS = 3


def _to44(E):
    if E.shape[-2:] == (3, 4):
        B = np.tile(np.eye(4), (len(E), 1, 1))
        B[:, :3, :4] = E
        return B
    return E


def umeyama(X, Y, w=None):
    """X → Y 의 sim3 (s, R, t).  Y ≈ s·R·X + t"""
    w = np.ones(len(X)) if w is None else w
    w = w / w.sum()
    mx, my = (w[:, None] * X).sum(0), (w[:, None] * Y).sum(0)
    Xc, Yc = X - mx, Y - my
    S = (w[:, None] * Xc).T @ Yc
    U, D, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    E = np.diag([1.0, 1.0, d])
    R = (U @ E @ Vt).T
    var = (w[:, None] * Xc ** 2).sum()
    s = float((D * np.array([1.0, 1.0, d])).sum() / max(var, 1e-12))
    return s, R, my - s * R @ mx


def robust_umeyama(X, Y, iters=HUBER_ITERS):
    """겹침 구간에 튀는 프레임이 있으므로 잔차 기반 재가중을 몇 번 돈다."""
    s, R, t = umeyama(X, Y)
    for _ in range(iters):
        r = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
        sc = 1.4826 * np.median(r) + 1e-6
        w = 1.0 / (1.0 + (r / (2.0 * sc)) ** 2)
        s, R, t = umeyama(X, Y, w)
    r = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    return s, R, t, float(np.median(r))


def load_windows(pose_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(pose_dir, "*.npz"))):
        z = np.load(f)
        out.append({"file": os.path.basename(f), "frames": z["frames"].astype(int),
                    "c2w": np.linalg.inv(_to44(z["extrinsics"])),
                    "owner": z["owner"].astype(bool)})
    return out


def _rotation_average(Rs):
    """회전 평균 (chordal L2) — SVD 로 가장 가까운 정규직교 행렬."""
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def stitch(windows, use_orientation=True):
    """윈도우들을 첫 윈도우 좌표계로 차례로 이어붙인다.

    ⚠️ **카메라 중심만으로 sim3 를 풀면 안 된다.** 처음에 그렇게 했다가 위치는 맞는데
    (이어붙이기 잔차 0.9mm) **회전 오차가 중앙 70°** 로 터졌다. 이유: 윈도우 하나는
    1.4m 걸어간 구간이라 카메라 중심들이 거의 **1차원 곡선**이고, 그 축 둘레의 회전이
    Umeyama 에서 구속되지 않는다 — 위치만 맞추고 자세는 아무 데나 두는 해가 최적이 된다.

    그래서 회전은 **겹치는 프레임들의 카메라 자세로 따로 평균**해서 구하고(구속이 확실하다),
    스케일·평행이동만 중심점으로 푼다.
    """
    acc = [{"s": 1.0, "R": np.eye(3), "t": np.zeros(3), "resid": 0.0,
            "n_overlap": 0, "rot_spread_deg": 0.0}]
    for i in range(1, len(windows)):
        prev, cur = windows[i - 1], windows[i]
        common = np.intersect1d(prev["frames"], cur["frames"])
        if len(common) < MIN_OVERLAP:
            acc.append({**acc[-1], "resid": np.nan, "n_overlap": len(common)})
            continue
        pi = np.searchsorted(prev["frames"], common)
        ci = np.searchsorted(cur["frames"], common)
        a = acc[i - 1]
        Ytarget = (a["s"] * (a["R"] @ prev["c2w"][pi, :3, 3].T)).T + a["t"]

        if use_orientation:
            # 겹치는 프레임마다 "이 윈도우의 자세 → 전역 자세"에 필요한 회전
            Rs = [(a["R"] @ prev["c2w"][p, :3, :3]) @ cur["c2w"][c, :3, :3].T
                  for p, c in zip(pi, ci)]
            R = _rotation_average(np.array(Rs))
            spread = float(np.mean([np.degrees(np.arccos(np.clip(
                (np.trace(R.T @ M) - 1) / 2, -1, 1))) for M in Rs]))
            X = (R @ cur["c2w"][ci, :3, 3].T).T          # 회전 고정 후 스케일·평행이동만
            s = float(np.linalg.norm(Ytarget - Ytarget.mean(0)) /
                      max(np.linalg.norm(X - X.mean(0)), 1e-9))
            t = Ytarget.mean(0) - s * X.mean(0)
            resid = float(np.median(np.linalg.norm(s * X + t - Ytarget, axis=1)))
        else:
            s, R, t, resid = robust_umeyama(cur["c2w"][ci, :3, 3], Ytarget)
            spread = np.nan
        acc.append({"s": s, "R": R, "t": t, "resid": resid,
                    "n_overlap": int(len(common)), "rot_spread_deg": spread})
    return acc


def global_poses(windows, acc, n_frames):
    """프레임별 전역 c2w + 그 프레임 뎁스에 곱할 스케일."""
    poses = np.tile(np.eye(4), (n_frames, 1, 1))
    scale = np.zeros(n_frames)
    seen = np.zeros(n_frames, bool)
    for w, a in zip(windows, acc):
        for k, fi in enumerate(w["frames"]):
            if not w["owner"][k] or fi >= n_frames:
                continue
            T = np.eye(4)
            T[:3, :3] = a["R"] @ w["c2w"][k, :3, :3]
            T[:3, 3] = a["s"] * (a["R"] @ w["c2w"][k, :3, 3]) + a["t"]
            poses[fi] = T
            scale[fi] = a["s"]
            seen[fi] = True
    return poses, scale, seen


def up_from_trajectory(centers):
    """카메라 중심들의 **분산이 가장 작은 방향** = 연직 (머리 높이는 거의 일정하다)."""
    C = centers - centers.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=True)
    up = Vt[-1]
    return up / np.linalg.norm(up)


def metric_scale(centers, scene_points, up, wearer_height=WEARER_HEIGHT):
    """전역 스케일 하나를 물리 사전지식으로 정한다: **카메라 높이 = 바닥에서 1.55m**.

    ⚠️ 처음엔 궤적만 보고 `median(h - h.min())` 을 카메라 높이라고 썼다가 스케일이
    5배 틀렸다. 당연하다 — 궤적만으로는 **바닥이 어디인지 알 수 없다**. 머리 높이는
    거의 일정해서(변동 0.18m) 그 차이는 높이가 아니라 걸음의 출렁임이었다.
    바닥은 재구성된 점구름의 아래쪽 꼬리에서 찾아야 한다.
    """
    hc = centers @ up
    hp = scene_points @ up
    floor = float(np.percentile(hp, 2.0))          # 아래쪽 꼬리 = 바닥
    cam_h = float(np.median(hc) - floor)
    span = float(np.percentile(hc, 95) - np.percentile(hc, 5))
    return wearer_height / max(cam_h, 1e-3), span, cam_h


def _scene_points(seq_dir, poses, scale, raw_dir="depth_raw_np", every=40, stride=16):
    """전역 스케일 추정을 위한 성긴 점구름 — 바닥을 찾는 데만 쓴다."""
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"])
    P = []
    d_dir = os.path.join(seq_dir, raw_dir)
    for i in range(0, len(poses), every):
        f = os.path.join(d_dir, "%06d.npy" % i)
        if not os.path.exists(f) or scale[i] == 0:
            continue
        d = np.load(f).astype(np.float32)[::stride, ::stride] * scale[i]
        v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        u, v, d = u.ravel() * stride, v.ravel() * stride, d.ravel()
        m = (d > 0.1) & (d < 12)
        if m.sum() < 10:
            continue
        u, v, d = u[m], v[m], d[m]
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)
        P.append(pc @ poses[i][:3, :3].T + poses[i][:3, 3])
    return np.concatenate(P) if P else np.zeros((0, 3))


def run(seq_dir, pose_dir="poses_raw_np", out_pose="pose/poses_da3.txt",
        out_scale="da3_pose_meta.json", wearer_height=WEARER_HEIGHT,
        raw_dir="depth_raw_np"):
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    n = len(open(os.path.join(seq_dir, "pose", "poses.txt")).read().strip().split("\n"))
    W = load_windows(os.path.join(seq_dir, pose_dir))
    if not W:
        raise RuntimeError("윈도우 포즈가 없다: %s" % pose_dir)
    acc = stitch(W)
    poses, scale, seen = global_poses(W, acc, n)

    C = poses[seen][:, :3, 3]
    up = up_from_trajectory(C)
    pts = _scene_points(seq_dir, poses, scale, raw_dir=raw_dir)
    if len(pts) < 100:
        raise RuntimeError("바닥을 찾을 점구름이 없다")
    if (pts @ up).mean() > (C @ up).mean():      # up 부호는 분산만으로 안 정해진다
        up = -up
    gs, span, cam_h = metric_scale(C, pts, up, wearer_height)
    poses[:, :3, 3] *= gs
    scale *= gs

    os.makedirs(os.path.dirname(os.path.join(seq_dir, out_pose)), exist_ok=True)
    np.savetxt(os.path.join(seq_dir, out_pose), poses.reshape(len(poses), 16), fmt="%.9g")
    meta = {"n_windows": len(W), "frames_covered": int(seen.sum()), "n_frames": n,
            "global_scale": gs, "wearer_height_prior_m": wearer_height,
            "up": up.tolist(), "head_height_span_m": span,
            "camera_height_before_scale_m": cam_h,
            "rot_spread_deg": [a.get("rot_spread_deg") for a in acc],
            "window_scales": [a["s"] for a in acc],
            "stitch_residual_m": [a["resid"] for a in acc],
            "overlaps": [a["n_overlap"] for a in acc],
            "depth_scale": scale.tolist()}
    json.dump(meta, open(os.path.join(seq_dir, out_scale), "w"))
    return meta
