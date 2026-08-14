"""뎁스 평가 — 정확도만 보지 않는다. 우리에게 중요한 건 **일관성**이다.

프레임 단위 AbsRel 이 좋아도 윈도우마다 스케일이 흔들리면 Khronos 는 정적 물체가
움직였다고 읽는다. 그래서 세 축을 함께 잰다:

    1. 정확도    AbsRel / RMSE / δ<1.25          (ADT GT 뎁스 대비)
    2. 시간 일관성 TAE — 인접 프레임을 GT 포즈로 워프한 잔차
    3. **3D 일관성** 정적 물체 중심의 시간축 산포 (미터) — 가짜 변화의 직접 선행지표
"""
import numpy as np


def depth_errors(pred, gt, min_d=0.2, max_d=10.0):
    """pred/gt: (H,W) 미터. 0 은 무효."""
    m = (gt > min_d) & (gt < max_d) & (pred > min_d) & np.isfinite(pred)
    if m.sum() < 100:
        return None
    p, g = pred[m], gt[m]
    r = np.maximum(p / g, g / p)
    return {
        "n": int(m.sum()),
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(r < 1.25)),
        "delta2": float(np.mean(r < 1.25 ** 2)),
        "bias": float(np.median(p / g)),        # 1 에서 벗어나면 계통 스케일 오차
    }


def unproject(depth, K, stride=1):
    """(H,W) 뎁스 → (M,3) 카메라 좌표 점. 0 은 버린다."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    m = d > 0
    u, v, d = u[m], v[m], d[m]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    return np.stack([x, y, d], axis=1)


def tae(d0, d1, T0, T1, K, min_d=0.2, max_d=10.0):
    """Temporal Alignment Error — d0 을 프레임1 로 워프해 상대 오차를 본다.

    포즈가 GT 라 순수하게 뎁스의 시간 불일치만 측정된다. 값이 작을수록 좋다.
    """
    H, W = d0.shape
    pts = unproject(d0, K)
    if len(pts) < 100:
        return None
    T10 = np.linalg.inv(T1) @ T0                       # camera0 → camera1
    p1 = pts @ T10[:3, :3].T + T10[:3, 3]
    z = p1[:, 2]
    ok = (z > min_d) & (z < max_d)
    p1, z = p1[ok], z[ok]
    uv = p1[:, :2] / z[:, None] @ K[:2, :2].T + K[:2, 2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    ref = d1[v, u]
    ok = (ref > min_d) & (ref < max_d)
    if ok.sum() < 100:
        return None
    z, ref = z[ok], ref[ok]
    return float(np.mean(np.abs(z - ref) / ref))


def static_centroids(depth, seg, K, T_wc, locals_wanted, min_area=300):
    """정적 인스턴스별 world 중심(뎁스 중앙값 기반). {local_id: (3,)}"""
    out = {}
    for lid in locals_wanted:
        m = (seg == lid) & (depth > 0)
        if m.sum() < min_area:
            continue
        v, u = np.nonzero(m)
        d = depth[v, u]
        keep = np.abs(d - np.median(d)) < 0.5          # 경계에서 튄 픽셀 제거
        if keep.sum() < min_area // 2:
            continue
        u, v, d = u[keep], v[keep], d[keep]
        x = (u - K[0, 2]) / K[0, 0] * d
        y = (v - K[1, 2]) / K[1, 1] * d
        c = np.median(np.stack([x, y, d], axis=1), axis=0)
        out[int(lid)] = T_wc[:3, :3] @ c + T_wc[:3, 3]
    return out


def dispersion(tracks, min_obs=8):
    """{id: [world 점들]} → 인스턴스별 산포(미터)의 통계.

    정적 물체인데도 프레임마다 다른 자리에 재구성된다면, 그 산포가 곧 Khronos 가
    보게 될 '가짜 이동'의 크기다.
    """
    vals = []
    for pts in tracks.values():
        if len(pts) < min_obs:
            continue
        P = np.asarray(pts)
        vals.append(float(np.mean(np.linalg.norm(P - P.mean(axis=0), axis=1))))
    if not vals:
        return None
    v = np.array(vals)
    return {"n_instances": len(v), "mean_m": float(v.mean()),
            "median_m": float(np.median(v)), "p90_m": float(np.percentile(v, 90))}
