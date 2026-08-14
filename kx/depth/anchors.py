"""전역 희소 앵커 — Aria MPS 반-조밀 포인트를 프레임별 깊이 관측으로 바꾼다.

이게 P1 정합의 심장이다. 슬라이딩 윈도우끼리 스케일을 이어붙이면(체이닝) 오차가
누적되지만, **모든 프레임을 같은 전역 포인트 클라우드에 절대 정렬**하면 드리프트가
원천적으로 생기지 않는다. 우리는 정확한 GT 포즈가 있으므로 이 방식이 가능하다.

전제 확인(2026-08-14): `mps/slam/closed_loop_trajectory.csv` 와 ADT `aria_trajectory.csv`
가 같은 타임스탬프에서 9mm 이내로 일치 → **MPS world 프레임 = ADT scene 프레임**.
따라서 포인트를 그대로 T_world_camera 로 변환하면 된다.

주의: 원본 포인트에는 ±43km 짜리 발산 점이 섞여 있다(전체의 0.1%). 불확실도
(`inv_dist_std`, `dist_std`) 필터가 선택이 아니라 필수다.
"""
import numpy as np
import pandas as pd

# GT 뎁스 대조로 튜닝(2026-08-14, party seq102 5프레임): 이 조합에서 프레임당 약 3,400개
# 앵커가 남고 그중 65%가 GT 뎁스와 ±10% 안에 든다. 더 조이면 앵커 수가 줄어 적합이
# 불안정해지고, 풀면 인라이어가 52%까지 떨어진다.
INV_STD_MAX = 0.002
DIST_STD_MAX = 0.01
CELL = 8              # 픽셀
SPREAD_MAX = 0.15     # 셀 내부 상대 깊이 폭. 넘으면 가림 경계로 보고 셀 통째로 버린다


def load_semidense(path, inv_std_max=INV_STD_MAX, dist_std_max=DIST_STD_MAX, bbox=None):
    """반-조밀 포인트 → (xyz [N,3], weight [N]).

    weight 는 역깊이 분산의 역수 — 관측이 확실한 점일수록 정합에서 발언권이 크다.
    """
    df = pd.read_csv(path)
    m = (df["inv_dist_std"].values < inv_std_max) & (df["dist_std"].values < dist_std_max)
    xyz = df.loc[m, ["px_world", "py_world", "pz_world"]].to_numpy(np.float64)
    inv_std = df.loc[m, "inv_dist_std"].to_numpy(np.float64)
    if bbox is not None:
        lo, hi = bbox
        keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        xyz, inv_std = xyz[keep], inv_std[keep]
    w = 1.0 / np.maximum(inv_std, 1e-4) ** 2
    return xyz, w / w.mean()


class AnchorProjector:
    """전역 포인트를 프레임 좌표로 투영한다."""

    def __init__(self, xyz, weight, K, width, height):
        self.xyz = np.ascontiguousarray(xyz)
        self.weight = weight
        self.K = K
        self.W, self.H = width, height

    def frame(self, T_wc, z_min=0.2, z_max=12.0, cell=CELL, spread_max=SPREAD_MAX):
        """T_world_camera → (u, v, z, w). 픽셀 정수 좌표.

        MPS 포인트는 궤적 전체에서 모은 것이라 현재 시점에서 벽 뒤에 가려진 점도 그대로
        투영된다. `cell` 픽셀 격자로 묶어 **셀 중앙값**을 대표로 삼고, 셀 내부 깊이 폭이
        크면(가림 경계) 셀을 통째로 버린다.

        ⚠️ 처음엔 셀별 **최근접**을 남겼는데(값싼 z-버퍼) 정반대 결과가 나왔다 —
        노이즈 근접점 하나가 정상 점들을 가려 앵커 중앙값이 GT 뎁스의 0.3~0.5배로
        내려앉았다. 반-조밀 포인트는 표면을 다 덮지 않으므로 '가장 가까운 게 보이는
        표면'이라는 전제가 성립하지 않는다. 중앙값으로 바꾸자 비율 0.98→1.00,
        인라이어 24%→65% (`scripts/selftest_align.py` A 항목).
        """
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        pc = (self.xyz - t) @ R                       # world → camera  (R^T (p - t))
        z = pc[:, 2]
        ok = (z > z_min) & (z < z_max)
        empty = (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0), np.empty(0))
        if not ok.any():
            return empty
        pc, z, w = pc[ok], z[ok], self.weight[ok]

        uv = pc[:, :2] / z[:, None] @ self.K[:2, :2].T + self.K[:2, 2]
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        ok = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        u, v, z, w = u[ok], v[ok], z[ok], w[ok]
        if len(z) == 0:
            return empty

        key = (v // cell).astype(np.int64) * (self.W // cell + 1) + (u // cell)
        order = np.lexsort((z, key))                  # 셀별로 묶고 셀 안에서 z 오름차순
        key, u, v, z, w = key[order], u[order], v[order], z[order], w[order]
        start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        cnt = np.diff(np.r_[start, len(key)])
        mid = start + cnt // 2
        lo = start + (cnt * 2) // 10
        hi = start + np.minimum((cnt * 8) // 10, cnt - 1)
        keep = (cnt <= 2) | ((z[hi] - z[lo]) / np.maximum(z[mid], 1e-6) <= spread_max)
        sel = mid[keep]
        return u[sel], v[sel], z[sel], w[sel]


def scene_bbox(poses, margin=8.0):
    """궤적 주변 상자 — 발산한 포인트를 통째로 잘라내는 값싼 1차 방어선."""
    t = poses[:, :3, 3]
    return t.min(axis=0) - margin, t.max(axis=0) + margin
