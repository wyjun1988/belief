"""증분 SfM 핵심 함수의 합성 데이터 회귀 검출기.

실측으로 잡았던 사고들을 고정한다:
- 삼각측량 검증 없이는 지도가 오염된다(씨앗 BA RMS 166,215px)
- 아일랜드 병합 sim3 는 스케일 차이를 흡수해야 한다(v1 스케일 구간별 1.18~2.16)
"""
import numpy as np
import pytest

from scripts.incremental_sfm import (accept_point, ate, pnp, sim3_apply_pose,
                                     umeyama3)


def look_at(C, target=np.zeros(3)):
    """C 에서 target 을 보는 카메라→월드 포즈."""
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0.001])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = C
    return T


K = np.array([[350.0, 0, 352], [0, 350, 352], [0, 0, 1]])


def project(T, X):
    R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    Xc = R @ X + t
    return np.array([K[0, 0] * Xc[0] / Xc[2] + K[0, 2],
                     K[1, 1] * Xc[1] / Xc[2] + K[1, 2]])


class TestAcceptPoint:
    def test_good_point_passes(self):
        X = np.array([0.3, 0.1, 0.2])
        views = [(look_at(np.array(c)), project(look_at(np.array(c)), X))
                 for c in [[3, 0, 0], [2.5, 1, 1], [2, -1, 2]]]
        assert accept_point(X, views, K)

    def test_behind_camera_rejected(self):
        T = look_at(np.array([3.0, 0, 0]))
        X_behind = np.array([6.0, 0, 0])          # 카메라 뒤
        assert not accept_point(X_behind, [(T, (352, 352))], K)

    def test_far_point_rejected(self):
        T = look_at(np.array([3.0, 0, 0]))
        X = np.array([-40.0, 0, 0])               # 25m 밖 — 시차 부족의 신호
        assert not accept_point(X, [(T, project(T, X))], K)

    def test_bad_reprojection_rejected(self):
        X = np.array([0.3, 0.1, 0.2])
        views = [(look_at(np.array(c)), project(look_at(np.array(c)), X) + 50)
                 for c in [[3, 0, 0], [2.5, 1, 1]]]
        assert not accept_point(X, views, K)

    def test_none_and_nan(self):
        assert not accept_point(None, [], K)
        assert not accept_point(np.array([np.nan, 0, 0]), [], K)


class TestUmeyama3:
    def test_recovers_random_sim3(self):
        rng = np.random.default_rng(7)
        A = rng.normal(size=(8, 3))
        ang = 0.7
        R = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        s, t = 1.7, np.array([0.5, -2.0, 1.0])   # 아일랜드 간 v1 스케일차 규모
        B = (s * (R @ A.T)).T + t
        s2, R2, t2 = umeyama3(A, B)
        assert abs(s2 - s) < 1e-9
        assert np.allclose(R2, R)
        assert np.allclose(t2, t)

    def test_three_points_minimum(self):
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        B = A * 0.5 + np.array([1, 1, 1])
        s, R, t = umeyama3(A, B)
        assert abs(s - 0.5) < 1e-9
        assert np.allclose((s * (R @ A.T)).T + t, B)


class TestSim3ApplyPose:
    def test_center_transforms_rotation_stays_orthonormal(self):
        T = look_at(np.array([2.0, 1.0, -1.0]))
        ang = 0.3
        R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                      [-np.sin(ang), 0, np.cos(ang)]])
        s, t = 1.5, np.array([1.0, 0, 2.0])
        O = sim3_apply_pose(T, s, R, t)
        # 카메라 중심은 sim3 를 그대로 따르고, 회전은 정규직교를 유지해야 한다
        assert np.allclose(O[:3, 3], s * R @ T[:3, 3] + t)
        assert np.allclose(O[:3, :3] @ O[:3, :3].T, np.eye(3), atol=1e-12)

    def test_projection_consistency(self):
        # 지도와 포즈에 같은 sim3 를 적용하면 재투영이 보존된다 (병합의 핵심 불변량)
        X = np.array([0.2, -0.1, 0.4])
        T = look_at(np.array([2.0, 1.0, -1.0]))
        uv = project(T, X)
        ang = -0.5
        R = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        s, t = 0.7, np.array([-1.0, 2.0, 0.5])
        assert np.allclose(project(sim3_apply_pose(T, s, R, t), s * R @ X + t), uv)


class TestPnP:
    def test_recovers_pose(self):
        rng = np.random.default_rng(3)
        pts = rng.normal(scale=0.8, size=(12, 3))
        T = look_at(np.array([3.0, 0.5, 0.5]))
        uv = [project(T, X) for X in pts]
        r = pnp(list(pts), uv, K)
        assert r is not None
        Te, _ = r
        assert np.linalg.norm(Te[:3, 3] - T[:3, 3]) < 1e-3

    def test_too_few_points(self):
        assert pnp([np.zeros(3)] * 4, [(0, 0)] * 4, K) is None


class TestAte:
    def test_sim3_invariance_and_median(self):
        rng = np.random.default_rng(1)
        gt = np.tile(np.eye(4), (20, 1, 1))
        gt[:, :3, 3] = rng.normal(size=(20, 3))
        est = {i: gt[i].copy() for i in range(20)}
        # 전역 sim3 를 걸어도 ATE 는 0 이어야 한다 (정렬 후 채점이므로)
        ang = 0.4
        R = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        for i in est:
            est[i] = sim3_apply_pose(est[i], 1.3, R, np.array([1, 2, 3.0]))
        med, p90, s = ate(est, gt)
        assert med < 1e-6
        assert abs(s - 1 / 1.3) < 1e-6
