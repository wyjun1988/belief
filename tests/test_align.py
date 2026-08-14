"""뎁스 정합 — 역깊이 affine 을 전역 앵커에 맞춘다.

지켜야 할 계약:
  · 알려진 (a,b) 로 오염시킨 뎁스를 되돌린다 (round-trip)
  · **편측 이상치**(가려진 앵커는 언제나 더 멀다)에 끌려가지 않는다
  · 파탄난 프레임은 게이트가 걸러 시간 사전으로 넘긴다
"""
import numpy as np

from kx.depth.align import apply_affine, fit_frame, gate, smooth_sequence

RNG = np.random.default_rng(0)


def synth(n=600, a=1.0, b=0.0, noise=0.0):
    """앵커 n개와 예측 뎁스. **정합식 방향은 `1/z = a·(1/d) + b`** 다.

    즉 (a, b) 는 예측을 진짜로 되돌리는 계수이고, 여기서는 그 관계가 정확히 성립하도록
    d 를 거꾸로 만든다. 방향을 뒤집어 쓰면 a 가 역수로 나온다(처음 그렇게 틀렸다).
    """
    z = RNG.uniform(0.6, 6.0, n)                       # 진짜 거리
    d = a / np.maximum(1.0 / z - b, 1e-6)              # 1/z = a/d + b 가 되도록
    if noise:
        d = d * (1.0 + RNG.normal(0, noise, n))
    depth = np.zeros((32, n), np.float32)
    depth[0] = d
    u = np.arange(n)
    v = np.zeros(n, int)
    return depth, u, v, z, np.ones(n)


def test_round_trip_recovers_affine():
    depth, u, v, z, w = synth(a=0.85, b=0.05)
    a, b, ni, rmse, ratio = fit_frame(depth, u, v, z, w, RNG)
    assert abs(a - 0.85) < 0.02 and abs(b - 0.05) < 0.01
    assert ratio > 0.95 and rmse < 0.01
    rec = apply_affine(depth, a, b)
    assert np.abs(rec[0] - z).max() < 0.05


def test_survives_one_sided_outliers():
    """가려진 앵커는 항상 *더 멀다* — 편측 오염에도 해가 무너지면 안 된다.

    Huber + 절대 잔차 조합에서 실제로 a≈0.03(깊이 일정) 퇴화 해가 나왔다.
    """
    depth, u, v, z, w = synth(n=800, a=1.0, b=0.0)
    bad = RNG.choice(800, 240, replace=False)
    z = z.copy()
    z[bad] *= RNG.uniform(1.5, 4.0, len(bad))          # 앵커만 더 멀게 (편측)
    a, b, ni, rmse, ratio = fit_frame(depth, u, v, z, w, RNG)
    assert 0.9 < a < 1.1, "편측 이상치에 끌려갔다 (a=%.3f)" % a
    assert 0.5 < ratio < 0.85, "이상치를 인라이어로 흡수했다 (ratio=%.2f)" % ratio


def test_gate_rejects_scale_outlier_frames():
    """인라이어가 많아도 스케일이 튀면 걸러야 한다 — 몇 장이 시퀀스를 망친다."""
    a = np.array([1.02, 1.03, 1.01, 2.76, 1.02, 0.001])
    ok = gate(a, np.zeros(6), n_inlier=np.full(6, 1800),
              rmse=np.full(6, 0.02), inlier_ratio=np.full(6, 0.8))
    assert list(ok) == [True, True, True, False, True, False]


def test_gate_rejects_low_inliers():
    ok = gate(np.full(4, 1.0), np.zeros(4), n_inlier=[1800, 5, 1800, 1800],
              rmse=[0.02, 0.02, 0.9, 0.02], inlier_ratio=[0.8, 0.8, 0.8, 0.05])
    assert list(ok) == [True, False, False, False]


def test_smooth_fills_frames_without_anchors():
    """인라이어 0인 프레임은 양옆에서 보간된 값을 받는다."""
    F = 20
    a = np.full(F, 1.0)
    n = np.full(F, 500)
    a[10], n[10] = 99.0, 0                              # 관측 없음 + 쓰레기 값
    sa, sb, used = smooth_sequence(a, np.zeros(F), n, lam=1.0)
    assert not used[10]
    assert abs(sa[10] - 1.0) < 0.05, "보간이 아니라 쓰레기 값을 썼다 (%.2f)" % sa[10]
