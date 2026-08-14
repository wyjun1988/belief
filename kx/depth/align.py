"""윈도우 간 뎁스 정합 — 역깊이 affine 을 전역 앵커에 맞추고 시간축으로 정칙화한다.

**문제.** DA3 를 겹치는 슬라이딩 윈도우로 돌리면 윈도우마다 (scale, shift) 가 조금씩
다르다. 그대로 융합하면 맵이 프레임마다 "숨쉬고", Khronos 는 그 호흡을 **물체가
움직였다**고 읽는다 — 즉 뎁스 드리프트가 곧 가짜 변화 감지가 된다.

**SOTA 지형.** 일반 비디오에서는 윈도우끼리 이어붙일 수밖에 없어서, 최근 연구
(DVD 의 global affine coherence, DepthSync, StableDPT)가 인접 윈도우 사이를 단일
선형 변환으로 보정하는 데 집중한다. 그러나 그 계열은 전부 **상대 정합**이라 긴
시퀀스에서 스케일이 서서히 흐른다.

**우리 설정은 더 유리하다.** 정확한 포즈와 전역 반-조밀 포인트가 있으므로 상대가
아니라 **절대 정합**을 할 수 있다:

    T1  DA3 를 포즈 조건부 다중뷰로 (윈도우 내부는 모델이 알아서 일관되게)
    T2  프레임마다 역깊이 affine (a_t, b_t) 를 전역 앵커에 로버스트 피팅  ← 드리프트 0
    T3  (a_t, b_t) 를 시간축 2차 차분으로 정칙화 + 앵커 빈곤 프레임 보간

정합식은 **역깊이(disparity) 공간**에서 푼다:  1/D̂ = a·(1/D_pred) + b.
뎁스 공간의 곱셈 스케일만으로는 모델의 near/far 편향을 못 잡고, 역깊이 affine 이
단안 뎁스 문헌의 표준 정합형(scale-shift invariant)이기도 하다.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

TUKEY_C = 4.685          # Tukey biweight 표준 상수 (정규 오차 기준 95% 효율)
S_MAX = 0.02             # 로버스트 스케일 상한(상대). 좋은 앵커의 잔차 중앙값이 ~2~3%
MIN_INLIERS = 30         # 이보다 적으면 그 프레임은 시간 사전으로만 채운다
RANSAC_ITERS = 100
RANSAC_TOL = 0.05        # **상대** 오차. 0.05 = 5%

# ⚠️ 잔차는 반드시 상대(relative)로 잰다. 처음엔 역깊이 절대 잔차를 썼는데, 가까운
# 앵커(역깊이 2.0)는 10% 오차만 나도 절대 잔차 0.2 라 전부 이상치가 되고 먼 앵커
# (역깊이 0.1)는 어떤 모델이든 통과한다. 그 결과 RANSAC 이 **먼 점만 맞추는 평평한
# 퇴화 해(a≈0, 즉 '깊이 일정')** 에 최고점을 준다 — 실제로 프레임의 절반에서
# a=0.03 짜리 해가 나왔다. 상대 잔차는 거리에 무관하게 같은 비중을 준다.
#
#   상대 잔차:  r = (a·x + b)/y − 1        (x = 1/D_pred, y = 1/z_anchor)
#   → 1/y 로 스케일한 선형계로 그대로 풀 수 있다: a·(x/y) + b·(1/y) = 1


def _ransac_affine(x, y, w, rng, iters=RANSAC_ITERS, tol=RANSAC_TOL):
    """2점 최소표본 RANSAC 으로 (a, b) 초기값. 점수는 상대 오차 기준."""
    n = len(x)
    if n < 8:
        return None
    best, best_score = None, -1.0
    idx = rng.integers(0, n, size=(iters, 2))
    for i, j in idx:
        if abs(x[i] - x[j]) < 1e-6:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if a <= 0:                                  # 뎁스 순서가 뒤집히는 해는 버린다
            continue
        b = y[i] - a * x[i]
        score = w[np.abs((a * x + b) / y - 1.0) < tol].sum()
        if score > best_score:
            best, best_score = (a, b), score
    return best


def _irls(x, y, w, a0, b0, iters=12):
    """Tukey 재하강 IRLS (상대 잔차). (a, b, inlier_mask) 반환.

    ⚠️ 처음엔 Huber 를 썼는데 첫 스텝부터 해가 무너졌다. 이유는 **이상치가 편측**이기
    때문이다 — 가려진 앵커는 언제나 보이는 표면보다 *멀다*(z_anchor > d_true) 이므로
    잔차가 한 방향으로만 쏠린다. Huber 는 재하강하지 않아 그런 이상치가 선형으로
    계속 끌어당기고, 게다가 적응 스케일 s 가 0.04→0.61 로 폭주하면서 이상치를 전부
    인라이어로 받아들여 '깊이 일정(a≈0)' 해로 수렴했다.

    두 가지로 고친다: (1) Tukey biweight — 문턱 밖은 가중치 정확히 0, (2) 스케일 상한
    `S_MAX` — 좋은 앵커의 상대 오차가 3% 안팎임을 아는데 s 를 자유롭게 둘 이유가 없다.
    """
    xs, bs = x / y, 1.0 / y                          # 각 방정식을 1/y 로 스케일
    a, b = a0, b0
    for _ in range(iters):
        r = a * xs + b * bs - 1.0
        s = min(1.4826 * np.median(np.abs(r)) + 1e-9, S_MAX)
        u = np.abs(r) / (TUKEY_C * s)
        ww = w * np.where(u < 1.0, (1.0 - np.minimum(u, 1.0) ** 2) ** 2, 0.0)
        if (ww > 0).sum() < MIN_INLIERS:
            break
        A = np.array([[(ww * xs * xs).sum(), (ww * xs * bs).sum()],
                      [(ww * xs * bs).sum(), (ww * bs * bs).sum()]])
        rhs = np.array([(ww * xs).sum(), (ww * bs).sum()])
        try:
            a_new, b_new = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break
        if a_new <= 0:                               # 퇴화 — 직전 해를 유지하고 중단
            break
        if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-7:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    if a <= 0:
        return None
    r = a * xs + b * bs - 1.0
    s = min(1.4826 * np.median(np.abs(r)) + 1e-9, S_MAX)
    return a, b, np.abs(r) < TUKEY_C * s


def fit_frame(depth_pred, u, v, z_anchor, w, rng, min_depth=0.1):
    """T2 — 한 프레임의 역깊이 affine.

    Returns (a, b, n_inlier, rmse_inv) 또는 앵커가 모자라면 None.
    """
    d = depth_pred[v, u]
    ok = (d > min_depth) & np.isfinite(d) & (z_anchor > min_depth)
    if ok.sum() < MIN_INLIERS:
        return None
    x = 1.0 / d[ok]                       # 모델 역깊이
    y = 1.0 / z_anchor[ok]                # 앵커 역깊이 (진짜 거리)
    ww = w[ok]

    init = _ransac_affine(x, y, ww, rng)
    if init is None:
        init = (np.median(y) / max(np.median(x), 1e-9), 0.0)
    out = _irls(x, y, ww, *init)
    if out is None:
        return None
    a, b, inl = out
    if inl.sum() < MIN_INLIERS:
        return None
    rel = (a * x[inl] + b) / y[inl] - 1.0
    return (float(a), float(b), int(inl.sum()), float(np.sqrt(np.mean(rel ** 2))),
            float(inl.mean()))


def gate(a, b, n_inlier, rmse, inlier_ratio,
         max_log_dev=0.35, min_ratio=0.20, max_rmse=0.15):
    """프레임별 적합 신뢰도 게이트 — 파탄난 소수 프레임을 T3 보간으로 넘긴다.

    ⚠️ 실측으로 알게 된 것: DA3 를 포즈 조건부로 돌리면 출력이 **이미 거의 metric**
    이라 대부분의 프레임에서 a≈1.03, b≈−0.01 이 나온다. 그런데 소수 프레임에서
    a 가 0.001 이나 2.76 으로 튀고(앵커가 한 깊이에 몰린 근접 화면 등) 그 몇 장이
    시퀀스 평균 AbsRel 을 0.12 → 0.72 로 망가뜨렸다. 인라이어 수만으로는 이 프레임을
    못 거른다 — 인라이어가 1800개나 되면서 해가 틀렸기 때문이다.

    그래서 세 가지로 거른다: (1) 시퀀스 중앙값 대비 스케일 이탈, (2) 인라이어 비율,
    (3) 상대 잔차. 걸린 프레임은 신뢰도 0 이 되어 `smooth_sequence` 가 양옆에서 메운다.
    """
    a = np.asarray(a, float)
    n = np.asarray(n_inlier, float)
    ok = n >= MIN_INLIERS
    if not ok.any():
        return np.zeros(len(a), bool)
    med = np.median(a[ok])
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(np.log(np.maximum(a, 1e-9) / max(med, 1e-9)))
    return (ok
            & (dev <= max_log_dev)
            & (np.asarray(inlier_ratio, float) >= min_ratio)
            & (np.nan_to_num(np.asarray(rmse, float), nan=9.9) <= max_rmse))


def smooth_sequence(a, b, n_inlier, lam=1.0, min_inliers=MIN_INLIERS):
    """T3 — (a_t, b_t) 를 시간축으로 정칙화하고 앵커 빈곤 프레임을 메운다.

    DVD 계열이 관측한 "global affine coherence"(윈도우 간 차이가 사실상 하나의 선형
    변환) 를 사전지식으로 쓴다: 진짜 (a,b) 는 시간에 대해 거의 평탄해야 하므로
    2차 차분에 벌점을 준다. 관측 신뢰도는 인라이어 수에 비례시킨다 — 인라이어 0인
    프레임은 자동으로 양옆에서 보간된 값을 받는다.

        min_θ  Σ_t c_t (θ_t − θ̂_t)²  +  λ Σ_t (θ_{t−1} − 2θ_t + θ_{t+1})²
    """
    F = len(a)
    c = np.where(np.asarray(n_inlier) >= min_inliers, np.asarray(n_inlier, float), 0.0)
    c = c / max(c.max(), 1.0)
    if c.sum() == 0:
        raise RuntimeError("앵커 인라이어가 있는 프레임이 하나도 없다 — 포즈/캘리브 확인")

    # D2: (F-2, F) 2차 차분
    D2 = diags([np.ones(F - 2), -2 * np.ones(F - 2), np.ones(F - 2)],
               offsets=[0, 1, 2], shape=(F - 2, F), format="csr")
    A = diags(c, format="csr") + lam * (D2.T @ D2)

    out = []
    for theta in (a, b):
        th = np.asarray(theta, float).copy()
        th[c == 0] = 0.0                                  # 관측 없는 프레임은 벌점만 남는다
        out.append(spsolve(A.tocsc(), c * th))
    return out[0], out[1], c > 0


def apply_affine(depth_pred, a, b, d_min=0.05, d_max=20.0):
    """역깊이 affine 을 적용해 정합된 미터 뎁스를 만든다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = a / np.maximum(depth_pred, 1e-6) + b
        d = 1.0 / inv
    d[~np.isfinite(d)] = 0.0
    d[(d < d_min) | (d > d_max)] = 0.0
    return d.astype(np.float32)
