"""전역 회전 평균 + 선형 병진/스케일 해 — 드리프트를 원리적으로 없애는 정석 경로.

**왜 바꾸는가.** 블록 좌표 하강(윈도우를 하나씩 이웃에 다시 맞추기)으로는 회전이
전역 수렴하지 않는다. 실측: 랜드마크 내부 산포는 0.141→0.046m 로 잘 줄었는데 GT 대비
자세는 11.35°→13.65° 로 **오히려 나빠졌다**. 자기들끼리 일관되지만 비틀린 해로 간다.

**정석(SfM 표준).** 회전과 병진을 분리해 각각 전역으로 푼다.

    1) 상대 회전 추출 — 공통 랜드마크 3개 이상인 **모든 윈도우 쌍**에서 Procrustes.
       인접 쌍만이 아니라 재방문 쌍까지 포함되므로 이게 진짜 루프 제약이다.
    2) 전역 회전 평균 — 상대 회전들로 {R_i} 를 한 번에 푼다(Weiszfeld 반복).
       순차 누적이 아니라 전역 해라 드리프트가 쌓일 자리가 없다.
    3) 회전을 고정하면 나머지가 **선형**이 된다:
           s_i (R_i x_ik) + t_i − L_k = 0
       미지수 (s_i, t_i, L_k) 에 대해 선형이므로 희소 최소자승 한 번으로 끝난다.
       IRLS 로 몇 번 재가중하면 이동한 물체·오연관이 걸러진다.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

MIN_COMMON = 3          # 상대 회전을 뽑기 위한 최소 공통 랜드마크
ROT_ITERS = 60
LSQ_IRLS = 4
HUBER_M = 0.20


def _proc(X, Y, w=None, iters=4):
    """Y ≈ s R X + t. (s, R, t, 잔차중앙)"""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    w = np.ones(len(X)) if w is None else w
    for _ in range(iters):
        ww = w / max(w.sum(), 1e-9)
        mx, my = (ww[:, None] * X).sum(0), (ww[:, None] * Y).sum(0)
        A, B = X - mx, Y - my
        S = (ww[:, None] * A).T @ B
        U, D, Vt = np.linalg.svd(S)
        d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        R = (U @ np.diag([1.0, 1.0, d]) @ Vt).T
        s = max(float((D * np.array([1.0, 1.0, d])).sum()
                      / max((ww * (A * A).sum(1)).sum(), 1e-12)), 1e-6)
        t = my - s * R @ mx
        r = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
        w = np.where(r <= HUBER_M, 1.0, HUBER_M / np.maximum(r, 1e-9))
    return s, R, t, float(np.median(r))


def relative_rotations(obs, min_common=MIN_COMMON):
    """모든 윈도우 쌍에서 상대 회전. R_ij 는 'j 좌표계 → i 좌표계' 회전."""
    keys = [set(o["pts"]) for o in obs]
    out = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            common = sorted(keys[i] & keys[j])
            if len(common) < min_common:
                continue
            X = np.array([obs[j]["pts"][k] for k in common])   # j 로컬
            Y = np.array([obs[i]["pts"][k] for k in common])   # i 로컬
            s, R, t, res = _proc(X, Y)
            out.append({"i": i, "j": j, "R": R, "n": len(common), "res": res})
    return out


def _rot_avg(Rs, w):
    M = np.tensordot(np.asarray(w), np.asarray(Rs), axes=(0, 0))
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def rotation_averaging(n, edges, init=None, iters=ROT_ITERS):
    """상대 회전 집합 → 전역 회전 {R_i}. 윈도우 0 고정."""
    R = [np.eye(3) for _ in range(n)] if init is None else [np.array(r) for r in init]
    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e["i"]].append((e["j"], e["R"], e))          # R_i ≈ R_ij R_j
        adj[e["j"]].append((e["i"], e["R"].T, e))        # R_j ≈ R_ij^T R_i
    for _ in range(iters):
        for i in range(1, n):
            if not adj[i]:
                continue
            cand = [Rij @ R[j] for j, Rij, _ in adj[i]]
            w = np.array([e["n"] / (1.0 + (e["res"] / HUBER_M) ** 2) for _, _, e in adj[i]])
            if w.sum() <= 0:
                continue
            R[i] = _rot_avg(cand, w / w.sum())
    # 잔차 진단
    resid = [np.degrees(np.arccos(np.clip(
        (np.trace((e["R"] @ R[e["j"]]).T @ R[e["i"]]) - 1) / 2, -1, 1))) for e in edges]
    return R, (float(np.median(resid)) if resid else np.nan)


def solve_translation_scale(obs, R, cam_weight=2.0, iters=LSQ_IRLS):
    """회전 고정 → (s_i, t_i, 랜드마크/카메라 전역위치) 를 희소 선형 최소자승으로.

        s_i (R_i x) + t_i − L = 0
    """
    n = len(obs)
    lm_windows = {}
    for i, o in enumerate(obs):
        for k in o["pts"]:
            lm_windows.setdefault(k, []).append(i)
    lm = sorted([k for k, v in lm_windows.items() if len(v) >= 2])
    cam_windows = {}
    for i, o in enumerate(obs):
        for f in o["cam"]:
            cam_windows.setdefault(f, []).append(i)
    cams = sorted([f for f, v in cam_windows.items() if len(v) >= 2])

    lm_idx = {k: p for p, k in enumerate(lm)}
    cam_idx = {f: len(lm) + p for p, f in enumerate(cams)}
    n_pt = len(lm) + len(cams)
    # 미지수: s_i (n), t_i (3n), P_j (3 n_pt).  윈도우 0 은 s=1, t=0 로 고정
    def col_s(i): return i
    def col_t(i): return n + 3 * i
    def col_p(j): return 4 * n + 3 * j

    rows = []
    for i, o in enumerate(obs):
        for k, x in o["pts"].items():
            if k in lm_idx:
                rows.append((i, np.asarray(x, float), lm_idx[k], 1.0))
        for f, c in o["cam"].items():
            if f in cam_idx:
                rows.append((i, np.asarray(c[:3, 3], float), cam_idx[f], cam_weight))

    w = np.array([r[3] for r in rows])
    for _ in range(iters):
        I, J, V, B = [], [], [], []
        eq = 0
        for (i, x, pj, _), wi in zip(rows, w):
            rx = R[i] @ x
            for a in range(3):
                if i > 0:
                    I += [eq, eq]; J += [col_s(i), col_t(i) + a]; V += [wi * rx[a], wi]
                    b = 0.0
                else:
                    b = -wi * rx[a]        # s_0=1, t_0=0 고정
                I.append(eq); J.append(col_p(pj) + a); V.append(-wi)
                B.append(b)
                eq += 1
        # 게이지 고정: s_i>0 유도용 약한 사전(스케일이 0으로 수축하는 것 방지)
        for i in range(1, n):
            I.append(eq); J.append(col_s(i)); V.append(0.05); B.append(0.05)
            eq += 1
        A = coo_matrix((V, (I, J)), shape=(eq, 4 * n + 3 * n_pt))
        sol = lsqr(A.tocsr(), np.array(B), atol=1e-10, btol=1e-10, iter_lim=4000)[0]
        s = np.concatenate([[1.0], sol[1:n]])
        t = np.concatenate([[np.zeros(3)], sol[n + 3:n + 3 * n].reshape(n - 1, 3)])
        P = sol[4 * n:].reshape(n_pt, 3)
        # IRLS 재가중
        r = np.array([np.linalg.norm(s[i] * (R[i] @ x) + t[i] - P[pj])
                      for (i, x, pj, _) in rows])
        w = np.array([r0[3] for r0 in rows]) * np.where(
            r <= HUBER_M, 1.0, HUBER_M / np.maximum(r, 1e-9))
    return s, t, P, float(np.median(r))


def run(obs, init_R=None):
    edges = relative_rotations(obs)
    R, rot_res = rotation_averaging(len(obs), edges, init=init_R)
    s, t, P, res = solve_translation_scale(obs, R)
    return ([{"s": float(s[i]), "R": R[i], "t": t[i]} for i in range(len(obs))],
            {"edges": len(edges), "rot_residual_deg": rot_res, "lsq_residual_m": res})
