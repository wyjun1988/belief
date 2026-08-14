"""루프 클로저 — 윈도우별 sim3 를 물체 랜드마크로 전역 보정한다.

**문제.** DA3 윈도우를 순차로 이어붙이면 국소적으로는 정확한데(240프레임 구간 ATE
0.17m·자세 2.6°) 918프레임 76윈도우를 지나며 드리프트가 쌓여 ATE 1.43m·자세 11.4°가
된다. 그 결과 물체 위치오차가 2.7m 로 벌어져 씬그래프가 무너진다.

**원인.** 순차 체이닝은 **재방문 정보를 통째로 버린다.** 91초 동안 착용자는 같은 소파·
식탁·선반을 몇 번씩 다시 보는데, 인접 윈도우 사이 제약만 쓰면 그 관측들이 서로를 잡아줄
기회가 없다.

**해법.** 물체를 랜드마크로 삼아 전역 최적화한다. 같은 인스턴스가 윈도우 i 와 j 에서
보였다면 두 관측은 **같은 3D 점**이어야 한다 — 그게 곧 루프 제약이다. 카메라도 같은
방식으로 넣는다(겹치는 프레임은 두 윈도우가 같은 카메라를 본 것이므로 랜드마크와 동형).

**움직인 물체는?** 라벨 없이 처리한다. 관측의 대부분(179/213)이 정적이므로, 랜드마크
위치를 **로버스트 평균**으로 잡고 sim3 도 로버스트로 풀면 이동한 물체는 자동으로 이상치가
된다. 이동 라벨을 쓰지 않으므로 GT 세그멘테이션을 SAM 으로 바꿔도 논리가 그대로다
(연관만 ReID 로 대체하면 된다 — 잘못된 연관도 같은 로버스트 손실이 걸러낸다).

알고리즘은 블록 좌표 하강이다(가벼운 번들 조정):
    1) 현재 sim3 로 각 랜드마크의 전역 위치를 로버스트 평균으로 확정
    2) 각 윈도우의 sim3 를 그 랜드마크들에 로버스트 정합
    3) 반복
회전은 중심점만으로 구속되지 않으므로(1차원 궤적 문제) **카메라 자세를 따로 평균**해
회전을 먼저 정하고 스케일·평행이동을 푼다 — `pose_stitch` 에서 배운 것과 같다.
"""
import numpy as np

MIN_WINDOWS = 2         # 이만큼의 서로 다른 윈도우에서 보여야 랜드마크로 쓴다
MIN_OBS = 3             # 윈도우 안에서 이만큼 관측돼야 그 윈도우의 대표점을 만든다
ITERS = 12
HUBER_M = 0.25          # m. 이보다 큰 잔차는 선형으로만 끌어당긴다(이동 물체·오연관 대비)


def _huber_w(r, delta=HUBER_M):
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))


def _robust_mean(P, W=None, iters=6):
    """(N,3) 점들의 로버스트 평균 — 이동한 물체의 관측을 눌러 없앤다."""
    P = np.asarray(P, float)
    w = np.ones(len(P)) if W is None else np.asarray(W, float)
    m = (w[:, None] * P).sum(0) / max(w.sum(), 1e-9)
    for _ in range(iters):
        r = np.linalg.norm(P - m, axis=1)
        ww = w * _huber_w(r)
        m = (ww[:, None] * P).sum(0) / max(ww.sum(), 1e-9)
    return m, float(np.median(np.linalg.norm(P - m, axis=1)))


def _rot_average(Rs, w=None):
    M = np.average(np.asarray(Rs), axis=0, weights=w)
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


MIN_LM_FOR_ROT = 6      # 이만큼의 랜드마크가 있으면 회전까지 랜드마크로 푼다


def _fit_sim3(X, Y, R_fixed=None, w=None, iters=6, solve_R=False):
    """X → Y 의 로버스트 sim3.

    ⚠️ 회전을 **랜드마크로 풀어야 한다.** 처음엔 카메라 자세 평균으로 R 을 먼저 고정하고
    스케일·평행이동만 랜드마크로 풀었는데, 그러면 랜드마크가 회전을 전혀 보지 못한다 —
    루프 클로저를 돌려도 자세 오차가 11.35° 에서 **한 자리도 안 움직였다.**
    (카메라 중심으로 R 을 풀 수 없었던 이유는 1.4m 궤적이 1차원이라 퇴화해서인데,
     방 전체에 흩어진 랜드마크는 그 문제가 없다.)
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    R = np.eye(3) if R_fixed is None else R_fixed
    s, t = 1.0, np.zeros(3)
    for _ in range(iters):
        if solve_R:
            mx = (w[:, None] * X).sum(0) / max(w.sum(), 1e-9)
            my = (w[:, None] * Y).sum(0) / max(w.sum(), 1e-9)
            A, B = X - mx, Y - my
            S = (w[:, None] * A).T @ B
            U, D, Vt = np.linalg.svd(S)
            d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
            R = (U @ np.diag([1.0, 1.0, d]) @ Vt).T
            s = float((D * np.array([1.0, 1.0, d])).sum()
                      / max((w * (A * A).sum(1)).sum(), 1e-12))
            s = max(s, 1e-6)
            t = my - s * R @ mx
        else:
            Xr = (R @ X.T).T
            mx = (w[:, None] * Xr).sum(0) / max(w.sum(), 1e-9)
            my = (w[:, None] * Y).sum(0) / max(w.sum(), 1e-9)
            A, B = Xr - mx, Y - my
            s = max(float((w * (A * B).sum(1)).sum()
                          / max((w * (A * A).sum(1)).sum(), 1e-12)), 1e-6)
            t = my - s * mx
        r = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
        w = _huber_w(r)
    return s, R, t


class LoopCloser:
    """윈도우별 sim3 를 랜드마크로 전역 보정.

    obs[i] = {"pts": {key: (3,)}, "cam": {frame: (4,4) c2w}}  — **윈도우 i 의 로컬 좌표계**
    """

    def __init__(self, obs, init):
        self.obs = obs
        self.T = [{"s": a["s"], "R": np.array(a["R"]), "t": np.array(a["t"])} for a in init]

    def _g(self, i, p):
        a = self.T[i]
        return a["s"] * (a["R"] @ np.asarray(p)) + a["t"]

    def run(self, iters=ITERS, verbose=False):
        keys = {}
        for i, o in enumerate(self.obs):
            for k in o["pts"]:
                keys.setdefault(k, []).append(i)
        land = {k: v for k, v in keys.items() if len(v) >= MIN_WINDOWS}
        cams = {}
        for i, o in enumerate(self.obs):
            for f in o["cam"]:
                cams.setdefault(f, []).append(i)
        shared = {f: v for f, v in cams.items() if len(v) >= 2}
        hist = []
        for it in range(iters):
            # 1) 랜드마크 전역 위치
            L, spread = {}, []
            for k, ws in land.items():
                P = [self._g(i, self.obs[i]["pts"][k]) for i in ws]
                m, sp = _robust_mean(P)
                L[k] = m
                spread.append(sp)
            C, Crot = {}, {}
            for f, ws in shared.items():
                P = [self._g(i, self.obs[i]["cam"][f][:3, 3]) for i in ws]
                C[f], _ = _robust_mean(P)
                Crot[f] = _rot_average([self.T[i]["R"] @ self.obs[i]["cam"][f][:3, :3] for i in ws])

            # 2) 윈도우 sim3 재적합 (윈도우 0 은 고정 = 게이지)
            for i in range(1, len(self.obs)):
                X, Y, w = [], [], []
                for k in self.obs[i]["pts"]:
                    if k in L:
                        X.append(self.obs[i]["pts"][k]); Y.append(L[k]); w.append(1.0)
                for f in self.obs[i]["cam"]:
                    if f in C:
                        X.append(self.obs[i]["cam"][f][:3, 3]); Y.append(C[f]); w.append(2.0)
                if len(X) < 4:
                    continue
                n_lm = sum(1 for k in self.obs[i]["pts"] if k in L)
                if n_lm >= MIN_LM_FOR_ROT:
                    # 랜드마크가 방 전체에 퍼져 있으므로 회전까지 여기서 푼다
                    s, R, t = _fit_sim3(np.array(X), np.array(Y), w=np.array(w), solve_R=True)
                else:
                    Rs = [Crot[f] @ self.obs[i]["cam"][f][:3, :3].T
                          for f in self.obs[i]["cam"] if f in Crot]
                    R = _rot_average(Rs) if Rs else self.T[i]["R"]
                    s, R, t = _fit_sim3(np.array(X), np.array(Y), R_fixed=R, w=np.array(w))
                self.T[i] = {"s": s, "R": R, "t": t}
            hist.append(float(np.median(spread)) if spread else np.nan)
            if verbose:
                print("   iter %2d  랜드마크 산포 중앙 %.4f m  (랜드마크 %d, 공유프레임 %d)"
                      % (it, hist[-1], len(L), len(shared)), flush=True)
        return self.T, {"landmarks": len(land), "shared_frames": len(shared),
                        "spread_history": hist}
