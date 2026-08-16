#!/usr/bin/env python3
"""DA3 윈도우를 SfM 앵커에 강체(sim3)로 올린다 — 커버리지 100% 포즈.

    $P scripts/anchor_stitch.py --seq <name> --sfm /tmp/poses_inc_e1.txt

관측: DA3 의 **윈도우 내부** 상대기하는 좋다(브릿지 ≤15프레임 무해). 오차는
윈도우를 사슬로 이어붙일 때 쌓인다(v1 스티칭). 그러니 사슬을 버리고, 각 윈도우
안에 있는 SfM 등록 프레임(≥3)으로 윈도우 **전체를 한 번에** sim3 앵커링한다.
앵커 없는 윈도우(관측 사막)만 이웃과의 겹침 프레임으로 국소 연결한다 —
드리프트는 앵커 사이 한두 홉에서만 쌓이고 시퀀스 전체로는 안 쌓인다.
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.depth.pose_stitch import load_windows, robust_umeyama   # noqa: E402
from scripts.incremental_sfm import ate, sim3_apply_pose        # noqa: E402


def fit_sim3(A, B):
    """B ≈ s R A + t. 점이 3개뿐이면 그대로, 많으면 로버스트."""
    if len(A) >= 4:
        s, R, t, _ = robust_umeyama(np.asarray(A), np.asarray(B))
        return s, R, t
    from scripts.incremental_sfm import umeyama3
    return umeyama3(np.asarray(A), np.asarray(B))


def compose(s1, R1, t1, s2, R2, t2):
    """(s1,R1,t1) ∘ (s2,R2,t2)"""
    return s1 * s2, R1 @ R2, s1 * R1 @ t2 + t1


def joint_solve(W, T0, sfm, reg, gtp, args):
    """윈도우별 (logs, rvec, t) 7파라미터 × 76 을 앵커·겹침 잔차로 동시에 푼다."""
    import cv2
    from scipy.optimize import least_squares
    nW = len(W)
    # 초기값: 게이트 통과 윈도우는 그 sim3, 나머지는 이웃에서 전파(느슨한 사슬), 스케일은 중앙값으로 클램프
    med_s = float(np.median([v[0] for v in T0.values()])) if T0 else 1.0
    init = dict(T0)
    q = deque(sorted(init))
    while q:
        k = q.popleft()
        for k2 in (k - 1, k + 1):
            if not (0 <= k2 < nW) or k2 in init:
                continue
            com = sorted(set(W[k]["frames"]) & set(W[k2]["frames"]))
            if len(com) < 3:
                continue
            j1 = {f: j for j, f in enumerate(W[k]["frames"])}
            j2 = {f: j for j, f in enumerate(W[k2]["frames"])}
            A = np.array([W[k2]["c2w"][j2[f]][:3, 3] for f in com])
            B = np.array([W[k]["c2w"][j1[f]][:3, 3] for f in com])
            s_, R_, t_ = fit_sim3(A, B)
            s_ = min(max(s_, 0.5), 2.0)
            init[k2] = compose(*init[k], s_, R_, t_)
            q.append(k2)
    for k in range(nW):
        if k not in init:
            init[k] = (med_s, np.eye(3), np.zeros(3))
    x0 = np.zeros(nW * 7)
    for k in range(nW):
        s_, R_, t_ = init[k]
        x0[k * 7] = np.log(min(max(s_, 0.3), 3.5))
        x0[k * 7 + 1:k * 7 + 4] = cv2.Rodrigues(np.asarray(R_))[0].ravel()
        x0[k * 7 + 4:k * 7 + 7] = t_

    # 잔차 목록 구성
    anch = []                                # (k, local xyz, sfm xyz)
    for k, w in enumerate(W):
        for j, f in enumerate(w["frames"]):
            if f in reg:
                anch.append((k, w["c2w"][j][:3, 3], sfm[f][:3, 3]))
    ovl = []                                 # (k, k2, local_k xyz, local_k2 xyz)
    for k in range(nW - 1):
        com = sorted(set(W[k]["frames"]) & set(W[k + 1]["frames"]))
        j1 = {f: j for j, f in enumerate(W[k]["frames"])}
        j2 = {f: j for j, f in enumerate(W[k + 1]["frames"])}
        for f in com:
            ovl.append((k, k + 1, W[k]["c2w"][j1[f]][:3, 3],
                        W[k + 1]["c2w"][j2[f]][:3, 3]))
    print("조인트: 앵커 잔차 %d · 겹침 잔차 %d" % (len(anch), len(ovl)))

    # 벡터화 — 윈도우별로 점을 묶어 한 번에 변환 (파이썬 루프 잔차는 2분+ 걸렸다)
    aK, aPL, aPG = {}, {}, {}
    for k, pl, pg in anch:
        aK.setdefault(k, []).append((pl, pg))
    for k, v in aK.items():
        aPL[k] = np.array([a for a, _ in v])
        aPG[k] = np.array([b for _, b in v])
    oK = {}
    for k, k2, p1, p2 in ovl:
        oK.setdefault((k, k2), []).append((p1, p2))
    oP1 = {kk: np.array([a for a, _ in v]) for kk, v in oK.items()}
    oP2 = {kk: np.array([b for _, b in v]) for kk, v in oK.items()}

    def unpack(x):
        out = {}
        for k in range(nW):
            out[k] = (np.exp(x[k * 7]), cv2.Rodrigues(x[k * 7 + 1:k * 7 + 4])[0],
                      x[k * 7 + 4:k * 7 + 7])
        return out

    W_SC = 2.0        # 인접 윈도우 로그스케일 차분 벌점 — 없으면 약구속 윈도우가
                      # 폭주한다(실측: 스케일 0.29~9.72). 같은 DA3 런의 스케일은 완만하다.

    def resid(x):
        P = unpack(x)
        parts = []
        for k in aPL:
            s_, R_, t_ = P[k]
            parts.append(((s_ * (aPL[k] @ R_.T) + t_) - aPG[k]).ravel())
        for (k, k2) in oP1:
            s1, R1, t1 = P[k]
            s2, R2, t2 = P[k2]
            parts.append(((s1 * (oP1[(k, k2)] @ R1.T) + t1)
                          - (s2 * (oP2[(k, k2)] @ R2.T) + t2)).ravel())
        parts.append(W_SC * (x[7::7] - x[:-7:7]))
        return np.concatenate(parts)

    from scipy.sparse import lil_matrix
    nres = (sum(len(v) for v in aPL.values())
            + sum(len(v) for v in oP1.values())) * 3 + (nW - 1)
    Sp = lil_matrix((nres, nW * 7), dtype=int)
    r0 = 0
    for k in aPL:
        n3 = len(aPL[k]) * 3
        Sp[r0:r0 + n3, k * 7:k * 7 + 7] = 1
        r0 += n3
    for (k, k2) in oP1:
        n3 = len(oP1[(k, k2)]) * 3
        Sp[r0:r0 + n3, k * 7:k * 7 + 7] = 1
        Sp[r0:r0 + n3, k2 * 7:k2 * 7 + 7] = 1
        r0 += n3
    for k in range(nW - 1):
        Sp[r0 + k, k * 7] = 1
        Sp[r0 + k, (k + 1) * 7] = 1
    res = least_squares(resid, x0, loss="huber", f_scale=0.2, method="trf",
                        jac_sparsity=Sp, max_nfev=200, verbose=0)
    x = res.x
    print("조인트 수렴: 잔차 RMS %.3f m · 스케일 범위 %.2f~%.2f"
          % (float(np.sqrt(np.mean(res.fun ** 2))),
             float(np.exp(x[::7].min())), float(np.exp(x[::7].max()))))

    out = np.tile(np.eye(4), (len(gtp), 1, 1))
    have = np.zeros(len(gtp), bool)
    for k, w in enumerate(W):
        s_ = np.exp(x[k * 7])
        R_ = cv2.Rodrigues(x[k * 7 + 1:k * 7 + 4])[0]
        t_ = x[k * 7 + 4:k * 7 + 7]
        for j, f in enumerate(w["frames"]):
            if w["owner"][j] or not have[f]:
                out[f] = sim3_apply_pose(w["c2w"][j], s_, R_, t_)
                have[f] = True
    for f in reg:
        out[f] = sfm[f]
        have[f] = True
    est = {i: out[i] for i in range(len(gtp)) if have[i]}
    med, p9, sc = ate(est, gtp)
    print("\n커버 %d/%d (%.0f%%)" % (int(have.sum()), len(gtp), 100 * have.mean()))
    print("**전체 ATE 중앙 %.3f m · p90 %.3f · 정합스케일 %.3f**" % (med, p9, sc))
    new = [i for i in est if i not in reg]
    if len(new) > 10:
        m2, p2, _ = ate({i: out[i] for i in new}, gtp)
        print("  (SfM 미등록이었던 %d프레임만: ATE 중앙 %.3f · p90 %.3f)"
              % (len(new), m2, p2))
    if args.out:
        np.savetxt(args.out, out.reshape(len(gtp), 16))
        print("→ %s" % args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--windows", default="poses_raw_np")
    ap.add_argument("--sfm", default="/tmp/poses_inc_e1.txt",
                    help="SfM 포즈(미등록 프레임은 항등). 미터 판을 주면 결과도 미터")
    ap.add_argument("--min-anchor", type=int, default=3)
    ap.add_argument("--joint", action="store_true",
                    help="윈도우 76개의 sim3 를 한 번에 최소자승 — 앵커·겹침 제약 동시."
                         " 앵커/사슬 이분법은 실패했다(게이트를 조이면 사슬 누적, 풀면 폭주)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    gtp = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    sfm = np.loadtxt(args.sfm).reshape(-1, 4, 4)
    reg = set(i for i in range(len(sfm)) if not np.allclose(sfm[i], np.eye(4)))
    W = load_windows(os.path.join(sd, args.windows))
    print("윈도우 %d개 · SfM 앵커 프레임 %d개" % (len(W), len(reg)))

    # ① 앵커 윈도우: 내부 SfM 프레임 ≥ min-anchor → 윈도우 좌표계 → SfM 좌표계 sim3
    # ⚠️ 게이트가 무르면 sim3 스케일이 폭주한다(실측: 기준선 0.05m 게이트에서
    # 앵커 윈도우 스케일 0.90~4.52 — 거의 일직선 3점에 7자유도를 풀었다). 앵커
    # ≥4 · 기준선 ≥0.3m · 잔차 ≤0.15m 를 다 넘어야 앵커 윈도우다. 미달은 사슬로.
    T = {}                                   # w idx → (s, R, t)
    resid = {}
    for k, w in enumerate(W):
        idx = [j for j, f in enumerate(w["frames"]) if f in reg]
        if len(idx) < max(args.min_anchor, 4):
            continue
        A = np.array([w["c2w"][j][:3, 3] for j in idx])
        B = np.array([sfm[w["frames"][j]][:3, 3] for j in idx])
        if np.linalg.norm(A.max(0) - A.min(0)) < 0.3:
            continue
        s, R, t = fit_sim3(A, B)
        r = float(np.median(np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)))
        if r > 0.15 or not (0.3 <= s <= 3.5):
            continue
        T[k], resid[k] = (s, R, t), r
    print("앵커 윈도우 %d/%d · 정합잔차 중앙 %.3f m"
          % (len(T), len(W), np.median(list(resid.values())) if resid else -1))

    if args.joint:
        joint_solve(W, T, sfm, reg, gtp, args)
        return

    # ② 무앵커 윈도우: 겹침 프레임으로 이웃 앵커 윈도우에서 전파 (BFS — 홉 최소)
    n_prop = 0
    q = deque(sorted(T))
    while q:
        k = q.popleft()
        for k2 in (k - 1, k + 1):
            if not (0 <= k2 < len(W)) or k2 in T:
                continue
            com = sorted(set(W[k]["frames"]) & set(W[k2]["frames"]))
            if len(com) < 3:
                continue
            j1 = {f: j for j, f in enumerate(W[k]["frames"])}
            j2 = {f: j for j, f in enumerate(W[k2]["frames"])}
            A = np.array([W[k2]["c2w"][j2[f]][:3, 3] for f in com])
            B = np.array([W[k]["c2w"][j1[f]][:3, 3] for f in com])
            if np.linalg.norm(A.max(0) - A.min(0)) < 0.05:
                continue
            s, R, t = fit_sim3(A, B)          # k2 로컬 → k 로컬
            T[k2] = compose(*T[k], s, R, t)
            n_prop += 1
            q.append(k2)
    print("전파로 배치된 윈도우 %d개 (미배치 %d)" % (n_prop, len(W) - len(T)))

    # ③ 프레임 포즈: owner 윈도우 우선, 없으면 아무 배치 윈도우
    out = np.tile(np.eye(4), (len(gtp), 1, 1))
    have = np.zeros(len(gtp), bool)
    for k in sorted(T):
        w = W[k]
        s, R, t = T[k]
        for j, f in enumerate(w["frames"]):
            if w["owner"][j] or not have[f]:
                out[f] = sim3_apply_pose(w["c2w"][j], s, R, t)
                have[f] = True
    # SfM 등록 프레임은 SfM 포즈를 그대로 쓴다 (가장 정확)
    for f in reg:
        out[f] = sfm[f]
        have[f] = True
    cov = int(have.sum())

    est = {i: out[i] for i in range(len(gtp)) if have[i]}
    med, p9, sc = ate(est, gtp)
    print("\n커버 %d/%d (%.0f%%)" % (cov, len(gtp), 100 * cov / len(gtp)))
    print("**전체 ATE 중앙 %.3f m · p90 %.3f · 정합스케일 %.3f**" % (med, p9, sc))
    new = [i for i in est if i not in reg]
    if len(new) > 10:
        m2, p2, _ = ate({i: out[i] for i in new}, gtp)
        print("  (SfM 미등록이었던 %d프레임만: ATE 중앙 %.3f · p90 %.3f)"
              % (len(new), m2, p2))

    if args.out:
        np.savetxt(args.out, out.reshape(len(gtp), 16))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
