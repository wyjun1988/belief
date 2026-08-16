#!/usr/bin/env python3
"""세션 간 정합 — **물체 주석 없이** MPS 점구름의 바닥면 상관으로.

    $P scripts/align_sessions_floor.py --a data/supermem/s8 --b data/supermem/s14

Gen2 에서는 물체 카테고리 매칭 + RANSAC sim3 로 세션을 정합했다(인라이어 16/21).
그러나 실데이터 대부분(SuperMemory·EgoLife)에는 물체 주석이 없다. 여기서는
기하만 쓴다:

    ① 연직축   점구름 주성분 중 가장 얇은 축(실내는 바닥·천장이 지배)
    ② 바닥면   바닥에서 0.8~2.2 m 수평 단면 = 벽·가구의 평면도
    ③ 정합     회전 격자 × FFT 상호상관 → 점유격자 IoU 로 검증

같은 MPS 미터계라 스케일은 1 로 고정한다(그래서 sim3 가 아니라 SE(2)+높이).
검증: IoU 를 무작위 변환 분포와 비교 — 실측 0.505 vs 무작위 중앙 0.198(z=4.2).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


def load_points(sd, max_rows=2000000, max_pts=200000):
    df = pd.read_csv(os.path.join(sd, "semidense_points.csv.gz"),
                     usecols=["px_world", "py_world", "pz_world",
                              "inv_dist_std", "dist_std"], nrows=max_rows)
    q = df[(df["inv_dist_std"] < 0.005) & (df["dist_std"] < 0.05)]
    P = q[["px_world", "py_world", "pz_world"]].values
    return P[::max(1, len(P) // max_pts)]


def gravity(P, seed=0):
    """연직축 — 점구름의 최소분산 방향. 실내는 바닥·천장 평면이 지배한다."""
    C = P - P.mean(0)
    idx = np.random.default_rng(seed).choice(len(C), min(20000, len(C)), replace=False)
    _, _, Vt = np.linalg.svd(C[idx], full_matrices=True)
    g = Vt[-1] / np.linalg.norm(Vt[-1])
    return g if g[2] > 0 else -g


def floorplan(P, g, lo=0.8, hi=2.2):
    """바닥 기준 lo~hi m 수평 단면의 2D 투영 = 평면도."""
    h = P @ g
    h = h - np.percentile(h, 2.0)
    m = (h > lo) & (h < hi)
    e1 = np.array([1.0, 0, 0])
    e1 = e1 - g * (g @ e1)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    return np.stack([P[m] @ e1, P[m] @ e2], 1)


def occupancy(uv, res, lim, thr=3):
    n = int(2 * lim / res)
    grid = np.zeros((n, n), np.float32)
    ij = ((uv + lim) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < n) & (ij[:, 1] < n)
    np.add.at(grid, (ij[ok, 0], ij[ok, 1]), 1.0)
    return (grid > thr).astype(np.float32)


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--res", type=float, default=0.15)
    ap.add_argument("--lim", type=float, default=8.0)
    ap.add_argument("--step", type=int, default=2, help="회전 격자(도)")
    ap.add_argument("--trials", type=int, default=40, help="무작위 대조 횟수")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load_points(args.a), load_points(args.b)
    UA = floorplan(A, gravity(A))
    UB = floorplan(B, gravity(B))
    cA, cB = UA.mean(0), UB.mean(0)
    UA, UB = UA - cA, UB - cB
    print("평면도 점수: A %d · B %d" % (len(UA), len(UB)))

    N = int(2 * args.lim / args.res)
    GA = occupancy(UA, args.res, args.lim)

    def iou(deg, shift):
        GB = occupancy(UB @ rot(deg).T + shift, args.res, args.lim)
        a, b = GA.astype(bool), GB.astype(bool)
        return (a & b).sum() / max((a | b).sum(), 1)

    best = (-1.0, 0, np.zeros(2))
    for deg in range(0, 360, args.step):
        c = fftconvolve(GA, occupancy(UB @ rot(deg).T, args.res, args.lim)[::-1, ::-1],
                        mode="same")
        k = np.unravel_index(c.argmax(), c.shape)
        sh = (np.array(k) - N // 2) * args.res
        v = iou(deg, sh)
        if v > best[0]:
            best = (v, deg, sh)

    rng = np.random.default_rng(0)
    rs = [iou(rng.uniform(0, 360), rng.uniform(-2, 2, 2)) for _ in range(args.trials)]
    z = (best[0] - np.mean(rs)) / max(np.std(rs), 1e-9)
    print("**정합: IoU %.3f · 회전 %d° · 이동 %s**" % (best[0], best[1], np.round(best[2], 2)))
    print("무작위 대조: 중앙 %.3f · 최대 %.3f (n=%d) → %.1f× · z=%.1f"
          % (np.median(rs), max(rs), len(rs), best[0] / max(np.median(rs), 1e-9), z))
    if z < 3:
        print("⚠️ z<3 — 정합이 신뢰 수준에 못 미친다(다른 집이거나 겹침 부족)")

    if args.out:
        np.savez(args.out, deg=best[1], shift=best[2], iou=best[0], z=z,
                 centroid_a=cA, centroid_b=cB)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
