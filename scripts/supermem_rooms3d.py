#!/usr/bin/env python3
"""SuperMemory 방 분할 — CLIP 한 장 분류(0.52) 대신 **MPS 3D 기하**로.

    $P scripts/supermem_rooms3d.py --sess s8 s14 s1 s19 s20

실측(2026-08-18): 프레임을 CLIP 으로 방 분류하면 0.52 다(침실의 57%가 복도로 오분류).
그 위에 belief 를 쌓으니 last-known 0.43 · 모델 0.42 로 최빈(0.69)에 못 미쳤다.
ADT 에서 belief 가 가구 top-3 1.00 을 낸 것은 방 분할이 **3D 구역 지도**였기 때문.

여기서는 SuperMemory 의 MPS 를 써서 같은 조건을 만든다:

    ① 궤적·점구름을 중력 정렬 → 바닥면 점유격자(우리 v1 regions 와 같은 표현)
    ② 카메라 위치를 격자에 투영 → **공간 군집**으로 방 후보 분할
    ③ 각 방의 대표 프레임 물체 조합 → Qwen 이 방 이름 결정(W3 에서 검증된 경로)
    ④ 프레임 → 방: 그 시각의 카메라 위치가 속한 군집 (한 장 분류가 아니라 위치)

핵심 차이: 프레임의 방을 **그림이 아니라 위치**로 정한다. 같은 부엌을 어느 각도에서
찍든 위치는 부엌이다 — CLIP 오분류의 원인이 사라진다.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")


def load_traj(sd, every=50):
    t = pd.read_csv(os.path.join(D, sd, "closed_loop_trajectory.csv"),
                    usecols=["tracking_timestamp_us", "tx_world_device",
                             "ty_world_device", "tz_world_device"])
    t = t.iloc[::every]
    sec = (t["tracking_timestamp_us"].values - t["tracking_timestamp_us"].values[0]) / 1e6
    P = t[["tx_world_device", "ty_world_device", "tz_world_device"]].values
    return sec, P


def gravity(P, seed=0):
    C = P - P.mean(0)
    idx = np.random.default_rng(seed).choice(len(C), min(20000, len(C)), replace=False)
    _, _, Vt = np.linalg.svd(C[idx], full_matrices=True)
    g = Vt[-1] / np.linalg.norm(Vt[-1])
    return g if g[2] > 0 else -g


def floor_basis(g):
    e1 = np.array([1.0, 0, 0])
    e1 = e1 - g * (g @ e1)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(g, e1)


def cluster_rooms(uv, k, seed=0):
    """카메라 체류 위치를 k 개 방으로 — KMeans(체류 시간 가중)."""
    from scipy.cluster.vq import kmeans2
    cen, lab = kmeans2(uv, k, minit="++", seed=seed, iter=50)
    return cen, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", nargs="+", default=["s8", "s14"])
    ap.add_argument("--k", type=int, default=4, help="방 개수")
    ap.add_argument("--out", default=os.path.join(D, "rooms3d.json"))
    args = ap.parse_args()

    out = {}
    for sd in args.sess:
        f = os.path.join(D, sd, "closed_loop_trajectory.csv")
        if not os.path.exists(f):
            print("%s: MPS 궤적 없음 — 건너뜀" % sd)
            continue
        try:
            sec, P = load_traj(sd)
        except Exception as e:
            # 깨진/빈 CSV 는 건너뛴다 — 예전엔 여기서 죽어 **저장 전에 중단**됐고,
            # 결과적으로 3D 방이 일부 세션에만 적용된 채 실험이 무효가 됐다.
            print("%s: 궤적 읽기 실패(%s) — 건너뜀" % (sd, str(e)[:60]))
            continue
        g = gravity(P)
        e1, e2 = floor_basis(g)
        uv = np.stack([P @ e1, P @ e2], 1)
        cen, lab = cluster_rooms(uv, args.k)
        spread = np.linalg.norm(uv.max(0) - uv.min(0))
        # 프레임(1fps) → 방: 그 시각의 카메라 위치가 속한 군집
        fr_sec = np.arange(0, int(sec.max()) + 1)
        fu = np.stack([np.interp(fr_sec, sec, uv[:, 0]),
                       np.interp(fr_sec, sec, uv[:, 1])], 1)
        d = np.linalg.norm(fu[:, None] - cen[None], axis=2)
        fr_room = np.argmin(d, 1)
        out[sd] = dict(centers=cen.tolist(), frame_room=fr_room.tolist(),
                       spread_m=float(spread),
                       dwell=dict(Counter(int(x) for x in fr_room)))
        print("%-4s 궤적 %.1f분 · 이동범위 %.1f m · 방 %d개 체류 %s"
              % (sd, sec.max() / 60, spread, args.k,
                 dict(Counter(int(x) for x in fr_room))))
    json.dump(out, open(args.out, "w"))
    print("→ %s" % args.out)


if __name__ == "__main__":
    main()
