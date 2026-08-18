#!/usr/bin/env python3
"""Nymeria — **벽으로 나뉜 진짜 집**에서 3D 방 분할을 검증한다.

    $P scripts/nymeria_rooms.py --loc Loc_BX

SuperMemory 는 개방형 원룸(논문 명시 'simulated home')이라 3D 군집 순도 0.81~0.93 을
내고도 GT 라벨(사람의 느슨한 기능 구획)과 어긋났다. Nymeria 는 실제 단독주택
(47채·201방·37채 복층)이라 그 교란이 없다.

방 GT 라벨이 없으므로 **세션 간 일치**로 채점한다:

    ① 같은 집(location)의 여러 세션 궤적을 공통 좌표계로 정렬
    ② 각 세션에서 독립적으로 방 군집화
    ③ **다른 날 세션의 군집이 같은 공간 분할로 수렴하는가** = 방 분할의 재현성

같은 집이면 방 경계는 날이 바뀌어도 같아야 한다. 수렴하면 분할이 실재를 잡은
것이고, 흩어지면 궤적 우연에 의존한 것이다. 라벨 없이도 검증 가능한 설계다.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "nymeria")


def load_traj(seq_dir, every=200):
    f = os.path.join(seq_dir, "recording_head", "mps", "slam",
                     "closed_loop_trajectory.csv")
    if not os.path.exists(f):
        return None, None
    t = pd.read_csv(f, usecols=["graph_uid", "tracking_timestamp_us", "tx_world_device",
                                "ty_world_device", "tz_world_device"])
    t = t.iloc[::every]
    P = t[["tx_world_device", "ty_world_device", "tz_world_device"]].values
    return t["graph_uid"].iloc[0], P


def gravity(P, seed=0):
    C = P - P.mean(0)
    idx = np.random.default_rng(seed).choice(len(C), min(20000, len(C)), replace=False)
    _, _, Vt = np.linalg.svd(C[idx], full_matrices=True)
    g = Vt[-1] / np.linalg.norm(Vt[-1])
    return g if g[2] > 0 else -g


def floor_uv(P, g):
    e1 = np.array([1.0, 0, 0])
    e1 = e1 - g * (g @ e1)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    return np.stack([P @ e1, P @ e2], 1)


def occupancy(uv, res=0.25, lim=12.0):
    n = int(2 * lim / res)
    G = np.zeros((n, n), np.float32)
    ij = ((uv + lim) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < n) & (ij[:, 1] < n)
    np.add.at(G, (ij[ok, 0], ij[ok, 1]), 1.0)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="loc49", help="시퀀스 폴더")
    ap.add_argument("--loc", default="Loc_49")
    ap.add_argument("--k", type=int, default=5, help="방 개수")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    metas = {}
    for f in glob.glob(os.path.join(D, "meta", "*.json")):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        metas[os.path.basename(f)[:-5]] = m

    seqs = []
    for sd in sorted(glob.glob(os.path.join(D, args.dir, "*"))):
        if not os.path.isdir(sd):
            continue
        n = os.path.basename(sd)
        m = metas.get(n, {})
        if args.loc and m.get("location") != args.loc:
            continue
        uid, P = load_traj(sd)
        if P is None or len(P) < 50:
            continue
        seqs.append(dict(name=n, date=m.get("date"), who=m.get("fake_name"),
                         uid=uid, P=P))
    if not seqs:
        sys.exit("궤적을 찾지 못했다 — 압축 해제를 확인하라")
    print("%s · 세션 %d개 · 날짜 %s"
          % (args.loc, len(seqs), sorted({s["date"] for s in seqs})))
    uids = Counter(s["uid"] for s in seqs)
    print("SLAM graph_uid %d종 %s"
          % (len(uids), "→ **공통 좌표계**" if len(uids) == 1 else "→ 세션별 좌표계(정렬 필요)"))

    # 세션별 3D 방 군집 + 점유격자
    from scipy.cluster.vq import kmeans2
    Pall = np.concatenate([s["P"] for s in seqs])
    g0 = gravity(Pall)
    ctr = floor_uv(Pall, g0).mean(0)      # 공통 원점(전 세션 평균) 하나만 뺀다
    grids, cents = [], []
    for s in seqs:
        uv = floor_uv(s["P"], g0) - ctr
        # ⚠️ 세션별 중심화 금지 — 공통 좌표계를 파괴해 IoU 가 0 이 된다(실측 사고).
        cen, lab = kmeans2(uv, args.k, minit="++", seed=0, iter=50)
        s["uv"], s["cen"], s["lab"] = uv, cen, lab
        grids.append(occupancy(uv))
        span = np.linalg.norm(uv.max(0) - uv.min(0))
        print("  %-44s %s · 이동범위 %.1f m · 체류 %s"
              % (s["name"][:44], s["date"], span,
                 dict(Counter(int(x) for x in lab))))

    # 세션 간 공간 일치: 점유격자 IoU (공통 좌표계면 정렬 불필요)
    if len(grids) >= 2:
        ious = []
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                a, b = grids[i] > 1, grids[j] > 1
                ious.append((a & b).sum() / max((a | b).sum(), 1))
        print("\n세션 간 점유격자 IoU: 중앙 %.3f · 최대 %.3f (n=%d쌍)"
              % (np.median(ious), max(ious), len(ious)))
        print("  같은 집이면 높아야 한다 — 낮으면 좌표계가 세션별로 달라 정렬이 먼저다")
    if args.out:
        json.dump({s["name"]: dict(date=s["date"], uid=s["uid"],
                                   centers=s["cen"].tolist()) for s in seqs},
                  open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
