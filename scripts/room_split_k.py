#!/usr/bin/env python3
"""방 분할의 k 를 **데이터가 정하게** 한다.

    $P scripts/room_split_k.py

실측(⑫): Nymeria 에서 k=2 가 최빈방 대비 **+43%**, k=3 +20%, k≥4 는 **기준선 미달**.
방 구성을 보면 이 집은 실제로 두 공간(벽으로 막힌 부엌 + 하나의 거실)이고,
k 를 올리면 거실을 임의로 쪼개 **서로 구별되지 않는 방**이 생긴다.

→ 방 분할의 목표는 "정확한 방 개수 맞히기" 가 아니라 **"물체 분포가 갈리는 경계
찾기"** 다. 여기서는 **방 간 물체 분포의 구별 가능성**으로 k 를 고른다:

    구별도(k) = 방 쌍의 상위 물체 집합 자카드 평균  (낮을수록 잘 갈림)

자카드가 문턱을 넘는 쌍이 생기면 그 k 는 과분할이다. belief GT 를 쓰지 않으므로
**실사용에서 그대로 쓸 수 있는 기준**이다.
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.nymeria_graph import D, house_frame, traj_of        # noqa: E402
from scripts.nymeria_belief import assign                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="owl_det.json")
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--topn", type=int, default=8, help="방별 상위 물체 수")
    ap.add_argument("--max-jaccard", type=float, default=0.60,
                    help="방 쌍 자카드가 이 값을 넘으면 과분할로 본다")
    ap.add_argument("--kmax", type=int, default=8)
    args = ap.parse_args()

    from scipy.cluster.vq import kmeans2
    det = json.load(open(os.path.join(D, args.det)))
    seqs = {}
    for sd in sorted(glob.glob(os.path.join(D, "loc49", "*"))):
        if os.path.isdir(sd):
            try:
                seqs[os.path.basename(sd)] = traj_of(sd)
            except Exception:
                pass
    _, e1, e2, ctr = house_frame(seqs)
    P = np.concatenate([p for _, p in seqs.values()])
    U = np.stack([P @ e1, P @ e2], 1) - ctr

    print("%-4s %-10s %-10s %s" % ("k", "평균자카드", "최대자카드", "판정"))
    best = None
    for k in range(2, args.kmax + 1):
        cen, _ = kmeans2(U, k, minit="++", seed=0, iter=60)
        obs, _ = assign(det, seqs, cen, e1, e2, ctr, args.score_thr)
        room = defaultdict(Counter)
        for (s, c), cnt in obs.items():
            for r, n in cnt.items():
                room[r][c] += n
        tops = {r: set(w for w, _ in v.most_common(args.topn)) for r, v in room.items()}
        js = []
        rs = sorted(tops)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = tops[rs[i]], tops[rs[j]]
                js.append(len(a & b) / max(len(a | b), 1))
        if not js:
            continue
        mean_j, max_j = float(np.mean(js)), float(np.max(js))
        ok = max_j <= args.max_jaccard
        print("%-4d %-10.3f %-10.3f %s" % (k, mean_j, max_j,
              "구별됨" if ok else "**과분할** (쌍 자카드 %.2f)" % max_j))
        if ok and (best is None or k > best[0]):
            best = (k, mean_j, max_j)
    if best:
        print("\n→ **선택 k = %d** (평균 자카드 %.3f · 최대 %.3f)" % best)
        print("   기준: 모든 방 쌍이 상위 %d개 물체에서 자카드 ≤ %.2f"
              % (args.topn, args.max_jaccard))
    else:
        print("\n→ 어떤 k 도 기준을 못 넘는다 — 이 공간은 물체 분포로 갈리지 않는다")


if __name__ == "__main__":
    main()
