#!/usr/bin/env python3
"""방 분할 기준을 **눈금 없이** 만든다 — 무작위 분할 대조군.

    $P scripts/room_split_null.py

자카드 **절대값은 데이터셋 간 비교가 불가능하다**(2026-08-20 실측):

    Nymeria     어휘 33   · 프레임당 검출 4.2
    SuperMemory 어휘 600  · 프레임당 검출 50

어휘가 600이면 상위 8개가 우연히 겹칠 확률 자체가 낮아 **아무 분할이나 자카드가
낮게 나온다.** 실제로 고정 문턱 0.60 을 대면 SuperMemory 는 k=2~6 전부 "구별됨"
으로 통과했다 — 방 belief 가 최빈방에 못 미치는데도.

→ 기준을 **같은 프레임을 무작위로 나눈 분할**과 비교한다. 관측 자카드가 무작위
분할보다 유의하게 낮아야 "공간이 실제로 갈린다"고 말할 수 있다. 어휘 크기·검출
밀도가 대조군에도 똑같이 들어가므로 **눈금이 상쇄된다.**
"""
import argparse, glob, json, os, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def jac(labels, dets, topn):
    room = {}
    for lb, ws in zip(labels, dets):
        room.setdefault(int(lb), Counter()).update(ws)
    tops = {r: set(w for w, _ in v.most_common(topn)) for r, v in room.items() if v}
    rs = sorted(tops)
    js = [len(tops[rs[i]] & tops[rs[j]]) / max(len(tops[rs[i]] | tops[rs[j]]), 1)
          for i in range(len(rs)) for j in range(i + 1, len(rs))]
    # **최대**를 쓴다 — 구별 안 되는 방이 한 쌍만 있어도 그 분할은 망가진다.
    # 평균을 쓰면 k 가 커질수록 쌍이 늘어 희석돼 z 가 **단조로 좋아진다**(실측:
    # Nymeria 에서 k=6 이 z=-5.33 으로 최고인데 belief 는 기준선 미달).
    return float(np.max(js)) if js else None


def evaluate(name, uv, dets, kmax, topn, nperm, seed=0):
    """uv: 프레임 2D 위치, dets: 프레임별 검출어 목록."""
    from scipy.cluster.vq import kmeans2
    rng = np.random.default_rng(seed)
    print("\n[%s] 프레임 %d · 어휘 %d · 프레임당 %.1f"
          % (name, len(dets), len(set(w for d in dets for w in d)),
             sum(map(len, dets)) / max(len(dets), 1)))
    print("%-4s %-9s %-9s %-8s %s" % ("k", "관측max", "무작위max", "z", "판정"))
    best = None
    for k in range(2, kmax + 1):
        cen, _ = kmeans2(uv, k, minit="++", seed=0, iter=60)
        lab = np.argmin(np.linalg.norm(uv[:, None] - cen[None], axis=2), 1)
        obs = jac(lab, dets, topn)
        if obs is None:
            continue
        # 대조군: 방 크기(프레임 수)는 그대로 두고 배정만 섞는다
        null = []
        for _ in range(nperm):
            p = rng.permutation(lab)
            v = jac(p, dets, topn)
            if v is not None:
                null.append(v)
        mu, sd = float(np.mean(null)), float(np.std(null))
        z = (obs - mu) / sd if sd > 1e-9 else 0.0
        ok = z <= -2.0          # 무작위보다 유의하게 잘 갈림
        print("%-4d %-9.3f %-9.3f %-8.2f %s"
              % (k, obs, mu, z, "**갈림**" if ok else "무작위와 다르지 않음"))
        if ok and (best is None or z < best[1]):
            best = (k, z)
    print("  → %s" % ("선택 k=%d (z=%.2f)" % best if best
                      else "**어떤 k 도 무작위 분할보다 낫지 않다**"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.35)
    args = ap.parse_args()

    out = {}
    # ── Nymeria (벽 있는 집)
    from scripts.nymeria_graph import D as ND, house_frame, traj_of
    det = json.load(open(os.path.join(ND, "owl_det.json")))
    seqs = {}
    for sd in sorted(glob.glob(os.path.join(ND, "loc49", "*"))):
        if os.path.isdir(sd):
            try:
                seqs[os.path.basename(sd)] = traj_of(sd)
            except Exception:
                pass
    _, e1, e2, ctr = house_frame(seqs)
    U, DT = [], []
    for s, (sec, P) in seqs.items():
        rows = det.get(s, [])
        u = np.stack([P @ e1, P @ e2], 1) - ctr
        for r in rows:
            i = int(np.argmin(np.abs(sec - r["sec"])))
            U.append(u[i])
            DT.append([w for w, v in r["det"].items() if v >= args.thr])
    out["nymeria"] = evaluate("Nymeria Loc_49 (벽 있는 집)",
                              np.array(U), DT, args.kmax, args.topn, args.nperm)

    # ── SuperMemory (개방형)
    from scripts.supermem_rooms3d import D as SD, load_traj, gravity, floor_basis
    for sd in ["s1", "s8", "s14"]:
        f = os.path.join(SD, sd, "closed_loop_trajectory.csv")
        dj = os.path.join(SD, "owl_sm_%s.json" % sd)
        if not (os.path.exists(f) and os.path.exists(dj)):
            continue
        d = json.load(open(dj))
        sec, P = load_traj(sd)
        g = gravity(P); e1, e2 = floor_basis(g)
        uv = np.stack([P @ e1, P @ e2], 1)
        fr = np.arange(0, int(sec.max()) + 1)
        fu = np.stack([np.interp(fr, sec, uv[:, 0]), np.interp(fr, sec, uv[:, 1])], 1)
        keys = sorted(d)[:len(fu)]
        DT = [[w for w, v in d[fn].items() if v >= args.thr] for fn in keys]
        out[sd] = evaluate("SuperMemory %s (개방형)" % sd, fu[:len(DT)], DT,
                           args.kmax, args.topn, args.nperm)

    print("\n" + "=" * 60)
    for k, v in out.items():
        print("%-10s %s" % (k, "k=%d (z=%.2f)" % v if v else "갈리지 않음"))


if __name__ == "__main__":
    main()
