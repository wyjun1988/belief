#!/usr/bin/env python3
"""SuperMemory(개방형 공간)에 **같은 자카드 기준**을 적용한다.

    $P scripts/room_split_k_sm.py --sess s1 s8 s14

`room_split_k.py` 가 Nymeria(벽 있는 집)에서 GT 없이 k=2 를 골라냈고 그 k 가
실제 belief 최고(+43%)였다. 같은 기준을 개방형 공간에 대면 **"방 belief 가 접힌
것이 공간 탓인지 층 탓인지"** 가 판정된다.

- 갈리는 경계가 나온다  → 공간은 갈리는데 belief 층이 못 쓴 것 = **층 탓**
- 어떤 k 도 기준 미달   → 물체 분포로 경계가 없다 = **공간 탓** (기존 가설 확증)
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.supermem_rooms3d import D, load_traj, gravity, floor_basis, cluster_rooms  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", nargs="+", default=["s1", "s8", "s14", "s19", "s20"])
    ap.add_argument("--thr", type=float, default=0.35, help="OWLv2 검출 문턱")
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--max-jaccard", type=float, default=0.60)
    ap.add_argument("--kmax", type=int, default=6)
    args = ap.parse_args()

    print("%-5s %-4s %-10s %-10s %s" % ("세션", "k", "평균자카드", "최대자카드", "판정"))
    picks = {}
    for sd in args.sess:
        f = os.path.join(D, sd, "closed_loop_trajectory.csv")
        dj = os.path.join(D, "owl_sm_%s.json" % sd)
        if not (os.path.exists(f) and os.path.exists(dj)):
            print("%-5s 자료 없음 — 건너뜀" % sd)
            continue
        det = json.load(open(dj))
        sec, P = load_traj(sd)
        g = gravity(P); e1, e2 = floor_basis(g)
        uv = np.stack([P @ e1, P @ e2], 1)
        fr_sec = np.arange(0, int(sec.max()) + 1)
        fu = np.stack([np.interp(fr_sec, sec, uv[:, 0]),
                       np.interp(fr_sec, sec, uv[:, 1])], 1)
        keys = sorted(det)
        best = None
        for k in range(2, args.kmax + 1):
            cen, _ = cluster_rooms(uv, k)
            lab = np.argmin(np.linalg.norm(fu[:, None] - cen[None], axis=2), 1)
            room = {}
            for i, fn in enumerate(keys):
                if i >= len(lab):
                    break
                c = room.setdefault(int(lab[i]), Counter())
                for w, s in det[fn].items():
                    if s >= args.thr:
                        c[w] += 1
            tops = {r: set(w for w, _ in v.most_common(args.topn))
                    for r, v in room.items() if v}
            rs = sorted(tops)
            js = [len(tops[rs[i]] & tops[rs[j]]) / max(len(tops[rs[i]] | tops[rs[j]]), 1)
                  for i in range(len(rs)) for j in range(i + 1, len(rs))]
            if not js:
                continue
            mj, xj = float(np.mean(js)), float(np.max(js))
            ok = xj <= args.max_jaccard
            print("%-5s %-4d %-10.3f %-10.3f %s"
                  % (sd, k, mj, xj, "구별됨" if ok else "**과분할**"))
            if ok and (best is None or k > best[0]):
                best = (k, mj, xj)
        picks[sd] = best
        print("  → %s\n" % ("선택 k=%d (평균 %.3f)" % best[:2] if best
                            else "**어떤 k 도 기준 미달 — 물체 분포로 갈리지 않는다**"))

    n_ok = sum(1 for v in picks.values() if v)
    print("=" * 58)
    print("세션 %d개 중 **%d개**만 물체 분포로 갈린다" % (len(picks), n_ok))
    if n_ok == 0:
        print("→ 개방형 공간에서 방 belief 가 접힌 것은 **공간 탓**이 맞다(층 탓 아님)")
    else:
        print("→ 갈리는 세션이 있다 — 공간 탓 가설을 다시 봐야 한다: %s"
              % [s for s, v in picks.items() if v])


if __name__ == "__main__":
    main()
