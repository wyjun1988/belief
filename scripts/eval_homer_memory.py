#!/usr/bin/env python3
"""HOMER+ 75일 스케일에서 **v2 메모리 설계**를 검증한다.

    $P scripts/eval_homer_memory.py --household HouseholdA --patrol 240

ADT 로는 못 재는 것을 잰다. ADT 는 90초이고 착용자가 직접 물건을 옮겨 **미관측 이동이
없다** — 그래서 v1 에서 belief 모델이 last-known 과 소수점까지 같았다. 여기서는
행위자(사용자)와 관측자(순찰하는 보조 장치)를 분리해 미관측 이동을 실제로 만든다.

관측 모델: 관측자가 `--patrol` 분마다 **방 하나씩 돌아가며** 훑는다. 그 순간 그 방에
있는 물체는 전부 보이고, **없는 것은 '거기 없다'는 증거**가 된다(사용자가 말한
"없어진 것은 업데이트가 안 된다"의 반대편 — 부재는 확실히 알 수 있다).

비교하는 믿음 규칙 넷:
    last-known   마지막으로 본 방. v1 에서 belief 모델이 정확히 이것과 같았다.
    +부재증거     믿던 방을 훑었는데 없었다면 그 믿음을 버린다
    루틴 사전     시간대별 그 물체의 과거 분포 (학습일에서 배운다) — v2 의 '밤 업데이트'
    혼합         관측이 신선하면 last-known, 오래됐으면 루틴 사전
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.homer.parse import load_household, timeline      # noqa: E402

HOMER = os.path.expanduser("~/work/home-jepa/data/homer_plus")
BUCKET = 180          # 분. 시간대 버킷 (3시간) — 루틴 사전의 해상도


def room_at(tl, t_idx, obj):
    return tl[t_idx][1].get(obj, (None, None))[1]


def patrol_schedule(tl, rooms, every_min):
    """[(스텝 인덱스, 훑는 방)] — every_min 마다 방을 하나씩 순환."""
    out, k, nxt = [], 0, tl[0][0]
    for i, (t, _, _) in enumerate(tl):
        if t >= nxt:
            out.append((i, rooms[k % len(rooms)]))
            k += 1
            nxt = t + every_min
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--household", default="HouseholdA")
    ap.add_argument("--root", default=HOMER)
    ap.add_argument("--train-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=10)
    ap.add_argument("--patrol", type=int, nargs="+", default=[60, 120, 240, 480, 720])
    ap.add_argument("--fresh-min", type=int, default=240, help="혼합 규칙의 신선도 기준(분)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hh = os.path.join(args.root, args.household)
    tr = timeline(load_household(hh, "train", args.train_days))
    te = timeline(load_household(hh, "test", args.test_days))
    names = load_household(hh, "test", 1)[0]["names"]
    rooms = sorted({r for _, loc, _ in te for _, (_, r) in loc.items() if r})
    objs = sorted({o for _, loc, _ in te for o in loc})
    print("%s · 학습 %d일 %d스텝 · 시험 %d일 %d스텝 · 방 %d · 물체 %d"
          % (args.household, args.train_days, len(tr), args.test_days, len(te),
             len(rooms), len(objs)))

    # --- 루틴 사전: 학습일에서 (물체, 시간대) → 방 분포 --------------------------
    prior = defaultdict(Counter)
    for t, loc, _ in tr:
        b = int((t % 1440) // BUCKET)
        for o, (_, r) in loc.items():
            if r:
                prior[(o, b)][r] += 1
    glob = defaultdict(Counter)
    for (o, _), c in prior.items():
        glob[o].update(c)

    def routine(o, t):
        b = int((t % 1440) // BUCKET)
        c = prior.get((o, b)) or glob.get(o)
        return c.most_common(1)[0][0] if c else None

    rows = []
    for P in args.patrol:
        sched = patrol_schedule(te, rooms, P)
        # 관측 이력: obj → [(시각, 방)] 그리고 부재 이력: (obj, 방) → 마지막 부재 시각
        last_seen, absent_at = {}, {}
        si, res = 0, defaultdict(lambda: [0, 0])
        gaps = []
        for i, (t, loc, _) in enumerate(te):
            while si < len(sched) and sched[si][0] <= i:
                _, R = sched[si]
                here = {o for o, (_, r) in loc.items() if r == R}
                for o in objs:
                    if o in here:
                        last_seen[o] = (t, R)
                    elif last_seen.get(o, (None, None))[1] == R:
                        absent_at[o] = t                 # 믿던 방을 훑었는데 없었다
                si += 1
            if i % 7:                                    # 질의는 성기게
                continue
            for o in objs:
                gt = loc.get(o, (None, None))[1]
                if gt is None or o not in last_seen:
                    continue
                ts, lr = last_seen[o]
                stale = absent_at.get(o, -1) > ts
                gaps.append(t - ts)
                ans = {
                    "last-known": lr,
                    "+부재증거": (routine(o, t) if stale else lr),
                    "루틴사전": routine(o, t),
                    "혼합": (lr if (t - ts) <= args.fresh_min and not stale else routine(o, t)),
                }
                # ⚠️ **층화가 없으면 이 지표는 무의미하다.** 물체 대부분이 한 방에
                # 머무르므로 아무 규칙이나 0.98 을 받는다(v1 에서 똑같이 겪었다).
                # 판별력이 있는 것은 **마지막 관측 이후 방이 바뀐 질의**뿐이다.
                moved = (gt != lr)
                for k, v in ans.items():
                    res[k][1] += 1
                    res[k][0] += (v == gt)
                    if moved:
                        res[k + "|이동"][1] += 1
                        res[k + "|이동"][0] += (v == gt)
        n, nm = res["last-known"][1], res["last-known|이동"][1]
        row = dict(patrol=P, n=n, n_moved=nm, mean_gap=float(np.mean(gaps)) if gaps else 0,
                   **{k: v[0] / max(v[1], 1) for k, v in res.items()})
        rows.append(row)
        print("  순찰 %4d분 · 공백 %5.0f분 · 질의 %6d (**이동 %4d = %.1f%%**)"
              % (P, row["mean_gap"], n, nm, 100 * nm / max(n, 1)))
        print("      전체   last-known %.3f · +부재 %.3f · 루틴 %.3f · 혼합 %.3f"
              % (row["last-known"], row["+부재증거"], row["루틴사전"], row["혼합"]))
        print("      **이동** last-known %.3f · +부재 %.3f · 루틴 %.3f · 혼합 %.3f"
              % (row["last-known|이동"], row["+부재증거|이동"],
                 row["루틴사전|이동"], row["혼합|이동"]))

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
