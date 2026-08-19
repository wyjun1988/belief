#!/usr/bin/env python3
"""방 belief 결정규칙 — 사전분포 가중이 최빈방을 넘는가.

    $P scripts/room_prior_rule.py --dump <supermem_belief --dump 산출>

진단(2026-08-19): 방 belief 모델은 퇴화하지 않았다. 최빈방이 절대 못 맞히는
비부엌 문항을 21~25% 잡아낸다. 다만 그걸 얻으려고 부엌 문항을 너무 많이 내준다 —
**이탈을 과잉 예측**한다. 능력 문제가 아니라 보정 문제라는 뜻이고, 그렇다면
사전분포로 눌러주면 최빈방을 넘을 수 있어야 한다. 이 스크립트가 그걸 판정한다.

    score(r) = p_model(r) + α · prior(r)

⚠️ 두 가지를 지킨다.
① **사전분포는 관측에서만 뽑는다.** GT 방 분포를 쓰면 최빈방 기준선을 그대로
   베끼는 것이라 "넘었다"가 무의미해진다. 여기서는 문항별 last-known 방의 분포를
   쓴다 — 전부 지각 산출물이다.
② **α 는 leave-one-out 으로 고른다.** 전체에 맞춰 고른 α 로 전체를 채점하면
   그건 상한이지 성능이 아니다. 둘 다 찍어서 격차를 보인다.
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]


def load(path):
    rows = [r for r in json.load(open(path)) if r.get("probs") and r.get("lastknown")]
    return rows


def prior_of(rows):
    """관측 사전분포 — last-known 방의 분포(정규화). GT 를 쓰지 않는다."""
    c = Counter(r["lastknown"] for r in rows)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def predict(r, prior, a):
    sc = {k: r["probs"].get(k, 0.0) + a * prior.get(k, 0.0) for k in set(r["probs"]) | set(prior)}
    return max(sc, key=sc.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load(args.dump)
    n = len(rows)
    gtc = Counter(r["gt"] for r in rows)
    maj_room, maj_n = gtc.most_common(1)[0]
    maj = maj_n / n
    print("[%s] %d문항 · GT 최빈방 %s %.3f" % (args.label or args.dump, n, maj_room, maj))

    prior = prior_of(rows)
    print("   관측 사전분포(last-known): %s"
          % ", ".join("%s %.2f" % (k, v) for k, v in sorted(prior.items(), key=lambda x: -x[1])))
    pm = max(prior, key=prior.get)
    print("   관측 최빈방 = %s (GT 최빈방과 %s)" % (pm, "일치" if pm == maj_room else "**불일치**"))

    # α 전수 (상한)
    accs = {}
    for a in ALPHAS:
        accs[a] = np.mean([predict(r, prior, a) == r["gt"] for r in rows])
    best_a = max(accs, key=accs.get)
    print("\n%-10s %s" % ("α", "정확도"))
    for a in ALPHAS:
        mark = " ←최고" if a == best_a else ""
        print("  %-8.2f %.3f%s" % (a, accs[a], mark))

    # leave-one-out — 각 문항의 α 를 나머지 n-1 에서 고른다
    ok = 0
    for i, r in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        pr2 = prior_of(rest)
        sc = {a: np.mean([predict(x, pr2, a) == x["gt"] for x in rest]) for a in ALPHAS}
        a_i = max(sc, key=sc.get)
        ok += predict(r, pr2, a_i) == r["gt"]
    loo = ok / n

    print("\n%-28s %.3f" % ("최빈방(GT 기준선)", maj))
    print("%-28s %.3f" % ("모델 단독 (α=0)", accs[0.0]))
    print("%-28s %.3f  (α=%.2f — 전체에 맞춘 상한)" % ("사전분포 가중 · 상한", accs[best_a], best_a))
    print("%-28s **%.3f**  %s" % ("사전분포 가중 · LOO", loo,
                                  "**최빈방 초과**" if loo > maj else "최빈방 미달"))


if __name__ == "__main__":
    main()
