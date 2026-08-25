#!/usr/bin/env python3
"""**시간 오라클을 걷어낸다.** 지금까지 `ok = ts > t0` 로 마지막 이동 이후 프레임만
후보로 삼았는데, `t0`(이동 시각)는 GT 다. 실제 시스템은 모른다.

⚠️ 왜 중요한가 — 사용자 예시: 물체가 거실 → 방 → 거실로 갔다. 이동 시각을 알면
마지막 구간만 보면 되지만, 모르면 **전체 기록에서 "어느 관측이 최신인가" 를 스스로
판단해야 한다.** 마지막 구간은 프레임이 적게 쌓이므로(방금 돌아왔으니) 여기서
재현율이 결정적이 된다. 재현율 0.143 이면 후반 3장을 놓치고 "방" 이라 답한다.

비교하는 방식:
  oracle   ts > t0 만 (지금까지 재던 것)
  all      전체 프레임에서 점수 상위 K — 시간 무시
  recent   점수 상위 K 중 **가장 늦은** 프레임의 방
  decay    점수 × 시간가중(최근일수록 크게)
"""
import json, glob, os, sys, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
MODE = sys.argv[1] if len(sys.argv) > 1 else "합"
K = int(os.environ.get("TOPK", "10"))
HL = float(os.environ.get("HALFLIFE", "900"))     # 시간가중 반감기(초)


def ci(a):
    a = np.asarray(a, float); n = len(a)
    if n == 0: return 0., 0., 0.
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


res = {}
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fq = "/tmp/qc_%s.npz" % hn
    if not os.path.exists(fq): continue
    z = np.load(fq, allow_pickle=True)
    st, si, ts, tg = z["st"], z["si"], z["ts"], list(z["tg"])
    g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    rooms = np.array([live[t]["room"] for t in ts])
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(tg):
        if cnt[g["gt0"][oid]["type"]] > 1: continue
        mv = [x for x in moves if x["oid"] == oid]
        t0 = mv[-1]["t"] if mv else 0
        tgt = mv[-1]["to"] if mv else g["gt0"][oid]["room"]
        S = {"글자": st[:, j], "이미지": si[:, j], "합": st[:, j] + si[:, j]}[MODE]
        vis = np.array([oid in live[t].get("vis", []) for t in ts])
        if vis.sum() < 3: continue
        T = ts.astype(float)

        def vote(idx):
            if not len(idx): return None
            r = list(rooms[idx]); return max(set(r), key=r.count)

        # ① 시간 오라클 — 지금까지 재던 것
        okd = np.where(T > t0)[0]
        res.setdefault("① 이동시각 GT (오라클)", []).append(
            vote(okd[np.argsort(-S[okd])[:K]]) == tgt if len(okd) else False)
        # ② 전체에서 점수 상위 K (시간 무시)
        top = np.argsort(-S)[:K]
        res.setdefault("② 전체 상위K (시간 무시)", []).append(vote(top) == tgt)
        # ③ 상위K 중 가장 늦은 프레임
        res.setdefault("③ 상위K 중 최신 1장", []).append(rooms[top[np.argmax(T[top])]] == tgt)
        # ④ 점수 × 시간가중
        w = S * np.exp(-(T.max() - T) / HL)
        res.setdefault("④ 점수×시간가중 상위K", []).append(vote(np.argsort(-w)[:K]) == tgt)
        # ⑤ 상위K 를 넉넉히(=재현율↑) 잡고 최신
        big = np.argsort(-S)[:K * 5]
        res.setdefault("⑤ 상위5K 중 최신 1장", []).append(rooms[big[np.argmax(T[big])]] == tgt)

print("=== 시간 오라클 제거 [%s 질의] · 상위 %d · 반감기 %.0fs ===" % (MODE, K, HL))
for k in ("① 이동시각 GT (오라클)", "② 전체 상위K (시간 무시)", "③ 상위K 중 최신 1장",
          "④ 점수×시간가중 상위K", "⑤ 상위5K 중 최신 1장"):
    if k not in res: continue
    m, lo, hi = ci(res[k])
    print("  %-24s %.3f [%.3f %.3f]  n=%d" % (k, m, lo, hi, len(res[k])))
