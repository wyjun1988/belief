#!/usr/bin/env python3
"""후보 걷기 구성 시뮬 — VLM 없이 캐시+GT 만으로 (floor, depth) 격자 탐색.

    THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ QC_PREFIX=/tmp/t7_q_ \\
      python scripts/rtx7_walksim.py

진단 A(이동후 진짜가 후보에 없음, 35%)를 어떤 걷기 설정이 없애는지 잰다.
각 (점수 하한 분위 floor, 걷기 깊이 depth)에 대해:
  - A율: 후보 목록에 이동후 진짜 크롭이 0장인 타겟 비율 (검색의 상한 손실)
  - 목록 내 이동후 진짜 평균 장수 · 그중 <5m(검증 가능역) 비율
VLM 재채점 전에 이 표로 설정을 고른다 — 재채점은 한 번만.
"""
import glob, json, os
import numpy as np

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
A3P = os.environ.get("A3_PREFIX", "/tmp/t7_a_")
QCP = os.environ.get("QC_PREFIX", "/tmp/t7_q_")
FLOORS = [float(x) for x in os.environ.get("FLOORS", "0.5,0.6,0.7,0.8").split(",")]
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "20,40,80").split(",")]

from collections import Counter
tgt = []   # (TS, ts, 이동후진짜 집합, 그중 <5m 집합)
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    ts, vocab = za["ts"], list(za["vocab"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    live = {m["t"]: m for m in g["live"]}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    moves = {}
    for m in g["moves"]: moves[m["oid"]] = m["t"]
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        if oid not in moves: continue
        t0 = moves[oid]
        TS = QS[:, j] + STx[:, j]
        post = set(); near = set()
        for i in range(len(ts)):
            t = int(ts[i]); m = live.get(t)
            if m and t > t0 and oid in (m.get("vis") or []):
                post.add(i)
                if (m.get("dist") or {}).get(oid, 99) < 5: near.add(i)
        tgt.append((TS, ts, post, near))

print("타겟 %d (이동물체·타입단일)" % len(tgt))
print("%-8s %-6s %-8s %-10s %-12s" % ("floor", "depth", "A율", "진짜/목록", "그중<5m"))
for fl in FLOORS:
    for dp in DEPTHS:
        A = 0; npost = []; nnear = []
        for TS, ts, post, near in tgt:
            th = np.quantile(TS, fl)
            c = sorted(np.where(TS >= th)[0], key=lambda i: -ts[i])[:dp]
            p = sum(1 for i in c if i in post)
            if p == 0: A += 1
            npost.append(p); nnear.append(sum(1 for i in c if i in near))
        print("q%-7.2f %-6d %-8.3f %-10.1f %-12.1f"
              % (fl, dp, A / len(tgt), np.mean(npost), np.mean(nnear)))
