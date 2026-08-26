#!/usr/bin/env python3
"""온라인 등록 — 매핑워크가 놓친 소형물의 기록을 **첫 확신 군집**으로 초기화.

    THOR_ROOT=data/thor5 A3_PREFIX=... QC_PREFIX=... python scripts/eval_enroll.py

검출 초기맵의 타겟 sg 가 0.2~0.5(§chain9)라 통합이 0.506 으로 주저앉았다.
대안: 배회에서 점수 문턱(q0.98)+시간군집(2장+)을 **가장 이른 것**부터 찾아
그 군집의 방(앵커 결합 국소화)을 초기 기록으로 삼는다.
채점: 물체의 **t=0 GT 방** 대비. (이동 전 등록이 목표 — 이동 후 첫 군집이면
새 방이 잡히는데 그것도 기록으로는 옳다 → 둘 다 채점)
"""
import json, glob, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor5")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/a5_5_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/q5_5_"))
res = {"sg0": [], "cur": [], "cover": 0, "n": 0, "map_sg": []}
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json")); sm = g.get("scene_meta")
    if not sm: continue
    live = {m["t"]: m for m in g["live"]}
    rt = g["room_types"]; rids = sorted(rt)
    rtypes = {}
    for v in sm["static"].values():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
    allt = set().union(*[set(c) for c in rtypes.values()])
    idf = {t: 1.0/max(sum(t in rtypes.get(r, ()) for r in rids), 1) for t in allt}
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    py, px = P // pw, P % pw
    arm = np.array([live[t]["room"] for t in ts])
    im = {}
    imf = os.path.join(os.path.realpath(hd), "initmap_owl.json")
    if os.path.exists(imf):
        best = {}
        for i2 in json.load(open(imf)):
            if i2["w"] > best.get(i2["type"], (0,))[0]:
                best[i2["type"]] = (i2["w"], i2["room"])
        im = {t: r for t, (w, r) in best.items()}
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        res["n"] += 1
        TS = QS[:, j] + STx[:, j]
        base = float(np.median(TS))
        th = np.quantile(TS, 0.98)
        hits = sorted(np.where(TS >= th)[0], key=lambda i: ts[i])
        evs = []
        for i in hits:
            if evs and ts[i] - ts[evs[-1][-1]] <= 90: evs[-1].append(i)
            else: evs.append([i])
        evs = [e for e in evs if len(e) >= 2]
        if not evs: continue
        res["cover"] += 1
        first = evs[0][:5]
        acc = {r: 0.0 for r in rids}
        for i in first:
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                w = float(S[i, c]) / (1 + d/6) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += w
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            t2 = sum(sc.values()) + 1e-9
            for r in rids: acc[r] += max(0., float(TS[i]) - base) * sc[r]/t2
        room = max(acc, key=acc.get)
        # 채점: 그 군집 시각의 진짜 방 (이동 전이면 초기 방, 후면 새 방 — 둘 다 옳음)
        t_ev = ts[first[0]]
        truth = v0["room"]
        for m in moves:
            if m["oid"] == oid and m["t"] <= t_ev: truth = m["to"]
        res["sg0"].append(room == v0["room"])
        res["cur"].append(room == truth)
        if v0["type"] in im:
            res["map_sg"].append(im[v0["type"]] == v0["room"])
print("=== 온라인 등록 (%s · 타겟 %d) ===" % (ROOT, res["n"]))
print("  커버리지 (확신 군집 존재)      **%.3f** (%d/%d)"
      % (res["cover"]/res["n"], res["cover"], res["n"]))
print("  등록 방 == 군집 시각의 진짜 방  **%.3f**" % np.mean(res["cur"]))
print("  등록 방 == t=0 초기 방        %.3f" % np.mean(res["sg0"]))
print("  (대조) 매핑워크 initmap sg    %.3f (n=%d)"
      % (np.mean(res["map_sg"]) if res["map_sg"] else 0, len(res["map_sg"])))
