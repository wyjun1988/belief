#!/usr/bin/env python3
"""국소화 축① — 확률적 방 융합 가중 스윕. (thor4 · 기준 머신 M1 Max)

    THOR_ROOT=data/thor4 python scripts/eval_locfusion.py

지금 이웃제한은 {카메라방·이웃 1.0 / 기타 0.25} 이진 곱이다. 방인지가 0.964 로
해결됐으니(r4_ Viterbi 예측방) 연속 가중 (같은방 w0, 이웃 w1, 기타 w2)로 바꾸고
스윕한다. 채점: 결합(타입가방+개체앵커) 국소화, 실전 프레임(문턱 후 최신 아님 —
상위10 점수가중, §78 레시피), 이동/제자리 분리.
"""
import json, glob, os, itertools
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/h4/cache/a3_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/h4/cache/qc_"))
AXP = os.path.expanduser(os.environ.get("AX_PREFIX", "~/khcache/h4/cache/ax_"))
RP = os.path.expanduser(os.environ.get("ROOM_PREFIX", "~/khcache/h4/cache/r4_"))

data = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq, fx, fr = A3P+hn+".npz", QCP+hn+".npz", AXP+hn+".npz", RP+hn+".npz"
    if not all(os.path.exists(f) for f in (fa, fq, fx, fr)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    zx = np.load(fx, allow_pickle=True); zr = np.load(fr, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json")); sm = g["scene_meta"]
    live = {m["t"]: m for m in g["live"]}
    rids = sorted(g["room_types"])
    prm = zr["room"]                                  # Viterbi 예측 방 (프레임별)
    rtypes = {}
    for v in sm["static"].values():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
    allt = set().union(*[set(c) for c in rtypes.values()])
    idf = {t: 1.0/max(sum(t in rtypes.get(r, ()) for r in rids), 1) for t in allt}
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    an_ = list(zx["anch"])
    if os.environ.get("MAPROOM", "gt") == "owl":
        # ② 검출 개체앵커 — 방 배정을 GT 가 아니라 initmap 군집(가장 가까운 같은
        # 타입, 2m 이내)에서. 못 찾으면 그 앵커는 버린다. GT 잔재의 마지막 조각.
        imf = os.path.join(os.path.realpath(hd), "initmap_owl.json")
        if not os.path.exists(imf): continue
        clusters = json.load(open(imf))
        ai = {}
        for k, a in enumerate(an_):
            v = sm["static"].get(a)
            if not v or "pos" not in v: continue
            best = (4.0, None)
            for c in clusters:
                if c["type"] != v["type"]: continue
                d = ((c["pos"][0]-v["pos"][0])**2 + (c["pos"][1]-v["pos"][1])**2) ** .5
                if d < best[0]: best = (d, c["room"])
            if best[1] and best[0] < 2.0: ai[k] = best[1]
    else:
        ai = {k: sm["static"][a]["room"] for k, a in enumerate(an_) if a in sm["static"]}
    if not ai: continue
    XS = zx["s"][:, list(ai)]; Xr = [ai[k] for k in ai]
    XSc = XS - np.median(XS, axis=0, keepdims=True)
    XPp = zx["p"][:, list(ai)]
    xy = np.stack([XPp // pw, XPp % pw], -1)
    py, px = P // pw, P % pw
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]
        TS = QS[:, j] + STx[:, j]
        base = float(np.median(TS))
        top = np.argsort(-TS)[:10]
        # 프레임별 방 원점수(방융합 전)를 선계산: bag(sc_b) + instance(sc_i)
        pre = []
        for i in top:
            cy, cx = py[i, ti], px[i, ti]
            scb = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                w = float(S[i, c]) / (1 + d/6) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): scb[r] += w
            sci = {r: 0.0 for r in rids}
            on = np.where(XSc[i] >= 0.15)[0]
            for k2 in on:
                d = np.hypot(xy[i, k2, 0]-cy, xy[i, k2, 1]-cx)
                sci[Xr[k2]] += float(XSc[i, k2]) / (1 + d/6)
            pre.append(dict(w=max(0., float(TS[i])-base), scb=scb, sci=sci,
                            room=prm[i]))
        data.append(dict(tgt=tgt, moved=bool(mv), pre=pre, rids=rids, adj=adj))

print("타겟 %d 적재 · 가중 스윕" % len(data))
def run(w0, w1, w2):
    hit_m, hit_s = [], []
    for d in data:
        acc = {r: 0.0 for r in d["rids"]}
        for p in d["pre"]:
            nb = d["adj"].get(p["room"], set())
            sc = {}
            for r in d["rids"]:
                g_ = w0 if r == p["room"] else w1 if r in nb else w2
                b = p["scb"][r]; i_ = p["sci"][r]
                tb = sum(p["scb"].values()) + 1e-9; ti_ = sum(p["sci"].values())
                sc[r] = (b/tb + (i_/ti_ if ti_ > 0 else 0)) * g_
            t2 = sum(sc.values()) + 1e-9
            top2 = sorted((v/t2 for v in sc.values()), reverse=True)[:2]
            conf = (top2[0] - top2[1]) if len(top2) > 1 else 1.0   # ③ 프레임 확신
            wgt = p["w"] * (conf if os.environ.get("CONF", "0") == "1" else 1.0)
            for r in d["rids"]: acc[r] += wgt * sc[r]/t2
        ans = max(acc, key=acc.get)
        (hit_m if d["moved"] else hit_s).append(ans == d["tgt"])
    n = len(hit_m) + len(hit_s)
    tot = (sum(hit_m) + sum(hit_s)) / n
    return tot, np.mean(hit_m), np.mean(hit_s)

print("%-22s %-8s %-8s %-8s" % ("(같은방,이웃,기타)", "전체", "이동", "제자리"))
best = (0, None)
for w0, w1, w2 in [(1, .2, .05), (1, .3, .05)]:
    t, m, s_ = run(w0, w1, w2)
    if t > best[0]: best = (t, (w0, w1, w2))
    print("%-22s **%.3f** %-8.3f %.3f" % ((w0, w1, w2), t, m, s_))
print("최적:", best)
