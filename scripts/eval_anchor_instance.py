#!/usr/bin/env python3
"""**개체 앵커로 국소화** — exemplar 로 검출한 앵커 개체가 타입 가방을 넘는가.

    THOR_ROOT=data/thor3 python scripts/eval_anchor_instance.py

배경: 타입 가방(지금) 0.644 · 개체 GT 오라클(최근접3) 0.868. exemplar 검출이
그 사이 어디에 착지하는지가 이 실험이다. 개체는 방이 하나로 정해지므로
idf 도 방별 목록도 필요 없다 — **개체가 보이면 그 방이다.**

규칙: 프레임마다 exemplar 점수가 문턱을 넘는 앵커 개체 중, 타겟과 화면상
가까운 것 상위 k 가 자기 방에 투표. 점수는 개체별 자기보정(중앙값 차감).
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
A3P = os.environ.get("A3_PREFIX", "/tmp/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/qc_")
GTF = os.environ.get("GTFRAME", "1") == "1"
AXP = os.environ.get("AX_PREFIX", "/tmp/ax_")
K = 10


def ci(a):
    a = np.asarray(a, float); n = len(a)
    if n == 0: return 0., 0., 0.
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


res = {}
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fx = A3P + hn + ".npz", AXP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fx)): continue
    fq = QCP + hn + ".npz"
    if not GTF and not os.path.exists(fq): continue
    za = np.load(fa, allow_pickle=True); zx = np.load(fx, allow_pickle=True)
    zq = np.load(fq, allow_pickle=True) if not GTF else None
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    XS, XP, xts, anames = zx["s"], zx["p"], zx["ts"], list(zx["anch"])
    assert list(xts) == list(ts), "프레임 정렬 불일치"
    g = json.load(open(hd + "/gt.json")); sm = g.get("scene_meta")
    if not sm: continue
    live = {m["t"]: m for m in g["live"]}
    aroom = {a: sm["static"][a]["room"] for a in anames if a in sm["static"]}
    ai = [k for k, a in enumerate(anames) if a in aroom]
    rooms_of = np.array([aroom[anames[k]] for k in ai], object)
    XSc = XS[:, ai] - np.median(XS[:, ai], axis=0, keepdims=True)   # 개체별 자기보정
    xy = np.stack([XP[:, ai] // pw, XP[:, ai] % pw], -1)            # (프레임, 앵커, yx)
    py, px = P // pw, P % pw
    rids = sorted({v["room"] for v in sm["static"].values()})
    # 타입 가방 기준선용
    rtypes = {}
    for v in sm["static"].values():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
    allt = set().union(*[set(c) for c in rtypes.values()])
    idf = {t: 1.0/max(sum(t in rtypes.get(r, ()) for r in rids), 1) for t in allt}
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    arm = np.array([live[t]["room"] for t in ts])
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for oid, v in g["gt0"].items():
        if not v["room"] or cnt[v["type"]] > 1 or v["type"] not in vocab: continue
        ti = vocab.index(v["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v["room"]; t0 = mv[-1]["t"] if mv else 0
        if GTF:
            gtf = [i for i, t in enumerate(ts) if t > t0 and oid in live[t].get("vis", [])][:K]
        else:
            QT = list(zq["tg"])
            if oid not in QT: continue
            j = QT.index(oid)
            TS = zq["si"][:, j] + zq["st"][:, j]
            gtf = list(np.argsort(-TS)[:K])          # 실전: 시간 오라클 없음
        if len(gtf) < 3: continue

        def bag(i):
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                w = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += w
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            return max(sc, key=sc.get)

        def inst(i, th, kk):
            cy, cx = py[i, ti], px[i, ti]
            on = np.where(XSc[i] >= th)[0]
            if not len(on): return None
            d = np.hypot(xy[i, on, 0]-cy, xy[i, on, 1]-cx)
            near = on[np.argsort(d)[:kk]]
            rs = list(rooms_of[near])
            return max(set(rs), key=rs.count)

        def gtinst(i, kk):
            m = live[ts[i]]
            cx0, cy0 = (m.get("ctr", {}) or {}).get(oid) or (None, None)
            if cx0 is None: return None
            cand = sorted(((np.hypot(c[0]-cx0, c[1]-cy0), aroom[a])
                           for a, c in (m.get("anch") or {}).items()
                           if c and a in aroom), key=lambda x: x[0])[:kk]
            rs = [r for _, r in cand]
            return max(set(rs), key=rs.count) if rs else None

        def bag_sc(i):
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                w = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += w
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            tot = sum(sc.values()) + 1e-9
            return {r: sc[r]/tot for r in rids}

        def inst_sc(i, th):
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            on = np.where(XSc[i] >= th)[0]
            for k2 in on:
                d = np.hypot(xy[i, k2, 0]-cy, xy[i, k2, 1]-cx)
                sc[rooms_of[k2]] += float(XSc[i, k2]) / (1.0 + d/6.0)
            tot = sum(sc.values())
            return {r: sc[r]/tot for r in rids} if tot > 0 else None

        vt = lambda L: max(set([x for x in L if x]), key=[x for x in L if x].count) \
                       if any(L) else None
        res.setdefault("① 타입 가방 (지금)", []).append(vt([bag(i) for i in gtf]) == tgt)
        for th in (.15, .20, .25):
            res.setdefault("② exemplar 개체 최근접3 θ=%.2f" % th, []).append(
                vt([inst(i, th, 3) for i in gtf]) == tgt)
        # ④ 결합 — 가방·개체 사후확률의 합 (프레임별), 프레임 합산 후 argmax
        for th, wI in ((.15, 1.0), (.20, 1.0), (.20, 2.0)):
            acc = {r: 0.0 for r in rids}
            for i in gtf:
                b = bag_sc(i); n2 = inst_sc(i, th)
                for r in rids:
                    acc[r] += b[r] + (wI * n2[r] if n2 else 0)
            res.setdefault("④ 결합 θ=%.2f w=%.0f" % (th, wI), []).append(
                max(acc, key=acc.get) == tgt)
        res.setdefault("③ GT 개체 최근접3 (오라클)", []).append(
            vt([gtinst(i, 3) for i in gtf]) == tgt)

print("=== 개체 앵커 국소화 (프레임 %s · 상위 %d) ===" % ("GT" if GTF else "실전 검색", K))
for k in sorted(res):
    m, lo, hi = ci(res[k])
    print("  %-28s %.3f [%.3f %.3f]  n=%d" % (k, m, lo, hi, len(res[k])))
