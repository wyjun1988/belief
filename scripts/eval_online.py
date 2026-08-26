#!/usr/bin/env python3
"""**온라인 상태기계 v1** — 동적 씬그래프를 시간 순서로 통째로 돌린다. (헤드라인)

    THOR_ROOT=data/thor4 SG_INIT=gt|hybrid python scripts/eval_online.py

물체마다 기록(record) 하나. 시간 순서로:
  초기화   SG_INIT=gt → t=0 GT 방 / hybrid → initmap(있으면)+온라인 등록(첫 군집)
  갱신     확신 군집(문턱 q0.98·2장+)을 검증기(실측 운용점 모의: 진짜 0.42 수용,
          혼동 개체당 0.05 로 전부-속음)로 확인 → 군집 2장+ 확인되면 그 방으로 덮어씀
          ⚠️ 덮어쓰기는 기록과 **다른 방**이 복수 확인될 때만 (T4 보호 비대칭 규칙)
  질의(끝) 기록 방 부재 게이팅(온라인) 발동 시 belief(기록 제외) / 아니면 기록
"""
import json, glob, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/h4/cache/a3_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/h4/cache/qc_"))
SG_INIT = os.environ.get("SG_INIT", "gt")
P_ACC, P_ID = 0.42, 0.05
PR = json.load(open("data/thor_prior.json"))

res = {"rec": [], "sys": [], "static": [], "moved_sys": [], "moved_rec": [],
       "case": Counter()}
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
    nrt = Counter(rt[r] for r in rids)
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
    AS = S[:, nT:]
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
        tgt = mv[-1]["to"] if mv else v0["room"]
        TS = QS[:, j] + STx[:, j]
        vis = np.array([oid in live[t].get("vis", []) for t in ts])
        base = float(np.median(TS))
        _vr = np.random.default_rng(hash((hn, oid)) % 2**31)

        def loc_of(idx):
            acc = {r: 0.0 for r in rids}
            for i in idx:
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
            return max(acc, key=acc.get) if acc else None

        # 군집 (시간순)
        th = np.quantile(TS, 0.98)
        hits = sorted(np.where(TS >= th)[0], key=lambda i: ts[i])
        evs = []
        for i in hits:
            if evs and ts[i] - ts[evs[-1][-1]] <= 90: evs[-1].append(i)
            else: evs.append([i])
        evs = [e for e in evs if len(e) >= 2]

        def ident(i):
            m = live[ts[i]]
            cx = (P[i, ti] % pw + .5) / pw * 384
            cy = (P[i, ti] // pw + .5) / ph * 384
            best = (1e9, None)
            for o2, c in list((m.get("ctr") or {}).items()) + \
                         list((m.get("anch") or {}).items()):
                if not c: continue
                d = np.hypot(c[0]-cx, c[1]-cy)
                if d < best[0]: best = (d, o2)
            return best[1] if best[0] < 40 else "bg%d" % (i // 20)

        coin = {}
        def verify(i):
            if vis[i]: return _vr.random() < P_ACC
            k = ident(i)
            if k not in coin: coin[k] = _vr.random() < P_ID
            return coin[k]

        # ── 상태기계 ──
        # v3: 기록은 **불변** — 질의 시점에 검증된 최신 증거와 마진 비교만 한다.
        # (영구 덮어쓰기는 국소화 노이즈로 정지 물체 기록을 오염시킨다 — v1 실측 −0.11)
        record = (v0["room"] if SG_INIT == "gt" else im.get(v0["type"]))
        ver_all = []
        for e in evs:
            ver = [i for i in e[:6] if verify(i)]
            if len(ver) >= 2:
                ver_all += ver
        if record is None and ver_all:
            record = loc_of(sorted(ver_all)[:5])   # 온라인 등록 (이른 것)
        # 질의용: 최신 검증 3장의 방 질량
        alt = None
        if ver_all:
            recent = sorted(ver_all, key=lambda i: -ts[i])[:3]
            mass = {r: 0.0 for r in rids}
            for i in recent:
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
                for r in rids: mass[r] += sc[r]/t2
            top = max(mass, key=mass.get)
            if record and top != record and mass[top] > 2.0 * mass.get(record, 0):
                alt = top                           # 마진 게이트 통과 — 경우 0
        if record is None:
            record = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                          for r in rids))[1]
        # 질의: 기록 방 부재 게이팅 (온라인 앞/뒤 1/3 + 앵커 게이팅)
        inr = np.where(arm == record)[0]
        fired = False
        if len(inr) >= 9:
            e_, l_ = inr[:len(inr)//3], inr[-len(inr)//3:]
            k2 = max(3, len(e_)//3)
            sig = AS[e_[np.argsort(-TS[e_])[:k2]]].mean(0)
            sig /= (np.linalg.norm(sig) + 1e-9)
            pe = AS[e_] @ sig/(np.linalg.norm(AS[e_],axis=1)+1e-9)
            pl = AS[l_] @ sig/(np.linalg.norm(AS[l_],axis=1)+1e-9)
            thp = np.quantile(pe, .7)
            ge, gl = e_[pe >= thp], l_[pl >= thp]
            if len(ge) >= 3 and len(gl) >= 3:
                drop = float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9))
                fired = drop >= 0.055        # thor4 q0.95 근사 절대문턱
        if alt is not None:
            ans = alt
            res["case"]["c0"] += 1
        elif fired:
            ans = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                       for r in rids if r != record))[1]
            res["case"]["c2"] += 1
        else:
            ans = record
            res["case"]["rec"] += 1
        res["sys"].append(ans == tgt)
        res["rec"].append(record == tgt)
        res["static"].append(v0["room"] == tgt)
        if mv:
            res["moved_sys"].append(ans == tgt); res["moved_rec"].append(record == tgt)

n = len(res["sys"])
print("=== 온라인 상태기계 v1 · %s · SG_INIT=%s · n=%d ===" % (ROOT, SG_INIT, n))
print("  정지 지도(t=0 GT)        %.3f" % np.mean(res["static"]))
print("  **기록(갱신 후)**         **%.3f**" % np.mean(res["rec"]))
print("  **최종 답(부재분기 포함)** **%.3f**" % np.mean(res["sys"]))
print("  이동만: 기록 %.3f · 최종 %.3f (n=%d)"
      % (np.mean(res["moved_rec"]), np.mean(res["moved_sys"]), len(res["moved_sys"])))
print("  분기: %s" % dict(res["case"]))
