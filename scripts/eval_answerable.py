#!/usr/bin/env python3
"""**답 가능성 계층별 채점** — 답을 맞출 수 있는 문제에서 얼마나 맞추는가.

    THOR_ROOT=data/thor4 python scripts/eval_answerable.py

균등 채점은 "원리적으로 답 불가능한 문제" 를 섞어 시스템 실력을 가린다. 관측
기록으로 계층을 가른다 (§51 의 --only-seen 원칙을 통합 파이프라인에 적용):

  T1 발견 가능      이동했고 새 방에서 3프레임+ 목격됨 → 검색이 답할 수 있다
  T2 부재추론 가능   이동했고 원래 방 재방문(목격은 없음) → 부재+belief 경로만
  T3 관측 불가      이동했고 아무 관측 없음 → 어떤 시스템도 belief 뿐
  T4 기록 유지      안 움직임 → 기록만으로 정답 (재확인 여부로 세분)
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.environ.get("A3_PREFIX", "/tmp/h4/cache/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/h4/cache/qc_")
AXP = os.environ.get("AX_PREFIX", "/tmp/h4/cache/ax_")
PR = json.load(open("data/thor_prior.json"))

rows = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    ZX = np.load(AXP + hn + ".npz", allow_pickle=True) if os.path.exists(AXP + hn + ".npz") else None
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
    if ZX is not None:
        an_ = list(ZX["anch"])
        ai = {k: sm["static"][a]["room"] for k, a in enumerate(an_) if a in sm["static"]}
        XS = ZX["s"][:, list(ai)]; Xr = [ai[k] for k in ai]
        XSc = XS - np.median(XS, axis=0, keepdims=True)
        XPp = ZX["p"][:, list(ai)]
        xy_ = np.stack([XPp // pw, XPp % pw], -1)
    AS = S[:, nT:]
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; sg = v0["room"]
        t0 = mv[-1]["t"] if mv else 0
        seen_new = sum(oid in live[t].get("vis", []) for t in ts if t > t0) >= 3 if mv else False
        revis = bool((arm[ts > t0] == sg).any()) if mv else bool((arm[len(arm)//2:] == sg).any())
        tier = ("T1" if mv and seen_new else "T2" if mv and revis
                else "T3" if mv else "T4r" if revis else "T4u")
        TS = QS[:, j] + STx[:, j]
        base = float(np.median(TS)); top = np.argsort(-TS)[:10]
        acc = {r: 0.0 for r in rids}
        for i in top:
            w = max(0.0, float(TS[i]) - base)
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                w2 = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += w2
            if ZX is not None:
                on = np.where(XSc[i] >= 0.15)[0]
                si_ = {r: 0.0 for r in rids}
                for k2 in on:
                    d = np.hypot(xy_[i, k2, 0]-cy, xy_[i, k2, 1]-cx)
                    si_[Xr[k2]] += float(XSc[i, k2]) / (1.0 + d/6.0)
                t2 = sum(si_.values())
                if t2 > 0:
                    for r in rids: sc[r] = sc[r]/(sum(sc.values())+1e-9) + si_[r]/t2
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            t3 = sum(sc.values()) + 1e-9
            for r in rids: acc[r] += w * sc[r] / t3
        find = max(acc, key=acc.get)
        # ── T1 공략: "충분히 확실한 검출 중 **가장 최근**" — 이동 전 기간이 길어
        # 상위권이 옛 자리 목격으로 도배되는 문제를 시간으로 가른다 (§eval_temporal
        # 의 실패와 달리, 문턱을 먼저 걸고 그 안에서 최신을 본다)
        def loc_of(idxs):
            a2 = {r: 0.0 for r in rids}
            for i in idxs:
                w = max(0.0, float(TS[i]) - base)
                cy, cx = py[i, ti], px[i, ti]
                sc = {r: 0.0 for r in rids}
                for c in range(nT, len(vocab)):
                    t = vocab[c]
                    if t not in idf or S[i, c] < .05: continue
                    d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                    w2 = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                    for r in rids:
                        if t in rtypes.get(r, ()): sc[r] += w2
                if ZX is not None:
                    on = np.where(XSc[i] >= 0.15)[0]
                    si_ = {r: 0.0 for r in rids}
                    for k2 in on:
                        d = np.hypot(xy_[i, k2, 0]-cy, xy_[i, k2, 1]-cx)
                        si_[Xr[k2]] += float(XSc[i, k2]) / (1.0 + d/6.0)
                    t2 = sum(si_.values())
                    if t2 > 0:
                        for r in rids: sc[r] = sc[r]/(sum(sc.values())+1e-9) + si_[r]/t2
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                t3 = sum(sc.values()) + 1e-9
                for r in rids: a2[r] += (w + 1e-6) * sc[r] / t3
            return max(a2, key=a2.get) if a2 else None
        top50 = np.argsort(-TS)[:50]
        recent = sorted(top50, key=lambda i: -ts[i])[:5]
        find_recent = loc_of(recent)
        inr = np.where(arm == sg)[0]
        drop = None
        if len(inr) >= 9:
            e = inr[:len(inr)//3]; l = inr[-len(inr)//3:]
            k2 = max(3, len(e)//3)
            sig = AS[e[np.argsort(-TS[e])[:k2]]].mean(0); sig /= (np.linalg.norm(sig)+1e-9)
            pe = AS[e] @ sig/(np.linalg.norm(AS[e],axis=1)+1e-9)
            pl = AS[l] @ sig/(np.linalg.norm(AS[l],axis=1)+1e-9)
            thp = np.quantile(pe, .7); ge = e[pe >= thp]; gl = l[pl >= thp]
            if len(ge) >= 3 and len(gl) >= 3:
                drop = float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9))
        bel = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                   for r in rids if r != sg))[1]
        rows.append(dict(tier=tier, sg=sg, tgt=tgt, find=find, find_recent=find_recent,
                         drop=drop, bel=bel))

drops = [r["drop"] for r in rows if r["drop"] is not None]
ta = float(np.quantile(drops, .95))
def sys_ans(r):
    if r["drop"] is not None and r["drop"] >= ta: return r["bel"]
    return r["sg"]

lab = {"T1": "T1 발견 가능 (새 방 목격)", "T2": "T2 부재추론만 가능",
       "T3": "T3 관측 불가 (belief 뿐)", "T4r": "T4 안 움직임·재확인됨",
       "T4u": "T4 안 움직임·미확인"}
print("=== 답 가능성 계층별 채점 · %s · n=%d ===" % (ROOT, len(rows)))
print("%-26s %-5s %-8s %-8s %-8s %s" % ("계층", "건수", "현시스템", "검색만", "이상적", "이상적의 정의"))
for t in ("T1", "T2", "T3", "T4r", "T4u"):
    rs = [r for r in rows if r["tier"] == t]
    if not rs: continue
    cur = np.mean([sys_ans(r) == r["tgt"] for r in rs])
    fnd = np.mean([r["find"] == r["tgt"] for r in rs])
    fr_ = np.mean([r["find_recent"] == r["tgt"] for r in rs])
    ideal = {"T1": 1.0, "T2": np.mean([r["bel"] == r["tgt"] for r in rs]),
             "T3": np.mean([r["bel"] == r["tgt"] for r in rs]),
             "T4r": 1.0, "T4u": 1.0}[t]
    idef = {"T1": "목격 프레임을 찾으면 1.0", "T2": "완벽 부재 + belief",
            "T3": "belief 가 상한", "T4r": "기록 유지 = 1.0", "T4u": "기록 유지 = 1.0"}[t]
    print("%-26s %-5d **%.3f**  %-8.3f 최신 **%.3f**  %-8.3f %s"
          % (lab[t], len(rs), cur, fnd, fr_, ideal, idef))
ans = [r for r in rows if r["tier"] != "T3"]
print("\n**답 가능 문제만(T3 제외) 현시스템: %.3f** (n=%d)"
      % (np.mean([sys_ans(r) == r["tgt"] for r in ans]), len(ans)))
print("전체(기존 방식):              %.3f (n=%d)"
      % (np.mean([sys_ans(r) == r["tgt"] for r in rows]), len(rows)))
