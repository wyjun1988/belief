#!/usr/bin/env python3
"""**belief 핸드오프 평가 — belief 자체 평가와 분리한다.**

    THOR_ROOT=data/thor4 python scripts/eval_handoff.py

세 시나리오의 분기(부재 발동 → belief 로 넘김)가 제대로 되는지를,
넘겨받은 belief 가 맞히는지와 **따로** 잰다. 섞으면 어느 쪽 탓인지 모른다.

  게이트 품질   발동 정밀도 = P(정말 떠났다 | 발동) · 재현율 = P(발동 | 떠났고 재방문)
  핸드오프 후   belief 정답률 (천장 0.58 — §75)
  경우 분포     0/1a/1b/2/3 각각 몇 건이고 각 경우의 정답률
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.environ.get("A3_PREFIX", "/tmp/h4/cache/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/h4/cache/qc_")
QA = float(os.environ.get("QA", "0.95"))          # 부재 문턱 분위
QF = float(os.environ.get("QF", "0.97"))          # 검색 확신 분위 (경우 0)
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
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    AS = S[:, nT:]
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; sg = v0["room"]
        TS = QS[:, j] + STx[:, j]
        base = float(np.median(TS))
        top = np.argsort(-TS)[:10]
        conf = float(TS[top].mean() - base)
        acc = {r: 0.0 for r in rids}
        for i in top:
            w = max(0.0, float(TS[i]) - base)
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                sc0 = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += sc0
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            tot = sum(sc.values()) + 1e-9
            for r in rids: acc[r] += w * sc[r] / tot
        find_room = max(acc, key=acc.get)
        inr = np.where(arm == sg)[0]
        drop = None; revis = False
        if len(inr) >= 9:
            e = inr[:len(inr)//3]; l = inr[-len(inr)//3:]
            k2 = max(3, len(e)//3)
            sig = AS[e[np.argsort(-TS[e])[:k2]]].mean(0)
            sig /= (np.linalg.norm(sig) + 1e-9)
            pe = AS[e] @ sig / (np.linalg.norm(AS[e], axis=1) + 1e-9)
            pl = AS[l] @ sig / (np.linalg.norm(AS[l], axis=1) + 1e-9)
            thp = np.quantile(pe, .7)
            ge = e[pe >= thp]; gl = l[pl >= thp]
            if len(ge) >= 3 and len(gl) >= 3:
                drop = float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9))
                revis = True
        bel = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]], 1), r)
                   for r in rids if r != sg))[1]
        rows.append(dict(sg=sg, tgt=tgt, moved=bool(mv), conf=conf, find=find_room,
                         drop=drop, revis=revis, bel=bel))

n = len(rows)
confs = sorted(r["conf"] for r in rows)
drops = [r["drop"] for r in rows if r["drop"] is not None]
tf = confs[min(int(len(confs)*QF), n-1)]
ta = float(np.quantile(drops, QA))

def case_of(r):
    if r["conf"] >= tf and r["find"] != r["sg"]: return "0"
    if r["drop"] is not None and r["drop"] >= ta: return "2"
    if r["revis"]: return "1a"
    return "1b3"

by = {}
for r in rows: by.setdefault(case_of(r), []).append(r)
print("=== belief 핸드오프 평가 · %s · 타겟 %d (이동 %d) · θf=q%.2f θa=q%.2f ==="
      % (ROOT, n, sum(r["moved"] for r in rows), QF, QA))
print("\n── 경우 분포와 각 경우의 정답률 ──")
lab = {"0": "경우0 발견(검색이 다른 방 확신)", "2": "경우2 부재발동 → belief",
       "1a": "경우1a 재방문·이상없음 → sg", "1b3": "경우1b/3 확인못함 → sg"}
for k in ("0", "2", "1a", "1b3"):
    rs = by.get(k, [])
    if not rs: print("  %-28s 0건" % lab[k]); continue
    ans = [(r["find"] if k == "0" else r["bel"] if k == "2" else r["sg"]) for r in rs]
    ok = np.mean([a == r["tgt"] for a, r in zip(ans, rs)])
    mvf = np.mean([r["moved"] for r in rs])
    print("  %-28s %3d건 · 정답 %.3f · 이동비율 %.2f" % (lab[k], len(rs), ok, mvf))

f2 = by.get("2", [])
gate_p = np.mean([r["moved"] for r in f2]) if f2 else 0
elig = [r for r in rows if r["moved"] and r["revis"]]
gate_r = np.mean([case_of(r) == "2" for r in elig]) if elig else 0
print("\n── 게이트 품질 (belief 와 무관) ──")
print("  발동 정밀도 P(떠남|발동)        **%.3f**  (발동 %d건)" % (gate_p, len(f2)))
print("  발동 재현율 P(발동|떠남·재방문)   **%.3f**  (해당 %d건)" % (gate_r, len(elig)))
print("\n── 핸드오프 후 belief (별도 평가) ──")
tm = [r for r in f2 if r["moved"]]
if tm:
    print("  정말 떠난 건에서 belief 정답     **%.3f**  (n=%d · 이론천장 0.58)"
          % (np.mean([r["bel"] == r["tgt"] for r in tm]), len(tm)))
allm = [r for r in rows if r["moved"]]
print("  (참고) 모든 이동 물체에서 belief   %.3f  (n=%d)"
      % (np.mean([r["bel"] == r["tgt"] for r in allm]), len(allm)))
tot = np.mean([( (r["find"] if case_of(r)=="0" else r["bel"] if case_of(r)=="2" else r["sg"]) == r["tgt"]) for r in rows])
sta = np.mean([r["sg"] == r["tgt"] for r in rows])
print("\n  전체 정답 %.3f vs 정지 지도 %.3f" % (tot, sta))
