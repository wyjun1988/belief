#!/usr/bin/env python3
"""**능력별 분해 평가** — 시스템을 여섯 능력으로 쪼개 각각 채점한다. (thor4)

    python scripts/eval_abilities4.py

  A 방 인지      프레임의 정적 점수만으로 그 프레임의 방을 맞히나
  B 프레임 선택   타겟 exemplar 로 타겟이 보이는 프레임을 골라내나
  C 국소화       고른 프레임에서 물체의 방을 맞히나 (이동/제자리 분리)
  D 재방문 판정   "씬그래프 방을 그 뒤에 다시 봤다" 를 (예측 방으로) 아나
  E 부재 검출     떠난 물체의 점수 하락을 게이팅으로 가르나 (운용점 포함)
  F belief      사전확률 argmax (이동 물체 · 이론천장 대비)
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.environ.get("A3_PREFIX", "/tmp/h4/cache/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/h4/cache/qc_")
PR = json.load(open("data/thor_prior.json"))
MV = json.load(open("data/thor_move.json"))

A = {"ok": 0, "n": 0}
B = {"p5": [], "auc": []}
C = {"mv": [], "st": []}
D = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
E = {"mv": [], "st": []}
F = {"hit": [], "ceil": []}

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
    arm = np.array([live[t]["room"] for t in ts])
    # ── A 방 인지: 정적 점수 행렬 → 방 사후확률 (행렬화) ──
    M = np.zeros((len(vocab) - nT, len(rids)))
    for c in range(nT, len(vocab)):
        t = vocab[c]
        if t not in idf: continue
        for k, r in enumerate(rids):
            if t in rtypes.get(r, ()): M[c - nT, k] = idf[t]
    post = (S[:, nT:] * (S[:, nT:] >= .05)) @ M
    pred_room = np.array(rids, object)[post.argmax(1)]
    A["ok"] += int((pred_room == arm).sum()); A["n"] += len(arm)
    py, px = P // pw, P % pw
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    AS = S[:, nT:]
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; sg = v0["room"]
        t0 = mv[-1]["t"] if mv else 0
        TS = QS[:, j] + STx[:, j]
        vis = np.array([oid in live[t].get("vis", []) for t in ts])
        # ── B 프레임 선택 ──
        if 3 <= vis.sum() < len(vis):
            order = np.argsort(-TS)
            B["p5"].append(float(vis[order[:5]].mean()))
            p_, n_ = TS[vis], TS[~vis]
            B["auc"].append(float((p_[:, None] > n_[None, :]).mean()))
        # ── C 국소화 (통합과 같은 레시피) ──
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
            _ar = arm[i] if os.environ.get("NB", "gt") == "gt" else pred_room[i]
            nb = {_ar} | adj.get(_ar, set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            t2 = sum(sc.values()) + 1e-9
            for r in rids: acc[r] += w * sc[r] / t2
        C["mv" if mv else "st"].append(max(acc, key=acc.get) == tgt)
        # ── D 재방문 판정 (예측 방 기준) — 이동 물체만 ──
        if mv:
            aft = ts > t0
            gtrev = bool((arm[aft] == sg).any())
            prrev = bool((pred_room[aft] == sg).any())
            if gtrev and prrev: D["tp"] += 1
            elif not gtrev and prrev: D["fp"] += 1
            elif gtrev and not prrev: D["fn"] += 1
            else: D["tn"] += 1
        # ── E 부재 (앵커 게이팅 하락) ──
        inr = np.where(arm == sg)[0]
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
                E["mv" if mv else "st"].append(
                    float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9)))
        # ── F belief ──
        if mv:
            F["hit"].append(max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                                 for r in rids if r != sg))[1] == tgt)
            dd = MV["dest"].get(v0["type"], {})
            pool = [r for r in rids if r != sg]
            w3 = np.array([dd.get(rt[r], .25)/max(nrt[rt[r]],1) for r in pool])
            F["ceil"].append(float(w3.max()/w3.sum()) if w3.sum() > 0 else 1/len(pool))

print("=== 능력별 분해 (thor4 · 60채) ===")
print("A 방 인지(프레임)     **%.3f**  (n=%d)" % (A["ok"]/max(A["n"],1), A["n"]))
print("B 프레임 선택        정밀도@5 **%.3f** · AUC %.3f  (n=%d)"
      % (np.mean(B["p5"]), np.mean(B["auc"]), len(B["p5"])))
print("C 국소화            이동 **%.3f** (n=%d) · 제자리 %.3f (n=%d)"
      % (np.mean(C["mv"]), len(C["mv"]), np.mean(C["st"]), len(C["st"])))
pr_ = D["tp"]/max(D["tp"]+D["fp"],1); rc_ = D["tp"]/max(D["tp"]+D["fn"],1)
print("D 재방문 판정(예측방)  정밀 **%.3f** · 재현 **%.3f**  (이동 %d건)"
      % (pr_, rc_, sum(D.values())))
a = np.array(E["mv"]); b = np.array(E["st"])
auc = (a[:, None] > b[None, :]).mean() + .5*(a[:, None] == b[None, :]).mean()
print("E 부재 검출          AUC **%.3f**  (이동 %d · 제자리 %d)" % (auc, len(a), len(b)))
for q in (.98, .95, .9, .8):
    th = np.quantile(np.concatenate([a, b]), q)
    tp = (a >= th).sum(); fp = (b >= th).sum()
    print("   문턱 q%.2f        정밀 %.3f · 재현 %.3f  (발동 %d)"
          % (q, tp/max(tp+fp,1), tp/len(a), tp+fp))
print("F belief(이동)       실측 **%.3f** · 이론천장 %.3f  (n=%d)"
      % (np.mean(F["hit"]), np.mean(F["ceil"]), len(F["hit"])))
