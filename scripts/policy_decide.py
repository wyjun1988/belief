#!/usr/bin/env python3
"""학습된 판단(로지스틱)의 질의별 결정 파일을 만든다 — eval_online POLICY_DECISIONS 가 읽는다 (2026-09-24).
HSSD 는 집 단위 5겹 교차검증 예측(자기 집을 본 적 없는 모델의 결정), c2set 은 HSSD 전체로 학습한 모델의 결정(전이).
문턱은 HSSD 교차검증에서만 고른다(0.45). 출력 행: {house, oid, cand, record, p, switch}
  python scripts/policy_decide.py
"""
import json, os, sys, numpy as np
sys.argv=["x"]; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_probe.py")).read().split('for name in ("HSSD","c2set"):')[0])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
TH=float(os.environ.get("POLICY_TH", "0.45"))
def cands(name):
    out=[]; R={(r["house"],r["oid"]):r for r in (json.loads(l) for l in open(SETS[name][1]))}
    for l in open(SETS[name][0]):
        p=json.loads(l); fr=p["frames"]; rec=p["record"]
        pc=collections.Counter(f["proj"] for f in fr if f.get("proj") and f["proj"]!=rec); cc=collections.Counter(f["cam"] for f in fr if f.get("cam") and f["cam"]!=rec)
        c=(pc.most_common(1)[0][0] if pc else (cc.most_common(1)[0][0] if cc else None)); out.append((p["house"],p["oid"],rec,c))
    return out
# 학습용(정답 있는 결정 가능 질의)과 적용용(후보가 있는 모든 질의)을 따로 만든다 — 적용은 정답을 모른다
def feats_all(name):
    X,Y,G,meta=build(name)
    return X,Y,G,meta
XH,YH,GH,MH=feats_all("HSSD")
# 적용용 특징: build 는 정답이 두 갈래 밖인 질의·①② 밖 질의를 뺀다 → 적용을 위해 같은 특징을 정답 없이 다시 만든다
def feats_apply(name):
    pol, rows, mfs = SETS[name]
    M={}
    for mf in mfs:
        for l in open(mf):
            d=json.loads(l)
            if d.get("ans"): M[(d["house"],d["oid"],int(d["t"]))]=float(d["margin"])
    X=[]; K2=[]
    for l in open(pol):
        p=json.loads(l); fr=p["frames"]; rec=p["record"]
        pc=collections.Counter(f["proj"] for f in fr if f.get("proj") and f["proj"]!=rec); cc=collections.Counter(f["cam"] for f in fr if f.get("cam") and f["cam"]!=rec)
        cand=(pc.most_common(1)[0][0] if pc else (cc.most_common(1)[0][0] if cc else None))
        if cand is None: continue
        fc=[f for f in fr if f.get("proj")==cand]; fk=[f for f in fr if f.get("proj")==rec]
        mc=[M.get((p["house"],p["oid"],f["t"])) for f in fc]; mc=[m for m in mc if m is not None]
        dc=[f["dist"] for f in fc if f.get("dist") is not None]; tt=[f["t"] for f in fc]
        X.append([len(fr), len(fc), len(fk), sum(1 for f in fr if f.get("cam")==cand), sum(1 for f in fr if f.get("cam")==rec),
           float(np.mean(mc)) if mc else -3.0, float(np.max(mc)) if mc else -3.0, float(np.mean(dc)) if dc else 5.0, float(np.min(dc)) if dc else 5.0,
           sum(1 for f in fc if f.get("ray")), (max(tt)-min(tt)) if len(tt)>1 else 0, float(np.mean([f["s_ab"] for f in fc if f.get("s_ab") is not None] or [0])),
           int(p["alt"]==cand), p["n_rr"]]); K2.append((p["house"],p["oid"],rec,cand))
    return np.array(X,float), K2
for name, out in (("HSSD", os.path.expanduser("~/khcache/bench-v2full/policy_decisions.jsonl")), ("c2set", os.path.expanduser("~/khcache/bench-c2set/policy_decisions.jsonl"))):
    XA,KA=feats_apply(name); P=np.zeros(len(KA))
    if name=="HSSD":
        houses=np.array([k[0] for k in KA]); hs=sorted(set(houses)); folds={h:i%5 for i,h in enumerate(hs)}
        for f in range(5):
            trh=set(h for h in hs if folds[h]!=f); tr=np.array([g in trh for g in GH]); te=np.array([folds[h]==f for h in houses])
            m=LogisticRegression(max_iter=3000,C=0.5).fit(XH[tr],YH[tr]); P[te]=m.predict_proba(XA[te])[:,1]
    else:
        P=LogisticRegression(max_iter=3000,C=0.5).fit(XH,YH).predict_proba(XA)[:,1]
    with open(out,"w") as fo:
        for (h,o,rec,c),p in zip(KA,P): fo.write(json.dumps(dict(house=h,oid=o,record=rec,cand=c,p=round(float(p),4),switch=bool(p>=TH)), ensure_ascii=False)+"\n")
    print("%s 결정 %d행 · 바꿈 %d → %s" % (name, len(KA), int((P>=TH).sum()), out))
