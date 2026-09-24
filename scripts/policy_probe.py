#!/usr/bin/env python3
"""학습된 판단(정책)의 신호 검사 — 손규칙보다 방을 더 잘 고를 여지가 있나 (2026-09-24, 사용자 제안 2 의 좁힌 형태).

입력: eval_online POLICY_DUMP(질의마다 채택 관문을 다 거친 프레임의 투영 방·거리·카메라방·제로샷 점수) + 같은 실행의 행 덤프(정답) + 채택 판정 마진.
결정: 질의마다 {기록 유지, 후보 방으로 바꿈} 두 갈래. 후보 방 = 통과 프레임 투영의 최다 비기록 방(없으면 카메라방 최다).
평가: 손규칙(현행 alt) 대비, 집 단위 교차검증(HSSD) + 다른 데이터셋 전이(HSSD→c2set).
  python scripts/policy_probe.py
"""
import json, os, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
K=os.path.expanduser("~/khcache")
SETS={"HSSD": (K+"/bench-v2full/policy_hssd.jsonl", K+"/bench-v2full/rows_CH_정책덤프_HSSD.jsonl", [K+"/bench-v2alias/scores/margin_all_standfresh.jsonl"]),
      "c2set": (K+"/bench-c2set/policy_c2set.jsonl", K+"/bench-c2set/rows_CH_정책덤프_c2set.jsonl", [K+"/rtx_0923/out_0923/adopt_margin_c2set_fix_0923.jsonl"])}
def build(name):
    pol, rows, mfs = SETS[name]
    R={(r["house"],r["oid"]):r for r in (json.loads(l) for l in open(rows))}
    M={}
    for mf in mfs:
        for l in open(mf):
            d=json.loads(l)
            if d.get("ans"): M[(d["house"],d["oid"],int(d["t"]))]=float(d["margin"])
    X=[]; Y=[]; G=[]; meta=[]
    for l in open(pol):
        p=json.loads(l); k=(p["house"],p["oid"]); r=R.get(k)
        if not r or r["case"] not in ("①이동없음","②재촬영"): continue
        fr=p["frames"]; rec=p["record"]
        proj=[f["proj"] for f in fr if f.get("proj")]; cam=[f["cam"] for f in fr if f.get("cam")]
        pc=collections.Counter(x for x in proj if x!=rec); cc=collections.Counter(x for x in cam if x!=rec)
        cand=(pc.most_common(1)[0][0] if pc else (cc.most_common(1)[0][0] if cc else None))
        if cand is None: continue                      # 바꿀 후보가 없으면 결정할 것도 없다(기록 유지와 같다)
        fc=[f for f in fr if f.get("proj")==cand]; fk=[f for f in fr if f.get("proj")==rec]
        mc=[M.get((p["house"],p["oid"],f["t"])) for f in fc]; mc=[m for m in mc if m is not None]
        dc=[f["dist"] for f in fc if f.get("dist") is not None]
        tt=[f["t"] for f in fc]
        x=[len(fr), len(fc), len(fk), sum(1 for f in fr if f.get("cam")==cand), sum(1 for f in fr if f.get("cam")==rec),
           float(np.mean(mc)) if mc else -3.0, float(np.max(mc)) if mc else -3.0, float(np.mean(dc)) if dc else 5.0, float(np.min(dc)) if dc else 5.0,
           sum(1 for f in fc if f.get("ray")), (max(tt)-min(tt)) if len(tt)>1 else 0, float(np.mean([f["s_ab"] for f in fc if f.get("s_ab") is not None] or [0])),
           int(p["alt"]==cand), p["n_rr"]]
        tgt=r["tgt"]
        y = 1 if tgt==cand else (0 if tgt==rec else -1)
        if y<0: continue                                # 정답이 둘 다 아니면 이 두 갈래 결정으로는 못 푼다
        X.append(x); Y.append(y); G.append(p["house"]); meta.append((r["case"], p["alt"]==cand, k))
    return np.array(X,float), np.array(Y), np.array(G), meta
FEAT=["통과프레임","후보방 투영","기록방 투영","후보방 카메라","기록방 카메라","후보 마진평균","후보 마진최대","후보 거리평균","후보 거리최소","후보 광선","후보 시간폭","후보 s_ab","규칙이 후보 선택","걸은 프레임"]
def score(pred, Y, meta):
    one=[(p,y) for p,y,m in zip(pred,Y,meta) if m[0]=="①이동없음"]; two=[(p,y) for p,y,m in zip(pred,Y,meta) if m[0]=="②재촬영"]
    acc=lambda v: sum(1 for p,y in v if p==y)
    return acc(one), len(one), acc(two), len(two)
for name in ("HSSD","c2set"):
    X,Y,G,meta=build(name); rule=np.array([int(m[1]) for m in meta])
    print("== %s: 결정 가능한 질의 %d (① %d · ②%d) · 정답이 후보방 %d · 기록방 %d" % (name, len(Y), sum(1 for m in meta if m[0]=="①이동없음"), sum(1 for m in meta if m[0]=="②재촬영"), int(Y.sum()), int((Y==0).sum())))
    a1,n1,a2,n2=score(rule,Y,meta); print("   손규칙        ① %d/%d · ② %d/%d · 합 %d" % (a1,n1,a2,n2,a1+a2))
    if name=="HSSD":
        for mname, mk in (("로지스틱", lambda: LogisticRegression(max_iter=2000, C=0.5)), ("부스팅", lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05))):
            pred=np.zeros(len(Y),int)
            for tr,te in GroupKFold(n_splits=5).split(X,Y,G):
                m=mk().fit(X[tr],Y[tr]); pred[te]=m.predict(X[te])
            a1,n1,a2,n2=score(pred,Y,meta); print("   %s(집단위5겹) ① %d/%d · ② %d/%d · 합 %d" % (mname,a1,n1,a2,n2,a1+a2))
        XH,YH=X,Y
        lr=LogisticRegression(max_iter=2000, C=0.5).fit(XH,YH); gb=GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05).fit(XH,YH)
        imp=sorted(zip(FEAT, gb.feature_importances_), key=lambda x:-x[1])[:6]; print("   부스팅 중요 특징:", ", ".join("%s %.2f" % t for t in imp))
    else:
        for mname, m in (("로지스틱(HSSD→)", lr), ("부스팅(HSSD→)", gb)):
            a1,n1,a2,n2=score(m.predict(X),Y,meta); print("   %s ① %d/%d · ② %d/%d · 합 %d" % (mname,a1,n1,a2,n2,a1+a2))
