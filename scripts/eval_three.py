# 사용자가 정리한 **3경우 판정**을 그대로 구현해 최종 정답률을 잰다.
#   경우1  씬그래프=방1, 재방문 안 했거나 재방문해도 부재증거 없음  → 방1 답
#   경우2  씬그래프=방1, 재방문했고 **부재 증거 있음**              → belief → 방2 답
#   경우3  씬그래프=방1, 추가 관측 없음, 실제로는 없음              → 방1 답 (원리적 오답)
#
# ⚠️ 부재 문턱 하나가 **양방향으로** 손해를 낸다. 제자리 물체가 압도적으로 많으므로
# (실측 310 : 50 = 6:1) 오탐 하나가 미탐 여섯 개만큼 비싸다. AUC 만으로는 이득 여부를
# 알 수 없어 문턱을 훑어 손익 곡선을 그린다.
import json, glob, os, sys, numpy as np
from collections import Counter
MODE = sys.argv[1] if len(sys.argv) > 1 else "이미지"
ROOT = "data/thor3"
PR = json.load(open("data/thor_prior.json"))

rows = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(hd)
    fa, fq = "/tmp/a3_%s.npz" % hn, "/tmp/q3_%s.npz" % hn
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, ts, vocab, nT = za["s"], za["ts"], list(za["vocab"]), int(za["nT"])
    QT, QS = list(zq["tg"]), zq["si"]
    g = json.load(open(hd + "/gt.json")); rt = g["room_types"]; rids = sorted(rt)
    live = {m["t"]: m for m in g["live"]}
    nrt = Counter(rt[r] for r in rids)
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for oid, v in g["gt0"].items():
        if not v["room"] or cnt[v["type"]] > 1 or oid not in QT: continue
        ti = vocab.index(v["type"]) if v["type"] in vocab else -1
        if ti < 0: continue
        qi = QT.index(oid)
        sg = v["room"]                       # 씬그래프 기록 (t=0 관측)
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else sg      # 질의 시점 진짜 방
        t0 = mv[-1]["t"] if mv else 0
        TS = S[:, ti] if MODE == "글자" else QS[:, qi]
        # 씬그래프 방을 **재방문**했나 (기록 이후)
        idx = [i for i, t in enumerate(ts) if live[t]["room"] == sg]
        if not idx: 
            rows.append(dict(sg=sg, tgt=tgt, moved=bool(mv), revis=False, drop=None,
                             typ=v["type"], rids=rids, nrt=nrt, rt=rt)); continue
        # 부재 증거 = 그 방에서의 점수가 **초기 대비 얼마나 떨어졌나** (자기보정)
        early = [i for i in idx if ts[i] <= t0 or not mv]
        late = [i for i in idx if ts[i] > t0] if mv else idx[len(idx)//2:]
        if len(early) < 3 or len(late) < 3:
            rows.append(dict(sg=sg, tgt=tgt, moved=bool(mv), revis=False, drop=None,
                             typ=v["type"], rids=rids, nrt=nrt, rt=rt)); continue
        drop = float(np.quantile(TS[early], .9) - np.quantile(TS[late], .9))
        rows.append(dict(sg=sg, tgt=tgt, moved=bool(mv), revis=True, drop=drop,
                         typ=v["type"], rids=rids, nrt=nrt, rt=rt))

def belief(r):
    p = PR.get(r["typ"], {})
    sc = {q: p.get(r["rt"][q], .25) / max(r["nrt"][r["rt"][q]], 1) for q in r["rids"]}
    sc.pop(r["sg"], None)                    # 방1 은 부재로 배제됐다 → 그 다음 후보
    return max(sc, key=sc.get) if sc else r["sg"]

dr = [r["drop"] for r in rows if r["drop"] is not None]
print("=== 3경우 판정 [%s 질의] · 타겟 %d개 ===" % (MODE, len(rows)))
print("  이동 %d · 제자리 %d · 씬그래프 방 재방문 %d"
      % (sum(r["moved"] for r in rows), sum(not r["moved"] for r in rows),
         sum(r["revis"] for r in rows)))
print("  부재판정 없음(항상 방1)  **%.3f**" % np.mean([r["sg"] == r["tgt"] for r in rows]))
print("  belief 단독            **%.3f**"
      % np.mean([max({q: PR.get(r["typ"], {}).get(r["rt"][q], .25)/max(r["nrt"][r["rt"][q]],1)
                      for q in r["rids"]}.items(), key=lambda x: x[1])[0] == r["tgt"] for r in rows]))
print("  ── 부재 문턱 스윕 ──")
print("  %-8s %-8s %-8s %-8s %-8s" % ("문턱", "최종답", "오탐", "미탐", "부재발동"))
for q in (1.0, .95, .9, .8, .7, .6, .5, .3, .0):
    th = np.quantile(dr, q) if dr else 1e9
    ok = fa_ = mi = fire = 0
    for r in rows:
        ab = r["revis"] and r["drop"] is not None and r["drop"] >= th
        fire += ab
        ans = belief(r) if ab else r["sg"]
        ok += ans == r["tgt"]
        if ab and not r["moved"]: fa_ += 1
        if not ab and r["moved"]: mi += 1
    print("  %-8.2f **%.3f**   %-8d %-8d %d" % (q, ok/len(rows), fa_, mi, fire))
