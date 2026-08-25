#!/usr/bin/env python3
"""**앵커의 어느 단계에서 손실이 나나.** 세 수준을 같은 프레임에서 비교한다.

  A  OWL 글자 점수로 정적 **타입** 인식      ← 지금 방식 (천장 0.652)
  B  GT 로 보이는 정적 **타입**              ← 검출을 완벽히 줌
  C  GT 로 보이는 정적 **인스턴스**          ← 검출 + 개체식별을 완벽히 줌

  A→B  = 검출 실패 몫 (가설① 앵커가 프레임에 안 나온다)
  B→C  = 개체 모호성 몫 (가설②③ 같은 타입이 여러 방에 있다 / 글자라 개체가 지워진다)

B→C 가 크면 **앵커도 exemplar 로 찾아야 한다**는 뜻이다.
"""
import json, glob, os, sys, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
K = int(os.environ.get("TOPK", "10"))


def ci(a):
    a = np.asarray(a, float); n = len(a)
    if n == 0: return 0., 0., 0.
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


res = {}; seen_anchor = []; ninst = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = "/tmp/a3_%s.npz" % hn, "/tmp/qc_%s.npz" % hn
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, ts, vocab, nT = za["s"], za["ts"], list(za["vocab"]), int(za["nT"])
    QT, QS, ST_ = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    sm = g.get("scene_meta")
    if not sm: continue
    live = {m["t"]: m for m in g["live"]}
    rt = g["room_types"]
    inst_room = {k: v["room"] for k, v in sm["static"].items()}
    inst_type = {k: v["type"] for k, v in sm["static"].items()}
    rids = sorted({r for r in inst_room.values()})
    if len(rids) < 2: continue
    rtypes = {}
    for k, v in sm["static"].items():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
    idf_t = {t: 1.0/max(sum(t in rtypes[r] for r in rids), 1)
             for t in set().union(*[set(rtypes[r]) for r in rids])}
    ninst.append(np.mean([len(set(v for k, v in inst_room.items() if inst_type[k] == t))
                          for t in idf_t]))
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1: continue
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; t0 = mv[-1]["t"] if mv else 0
        # **프레임은 GT 로 준다** — 천장을 재는 것이므로 프레임 선택 오류를 배제
        gtf = [i for i, t in enumerate(ts) if t > t0 and oid in live[t].get("vis", [])][:K]
        if len(gtf) < 3: continue
        ti = vocab.index(v0["type"]) if v0["type"] in vocab else -1
        for lvl in ("A", "B", "C"):
            votes = []
            for i in gtf:
                m = live[ts[i]]
                sc = {r: 0.0 for r in rids}
                if lvl == "A":                       # OWL 글자 점수 (지금 방식)
                    for c in range(nT, len(vocab)):
                        t = vocab[c]
                        if t not in idf_t or S[i, c] < .05: continue
                        for r in rids:
                            if t in rtypes[r]: sc[r] += float(S[i, c]) * idf_t[t]
                else:
                    an = [a for a in m.get("anch", {}) if a in inst_room]
                    if lvl == "B":                   # GT 타입 (검출 완벽, 개체 모름)
                        for t in {inst_type[a] for a in an}:
                            if t not in idf_t: continue
                            for r in rids:
                                if t in rtypes[r]: sc[r] += idf_t[t]
                    else:                            # GT 인스턴스 (개체까지 완벽)
                        for a in an: sc[inst_room[a]] += 1.0
                    if lvl == "B": seen_anchor.append(len(an))
                nb = {m["room"]} | adj.get(m["room"], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                votes.append(max(rids, key=lambda r: sc[r]))
            res.setdefault(lvl, []).append(max(set(votes), key=votes.count) == tgt)

lab = {"A": "A  OWL 글자로 타입 인식 (지금)", "B": "B  GT 타입 (검출 완벽)",
       "C": "C  GT 인스턴스 (개체까지 완벽)"}
print("=== 앵커 수준별 방 특정 (프레임은 GT · 상위 %d) ===" % K)
for k in ("A", "B", "C"):
    if k not in res: continue
    m, lo, hi = ci(res[k])
    print("  %-30s %.3f [%.3f %.3f]  n=%d" % (lab[k], m, lo, hi, len(res[k])))
if "A" in res and "C" in res:
    a, b, c = np.mean(res["A"]), np.mean(res["B"]), np.mean(res["C"])
    print("\n  **A→B 검출 실패 몫  %+.3f**" % (b - a))
    print("  **B→C 개체 모호성 몫 %+.3f**" % (c - b))
print("\n  프레임당 보이는 정적 인스턴스 중앙 %.0f개 · 타입당 인스턴스 평균 %.1f개"
      % (np.median(seen_anchor) if seen_anchor else 0, np.mean(ninst)))
