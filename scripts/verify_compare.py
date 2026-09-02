#!/usr/bin/env python3
"""검증기 비교 — 같은 후보 크롭 집합에서 VLM 2AFC 와 exemplar 재식별 중 무엇이 진짜를 가르는가.

    THOR_ROOT=data/hssd20S A3_PREFIX=... QC_PREFIX=... SCORES=t1_scores.jsonl \\
      python scripts/verify_compare.py

AUDIT_20260902 (a): VLM s_ab 는 "머그냐 컵이냐"(범주)를 묻고 AUC 0.78. exemplar 코사인
(qc 캐시 si = 패치×질의 코사인 최댓값)은 "**그** 머그냐"(개체)를 재고 검색 AUC 0.93 —
그런데 채택 판정엔 한 번도 안 썼다. 여기서 **동일 크롭**에 대해 정면 비교한다.
진짜 = 이동 후 프레임에서 그 물체가 보임(GT). 점수 6종의 풀 AUC · 타겟내 z-정규화 AUC ·
오검출 기각 0.95 에서의 진짜 수용률.
"""
import glob, json, os
import numpy as np

ROOT = os.environ.get("THOR_ROOT", "data/hssd20S")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "/tmp/hsc_a_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "/tmp/hsc_q_"))
SC = os.environ.get("SCORES", "/tmp/t1_scores_hsc.jsonl")

def auc(y, x):
    y = np.asarray(y, bool); x = np.asarray(x, float)
    if y.all() or (~y).all(): return float("nan")
    r = x.argsort().argsort().astype(float)
    # 동점 평균 순위
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(len(x))
        for u in np.where(cnt > 1)[0]:
            m = inv == u; ranks[m] = ranks[m].mean()
        r = ranks
    return float((r[y].sum() - y.sum() * (y.sum() - 1) / 2) / (y.sum() * (~y).sum()))

recs = [json.loads(l) for l in open(SC)]
cache = {}
rows = []   # (house, oid, truth, near, s_ab, s_ac, QS, STx, TS, OWL)
for rc in recs:
    hn, oid = rc["house"], rc["oid"]
    if hn not in cache:
        hd = [d for d in glob.glob(ROOT + "/house_*") if os.path.basename(os.path.realpath(d)) == hn]
        if not hd: continue
        za = np.load(A3P + hn + ".npz", allow_pickle=True); zq = np.load(QCP + hn + ".npz", allow_pickle=True)
        g = json.load(open(hd[0] + "/gt.json"))
        cache[hn] = (za, zq, g, {m["t"]: m for m in g["live"]}, {m["oid"]: m["t"] for m in g["moves"]})
    za, zq, g, live, mvt = cache[hn]
    S, ts, vocab = za["s"], za["ts"], list(za["vocab"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    if oid not in QT or oid not in mvt: continue
    j = QT.index(oid); ti = vocab.index(g["gt0"][oid]["type"]); t0 = mvt[oid]
    for i, s_ab, s_ac in rc["scored"]:
        t = int(ts[i]); m = live.get(t, {})
        truth = t > t0 and oid in (m.get("vis") or [])
        near = truth and (m.get("dist") or {}).get(oid, 99) < 5
        rows.append((hn, oid, truth, near, s_ab, s_ac, float(QS[i, j]), float(STx[i, j]),
                     float(QS[i, j] + STx[i, j]), float(S[i, ti])))

R = np.array([r[2:] for r in rows], float); keys = [(r[0], r[1]) for r in rows]
y, near = R[:, 0] > 0, R[:, 1] > 0
names = ["s_ab(VLM 2AFC)", "s_ac(VLM 거부권)", "QS(exemplar 코사인)", "STx(글자질의)", "TS=QS+STx", "OWL 검출점수"]
print("크롭 %d · 진짜(이동후 목격) %d · 그중 <5m %d · 타겟 %d" % (len(rows), y.sum(), near.sum(), len(set(keys))))
# 타겟내 z-정규화 (문턱을 타겟마다 다르게 둘 수 있을 때의 분리력)
Z = R[:, 2:].copy()
for k in set(keys):
    m = np.array([kk == k for kk in keys])
    if m.sum() > 1:
        Z[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-9)
print("%-22s %-9s %-9s %-11s %-11s" % ("점수", "풀AUC", "z-AUC", "수용@기각.95", "<5m수용"))
for c, nm in enumerate(names):
    x = R[:, 2 + c]; th = np.quantile(x[~y], 0.95)
    print("%-22s %-9.3f %-9.3f %-11.3f %-11.3f" % (nm, auc(y, x), auc(y, Z[:, c]),
          float((x[y] >= th).mean()), float((x[near] >= th).mean()) if near.any() else float("nan")))
# 결합: VLM 과 exemplar 의 z 합
xz = Z[:, 0] + Z[:, 2]; th = np.quantile(xz[~y], 0.95)
print("%-22s %-9.3f %-9s %-11.3f" % ("z(s_ab)+z(QS) 결합", auc(y, xz), "—", float((xz[y] >= th).mean())))
