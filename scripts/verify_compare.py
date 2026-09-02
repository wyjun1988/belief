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

# ── 크롭 단위 진실: "프레임에 보임"(GT) ≠ "OWL 박스가 그 물체를 잡음" ──
# 진짜 프레임이라도 박스가 다른 데 있으면 크롭엔 물체가 없다 → 검증기가 우연 수준인 게 당연.
R_PX = float(os.environ.get("HIT_PX", "60"))
hit = np.zeros(len(rows), bool); dist_t = np.full(len(rows), np.nan)
k_ = 0
for rc in recs:
    hn, oid = rc["house"], rc["oid"]
    if hn not in cache: continue
    za, zq, g, live, mvt = cache[hn]
    if oid not in list(zq["tg"]) or oid not in mvt: continue
    vocab = list(za["vocab"]); ti = vocab.index(g["gt0"][oid]["type"])
    BX = za["bx"] if "bx" in za.files else None
    ts, P_, pw, ph = za["ts"], za["p"], int(za["pw"]), int(za["ph"])
    def _bc(i):   # 검출 중심 px — 박스 있으면 박스, 없으면 argmax 패치 중심 (파이프라인과 동일 후퇴)
        if BX is not None: return float(BX[i, ti][0]) * 768, float(BX[i, ti][1]) * 768
        return (int(P_[i, ti]) % pw + .5) / pw * 768, (int(P_[i, ti]) // pw + .5) / ph * 768
    for i, s_ab, s_ac in rc["scored"]:
        t = int(ts[i]); m = live.get(t, {})
        c = (m.get("ctr") or {}).get(oid)
        if c:
            bcx, bcy = _bc(i)
            hit[k_] = np.hypot(bcx - c[0], bcy - c[1]) <= R_PX
            dist_t[k_] = (m.get("dist") or {}).get(oid, np.nan)
        k_ += 1
print("박스 캐시:", "있음" if any("bx" in v[0].files for v in cache.values()) else "없음 → 패치 중심으로 적중 판정")
post = y
print()
print("진짜(이동후 목격) %d 중 OWL 박스가 물체 %.0fpx 안에 있음: %d (%.2f)  |  <5m: %d/%d  5m+: %d/%d"
      % (post.sum(), R_PX, (post & hit).sum(), (post & hit).sum() / max(post.sum(), 1),
         (post & hit & near).sum(), near.sum(), (post & hit & ~near).sum(), (post & ~near).sum()))
# 박스가 물체를 잡은 크롭(hit) vs 순수 오검출(비진짜) 로 다시 AUC — 검증기 본연의 분리력
neg = ~y
for c, nm in enumerate(names[:3]):
    x = R[:, 2 + c]; yy = np.concatenate([np.ones((post & hit).sum(), bool), np.zeros(neg.sum(), bool)])
    xx = np.concatenate([x[post & hit], x[neg]]); th = np.quantile(x[neg], 0.95)
    print("  %-22s AUC(박스적중 진짜 vs 오검출) %.3f · 수용@기각.95 %.3f (n진짜=%d)"
          % (nm, auc(yy, xx), float((x[post & hit] >= th).mean()) if (post & hit).any() else float("nan"), (post & hit).sum()))

# ── 상류: 이동 후 보인 **모든** 프레임에서 OWL 이 물체를 검출했는가 (후보 목록과 무관) ──
TH_OWL = float(os.environ.get("TH", "0.12"))
rec_b = {"<2m": [0, 0], "2-5m": [0, 0], "5m+": [0, 0]}
for hn, (za, zq, g, live, mvt) in cache.items():
    S, ts, vocab = za["s"], za["ts"], list(za["vocab"]); BX = za["bx"] if "bx" in za.files else None
    for oid, t0 in mvt.items():
        if oid not in g["gt0"] or g["gt0"][oid]["type"] not in vocab: continue
        ti = vocab.index(g["gt0"][oid]["type"])
        for i in range(len(ts)):
            t = int(ts[i]); m = live.get(t)
            if not m or t <= t0 or oid not in (m.get("vis") or []): continue
            d = (m.get("dist") or {}).get(oid, 99); c = (m.get("ctr") or {}).get(oid)
            b = "<2m" if d < 2 else "2-5m" if d < 5 else "5m+"
            rec_b[b][1] += 1
            if c is not None and S[i, ti] >= TH_OWL:
                if BX is not None: bcx, bcy = float(BX[i, ti][0]) * 768, float(BX[i, ti][1]) * 768
                else:
                    P_, pw, ph = za["p"], int(za["pw"]), int(za["ph"])
                    bcx, bcy = (int(P_[i, ti]) % pw + .5) / pw * 768, (int(P_[i, ti]) // pw + .5) / ph * 768
                if np.hypot(bcx - c[0], bcy - c[1]) <= R_PX: rec_b[b][0] += 1
print("OWL 검출 회수율 (이동 후 보인 전 프레임, S≥%.2f & 박스 %.0fpx): %s"
      % (TH_OWL, R_PX, " · ".join("%s %d/%d=%.2f" % (k, v[0], v[1], v[0] / max(v[1], 1)) for k, v in rec_b.items())))
