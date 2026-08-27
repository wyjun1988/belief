#!/usr/bin/env python3
"""부재 축 2건 — ① 변화점 탐지(온라인 격차 회수) ② 최근 부재 확인 능력(belief 인터페이스).

    THOR_ROOT=data/thor4 python scripts/eval_absence2.py

① 앞/뒤 1/3 고정 분할 대신 **방문 에피소드 경계마다** 하락을 재고 최대를 취한다.
   이동 시각을 모르는 온라인 조건에서 능력치(0.78)에 얼마나 접근하는가.
② (물체, 방) 쌍 단위 "R 에 없음" 주장 — belief 모델에 넘길 배제 목록의 품질.
   최근(마지막 1/4) 방문했고 자리가 보인 방에 대해서만 주장한다.
"""
import json, glob, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/h4/cache/a3_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/h4/cache/qc_"))

cp_mv, cp_st = [], []      # ① 변화점 신호
old_mv, old_st = [], []    # 기존 1/3 분할
pairs = []                 # ② (물체,방) 부재 주장
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, ts = za["s"], za["ts"]; vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json")); sm = g["scene_meta"]
    live = {m["t"]: m for m in g["live"]}
    rids = sorted(g["room_types"])
    arm = np.array([live[t]["room"] for t in ts])
    AS = S[:, nT:]
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    Tend = ts.max()
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]
        TS = QS[:, j] + STx[:, j]
        med = float(np.median(TS))

        def gated(idx):
            """자리 서명 게이팅 — 초기 상위 프레임의 정적 벡터와 닮은 프레임만."""
            if len(idx) < 9: return None, None
            e = idx[:len(idx)//3]
            k2 = max(3, len(e)//3)
            sig = AS[e[np.argsort(-TS[e])[:k2]]].mean(0)
            sig /= (np.linalg.norm(sig) + 1e-9)
            pv = AS[idx] @ sig / (np.linalg.norm(AS[idx], axis=1) + 1e-9)
            thp = np.quantile(pv[:len(idx)//3], .7)
            gi = idx[pv >= thp]
            return (gi, sig) if len(gi) >= 6 else (None, None)

        # ── ① 변화점: 기록 방 방문 에피소드 경계 스캔 ──
        inr = np.where(arm == v0["room"])[0]
        gi, _ = gated(inr)
        if gi is not None:
            visits = [[gi[0]]]
            for i in gi[1:]:
                if ts[i] - ts[visits[-1][-1]] <= 120: visits[-1].append(i)
                else: visits.append([i])
            drop_old = float(np.quantile(TS[gi[:len(gi)//3]], .9)
                             - np.quantile(TS[gi[-len(gi)//3:]], .9))
            best = -9
            if len(visits) >= 2:
                for k in range(1, len(visits)):
                    a = np.concatenate(visits[:k]); b = np.concatenate(visits[k:])
                    if len(a) >= 3 and len(b) >= 3:
                        best = max(best, float(np.quantile(TS[a], .9)
                                               - np.quantile(TS[b], .9)))
            if best > -9:
                (cp_mv if mv else cp_st).append(best)
                (old_mv if mv else old_st).append(drop_old)

        # ── ② 최근 부재 확인: 마지막 1/4 에 방문한 방마다 주장 ──
        for R in rids:
            ir = np.where(arm == R)[0]
            rec = [i for i in ir if ts[i] > Tend * 0.75]
            if len(rec) < 5: continue
            gi2, sig2 = gated(ir)
            if gi2 is None: continue
            grec = [i for i in rec if i in set(gi2.tolist())]
            if len(grec) < 3: continue
            score = float(np.quantile(TS[grec], .9)) - med   # 자기보정
            pairs.append(dict(absent_gt=(R != tgt), s=-score))  # 낮을수록 부재

def auc(a, b):
    a, b = np.array(a), np.array(b)
    return (a[:, None] > b[None, :]).mean() + .5*(a[:, None] == b[None, :]).mean()

print("=== ① 변화점 탐지 (이동 %d · 제자리 %d) ===" % (len(cp_mv), len(cp_st)))
print("  기존 1/3 분할  AUC **%.3f**" % auc(old_mv, old_st))
print("  변화점 최대    AUC **%.3f**   (능력치 상한 0.781)" % auc(cp_mv, cp_st))
ab = [p for p in pairs if p["absent_gt"]]; pr_ = [p for p in pairs if not p["absent_gt"]]
print("\n=== ② 최근 부재 확인 — (물체,방) 쌍 %d (부재 %d · 존재 %d) ===" % (len(pairs), len(ab), len(pr_)))
sa = np.array([p["s"] for p in ab]); sp = np.array([p["s"] for p in pr_])
print("  분리 AUC **%.3f**" % ((sa[:, None] > sp[None, :]).mean()))
for q in (.5, .7, .9):
    th = np.quantile(sp, q)
    prec = (sa >= th).sum() / max((sa >= th).sum() + (sp >= th).sum(), 1)
    print("  존재오인 %.0f%% 문턱: '없음' 주장 정밀 **%.3f** · 재현 %.3f · 물체당 주장 %.1f방"
          % (100*(1-q), prec, (sa >= th).mean(),
             ((sa >= th).sum() + (sp >= th).sum()) / max(len(set([id(p) for p in pairs])) and len(pairs)/4.2, 1)))
