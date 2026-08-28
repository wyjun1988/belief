#!/usr/bin/env python3
"""검증기 문턱 스윕 — 캘리브레이션 쌍 점수(exp_vlm_verify3 산출 jsonl)에서
s_ab AUC 와 기각률별 운용점을 계산한다. §89 원칙: 문턱은 다른 도메인에서
이식하지 않고 **그 도메인의 쌍 점수로 현지 산출**한다.

    python scripts/rtx7_sweep.py res768_scores.jsonl [--pick 0.99]

마지막 줄에 선택 문턱 숫자만 출력한다 → 쉘에서 TH=$(... | tail -1) 로 받는다.
"""
import json, sys
import numpy as np

f = sys.argv[1]
pick = float(sys.argv[sys.argv.index("--pick") + 1]) if "--pick" in sys.argv else 0.99
rec = [json.loads(l) for l in open(f) if l.strip()]
def _auc(ch):
    p_ = np.array([r[ch] for r in rec if r["truth"] and ch in r])
    n_ = np.array([r[ch] for r in rec if not r["truth"] and ch in r])
    if not (len(p_) and len(n_)): return None
    sc_ = np.concatenate([p_, n_]); o_ = np.argsort(sc_, kind="mergesort")
    rk_ = np.empty(len(sc_)); rk_[o_] = np.arange(1, len(sc_) + 1)
    return (rk_[:len(p_)].sum() - len(p_) * (len(p_) + 1) / 2) / (len(p_) * len(n_))

pos = np.array([r["s_ab"] for r in rec if r["truth"]])
neg = np.array([r["s_ab"] for r in rec if not r["truth"]])
assert len(pos) and len(neg), "양·음성 쌍이 모두 있어야 한다"
# 채널 3종 전부 보고 — "0.944 가 어느 채널이었나" 류 혼선을 즉시 드러낸다
print("채널별 AUC: " + " · ".join("%s %.3f" % (c, _auc(c))
      for c in ("s_yn", "s_ab", "s_ac") if _auc(c) is not None))

# 거리 버킷별 (dist 필드가 있을 때) — "원거리 가시가 원인인가" 판별
if any("dist" in r for r in rec):
    for lo, hi, tag in ((0, 2, "<2m"), (2, 5, "2-5m"), (5, 99, "5m+")):
        sub = [r for r in rec if lo <= r.get("dist", -1) < hi]
        p_ = np.array([r["s_ab"] for r in sub if r["truth"]])
        n_ = np.array([r["s_ab"] for r in sub if not r["truth"]])
        if len(p_) >= 10 and len(n_) >= 10:
            sc_ = np.concatenate([p_, n_]); o_ = np.argsort(sc_, kind="mergesort")
            rk_ = np.empty(len(sc_)); rk_[o_] = np.arange(1, len(sc_) + 1)
            a_ = (rk_[:len(p_)].sum() - len(p_) * (len(p_) + 1) / 2) / (len(p_) * len(n_))
            print("  거리 %-5s AUC %.3f (양 %d/음 %d)" % (tag, a_, len(p_), len(n_)))

sc = np.concatenate([pos, neg])
o = np.argsort(sc, kind="mergesort")
rk = np.empty(len(sc)); rk[o] = np.arange(1, len(sc) + 1)
auc = (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print("쌍 %d (양 %d / 음 %d) · s_ab AUC %.3f" % (len(rec), len(pos), len(neg), auc))
for q in (0.95, 0.98, 0.99):
    th = float(np.quantile(neg, q))
    print("  기각 %.2f → 문턱 %+.3f · 진짜 수용 %.3f" % (q, th, float((pos >= th).mean())))
print(round(float(np.quantile(neg, pick)), 3))
