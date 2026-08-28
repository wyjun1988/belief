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
pos = np.array([r["s_ab"] for r in rec if r["truth"]])
neg = np.array([r["s_ab"] for r in rec if not r["truth"]])
assert len(pos) and len(neg), "양·음성 쌍이 모두 있어야 한다"

sc = np.concatenate([pos, neg])
o = np.argsort(sc, kind="mergesort")
rk = np.empty(len(sc)); rk[o] = np.arange(1, len(sc) + 1)
auc = (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print("쌍 %d (양 %d / 음 %d) · s_ab AUC %.3f" % (len(rec), len(pos), len(neg), auc))
for q in (0.95, 0.98, 0.99):
    th = float(np.quantile(neg, q))
    print("  기각 %.2f → 문턱 %+.3f · 진짜 수용 %.3f" % (q, th, float((pos >= th).mean())))
print(round(float(np.quantile(neg, pick)), 3))
