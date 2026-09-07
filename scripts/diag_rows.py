#!/usr/bin/env python3
"""eval_online ROWS_OUT 덤프(물체 단위) 분해 — 경우×분기 정답률, ②·③ 실패 유형, 타입별 오답.
    python scripts/diag_rows.py ~/khcache/bench-h150/rows_D.jsonl"""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
print("물체 %d · 집 %d" % (len(rows), len({r["house"] for r in rows})))
by = collections.defaultdict(list)
for r in rows: by[(r["case"], r["branch"])].append(r["ok"])
print("%-26s %-5s %5s %s" % ("경우", "분기", "n", "정답"))
for (c, b), v in sorted(by.items()): print("%-26s %-5s %5d %.3f" % (c, b, len(v), sum(v) / len(v)))
for case in ("②재촬영", "③확인기회O"):
    sel = [r for r in rows if r["case"] == case]
    if not sel: continue
    print("\n== %s (n=%d) ==" % (case, len(sel)))
    print("  답=기록(그대로) %d · 답≠기록(갱신) %d · 정답 %d" % (sum(r["ans"] == r["record"] for r in sel), sum(r["ans"] != r["record"] for r in sel), sum(r["ok"] for r in sel)))
    print("  오답 예:", [(r["house"][-4:], r["type"], "기록", r["record"], "답", r["ans"], "정답", r["tgt"]) for r in sel if not r["ok"]][:8])
bad = collections.Counter(r["type"] for r in rows if not r["ok"]); print("\n오답 타입 상위:", bad.most_common(8))
