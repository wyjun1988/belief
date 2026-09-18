#!/usr/bin/env python3
"""9-50 요인 실험 판독 — 각 판이 기준에서 **한 가지만** 바꿨으므로 효과를 그대로 뺄셈한다 (2026-09-19).

  python scripts/read_9_50.py ~/khcache/h100_950/out_9_50
읽는 것: 각 `<판>.log` 의 마지막 `EVALGRP[final]` 줄 (갈래별 AUC). 합산 `EVAL` 은 갈래 비율에 끌려다녀 쓰지 않는다.
"""
import os, re, sys, glob, collections

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/khcache/h100_950/out_9_50")
WHAT = {  # 판 이름 → 기준(A0_base)에서 바꾼 것 하나
    "A0_base": "— (기준: v2+og+c2 · narrow · r16)",
    "A1_r32": "어댑터 용량 r16 → 32",
    "A2_hn": "+ 시뮬 교차집 어려운음성",
    "A3_real": "★ + 실사 IT3DEgo",
    "A4_vision": "narrow → full (비전 타워)",
    "A5_lr3": "학습률 → 3e-5",
    "P0_presence": "과제 = 부재 단독",
    "C1_real_r32": "[조합] 실사 + r32",
    "C2_multi": "[조합] 채택+부재 한 어댑터",
    "C3_multi_r32": "[조합] 합본 + r32",
    "C4_real_vis": "[조합] 실사 + 비전",
}
rows = {}
for p in sorted(glob.glob(os.path.join(D, "*.log"))):
    nm = os.path.basename(p)[:-4]
    last = None; done = None; oom = False
    for l in open(p, errors="ignore"):
        if l.startswith("EVALGRP[final]"): last = l.strip()
        elif "LORA_TRAIN_DONE" in l: done = l.strip()
        elif "OutOfMemory" in l or "Traceback" in l: oom = True
    g = {}
    if last:
        for k, v, n in re.findall(r"(\w+) ([0-9.]+)\(n=(\d+)\)", last): g[k] = (float(v), int(n))
    rows[nm] = dict(grp=g, done=bool(done), bad=oom)

base = rows.get("A0_base", {}).get("grp") or {}
keys = ["adopt", "adoptreal", "presence"]
print(f"{'판':14s} {'바꾼 것':30s} " + " ".join(f"{k:>10s}" for k in keys) + "   상태")
for nm in list(WHAT) + [n for n in rows if n not in WHAT]:
    r = rows.get(nm)
    if not r: continue
    cells = []
    for k in keys:
        if k in r["grp"]:
            v = r["grp"][k][0]
            d = (v - base[k][0]) if (base.get(k) and nm != "A0_base") else None
            cells.append(f"{v:.3f}" + (f"{d:+.3f}" if d is not None else "     "))
        else: cells.append(" " * 10)
    st = "완료" if r["done"] else ("✗오류" if r["bad"] else "미완")
    print(f"{nm:14s} {WHAT.get(nm,''):30s} " + " ".join(f"{c:>10s}" for c in cells) + f"   {st}")
if base:
    print(f"\n기준 표본: " + " · ".join(f"{k} n={base[k][1]}" for k in keys if k in base))
print("\n※ 숫자는 고정 검증셋의 **갈래별 AUC**. 뒤의 부호는 기준 대비 차이.")
print("※ 최종 판정은 133채 실측(adopt_margin_*.jsonl → bench) 으로 한다 — val 차이는 표본이 작다.")
