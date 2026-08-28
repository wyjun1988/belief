#!/usr/bin/env python3
"""T1 실검 해부 — 걸은 크롭을 GT 로 삼분해: 이동후 진짜 / 이동전 진짜(낡음) / 오검출.

    THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ TH=3.261 \\
      SCORES=t1_scores_t7.jsonl python scripts/rtx7_diag.py

실검이 낮은 이유가 어디인지 가른다:
  A. 걸은 후보에 이동후 진짜가 아예 없음  → 후보 랭킹 문제 (검색 개선)
  B. 있는데 문턱을 못 넘음               → 문턱 과엄 / VLM 원거리 불능
  C. 넘는데 이동전 진짜가 먼저 수용됨     → 검증기는 정체만 봄 — 최신성 규칙 문제
"""
import glob, json, os
import numpy as np

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
A3P = os.environ.get("A3_PREFIX", "/tmp/t7_a_")
TH = float(os.environ.get("TH", "3.261"))
SC = os.environ.get("SCORES", "t1_scores_t7.jsonl")

recs = [json.loads(l) for l in open(SC) if l.strip()]
byh = {}
for r in recs: byh.setdefault(r["house"], []).append(r)

CLS = ("이동후진짜", "이동전진짜", "오검출")
walked = {c: [0, 0, 0] for c in CLS}          # [<2m, 2-5m, 5m+] — 오검출은 [0]에 합산
acc = {c: 0 for c in CLS}                     # 문턱 통과 수
n_t = 0
caseA = caseB = caseC = ok3 = 0               # 타겟 단위 판정
for hn, rs in sorted(byh.items()):
    fa = A3P + hn + ".npz"
    hd = os.path.join(ROOT, hn)
    if not (os.path.exists(fa) and os.path.exists(hd + "/gt.json")):
        print("건너뜀(캐시/GT 없음):", hn); continue
    ts = np.load(fa, allow_pickle=True)["ts"]
    g = json.load(open(hd + "/gt.json"))
    live = {m["t"]: m for m in g["live"]}
    mvs = {}
    for m in g["moves"]: mvs[m["oid"]] = m["t"]   # 마지막 이동 시각
    for r in rs:
        oid = r["oid"]; t0 = mvs.get(oid, -1)
        n_t += 1
        kinds = []
        for i, s in r["scored"]:
            t = int(ts[i]); m = live.get(t, {})
            true = oid in (m.get("vis") or [])
            d = (m.get("dist") or {}).get(oid, -1) if true else -1
            c = ("이동후진짜" if t > t0 else "이동전진짜") if true else "오검출"
            b = 0 if (not true or d < 2) else (1 if d < 5 else 2)
            walked[c][b] += 1
            passed = s >= TH
            if passed: acc[c] += 1
            kinds.append((c, passed, s))
        first3 = [k for k in kinds if k[1]][:3]   # 걷기 순서 = 최신순 → 정지-3 수용 집합
        if not any(k[0] == "이동후진짜" for k in kinds): caseA += 1
        elif not first3: caseB += 1
        elif sum(k[0] == "이동후진짜" for k in first3) < 2: caseC += 1
        else: ok3 += 1

print("타겟 %d (이동물체·타입단일)" % n_t)
print("\n걸은 크롭 구성 (거리는 진짜만):")
for c in CLS:
    w = walked[c]; tot = sum(w)
    dd = "  <2m %d · 2-5m %d · 5m+ %d" % tuple(w) if c != "오검출" else ""
    print("  %-8s %5d장 · 문턱통과 %4d (%.2f)%s"
          % (c, tot, acc[c], acc[c] / max(tot, 1), dd))
print("\n타겟 단위 판정 (수용 3장 기준):")
print("  A. 이동후 진짜가 후보에 없음(랭킹 문제)   %4d (%.2f)" % (caseA, caseA / max(n_t, 1)))
print("  B. 있는데 아무것도 문턱 못 넘음(과엄)     %4d (%.2f)" % (caseB, caseB / max(n_t, 1)))
print("  C. 수용 3장 과반이 낡음/오검출(오염)      %4d (%.2f)" % (caseC, caseC / max(n_t, 1)))
print("  OK 수용 과반이 이동후 진짜                %4d (%.2f)" % (ok3, ok3 / max(n_t, 1)))
