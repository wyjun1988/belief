#!/usr/bin/env python3
"""IT3DEgo 실사 자동 QA — 현재/직전 위치 구간 검색 채점. (아이맥, 캐시 재사용)

    python scripts/it3d_autoqa.py

GT: 물체마다 3D 센터의 **시간 구간**(3d_center_annot: t0 t1 oi x y z seg).
문항:
  now     영상 끝에서 "X 지금 어디?" — 검색 상위 프레임이 **마지막 구간**에 들면 정답
  before  구간 2개+ 물체 — 직전 사건 선택(v2 규칙)이 **직전 구간**에 들면 정답
기저: 후보 프레임 무작위 선택의 구간 적중률.
"""
import glob, json, os
import numpy as np
from collections import Counter
import sys
sys.path.insert(0, ".")
from scripts.it3d_absence import load_ann, base_label

now_ok = now_n = bef_ok = bef_n = 0
rnd_now = []; rnd_bef = []
for f in sorted(glob.glob("data/it3dego/cache_all/*.all.npz")):
    vn = os.path.basename(f)[:-len(".all.npz")]
    ad = os.path.join("data/it3dego/ann/annotations", vn)
    if not os.path.isdir(ad): continue
    z = np.load(f, allow_pickle=True)
    ts, S = z["ts"], z["owl"]
    labs, segs, box = load_ann(ad)
    words = [base_label(l) for l in labs]
    seg3 = {}
    for line in open(os.path.join(ad, "3d_center_annot.txt")):
        p = line.split()
        if len(p) < 7: continue
        t0, t1, oi = int(p[0]), int(p[1]), int(p[2])
        seg3.setdefault(oi, []).append((t0, t1))
    for oi, w in enumerate(words):
        if words.count(w) > 1: continue
        sgs = sorted(seg3.get(oi, []))
        if not sgs: continue
        sc = S[:, oi]
        # now: 마지막 구간
        lo, hi = sgs[-1]
        # 우리 표준: 문턱 후 최신 (점수순만 쓰면 옛 구간 목격이 상위 도배 — T1 문제)
        th0 = np.quantile(sc, 0.95)
        cand = np.where(sc >= th0)[0]
        top = sorted(cand, key=lambda i: -ts[i])[:3]
        hit = any(lo <= ts[i] <= hi for i in top)
        now_ok += hit; now_n += 1
        rnd_now.append(np.mean([(lo <= t <= hi) for t in ts]))
        # before: 직전 구간 (v2: 문턱 통과 사건 군집 → 직전 사건)
        if len(sgs) >= 2:
            plo, phi = sgs[-2]
            th = np.quantile(sc, 0.95)
            hits = sorted(np.where(sc >= th)[0], key=lambda i: ts[i])
            evs = []
            for i in hits:
                if evs and ts[i] - ts[evs[-1][-1]] <= 3e8: evs[-1].append(i)
                else: evs.append([i])
            evs = [e for e in evs if len(e) >= 2]
            if len(evs) >= 2:
                pick = sorted(evs[-2], key=lambda i: -sc[i])[:3]
                hitb = any(plo <= ts[i] <= phi for i in pick)
                bef_ok += hitb; bef_n += 1
                rnd_bef.append(np.mean([(plo <= t <= phi) for t in ts]))
print("=== IT3DEgo 실사 자동 QA (%d영상) ===" % len(glob.glob("data/it3dego/cache_all/*.all.npz")))
print("now(현재 구간 적중)   %d/%d = %.3f  · 무작위 기저 %.3f"
      % (now_ok, now_n, now_ok/max(now_n,1), np.mean(rnd_now)))
if bef_n:
    print("before(직전 구간)     %d/%d = %.3f  · 무작위 기저 %.3f"
          % (bef_ok, bef_n, bef_ok/bef_n, np.mean(rnd_bef)))
