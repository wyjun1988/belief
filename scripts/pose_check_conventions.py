#!/usr/bin/env python3
"""PnP 포즈(POSE_JSONL) vs 데이터셋 gt.json 라이브 포즈 — 규약 가설 검사 (2026-09-11, OG 24채 포즈 대조 0.56/0.38 진단용).
  python scripts/pose_check_conventions.py <THOR_ROOT> <pose_all.jsonl>
출력: 집별 오차 · 오차 분포(양봉?) · 위치 x/z 반전·교환, yaw 부호/오프셋(±90·180·180−yaw) 조합별 ≤0.5 m·≤10° 비율 → 가장 맞는 변환."""
import json, math, glob, os, sys, collections, numpy as np
root, pj = sys.argv[1], sys.argv[2]; rows = [json.loads(l) for l in open(pj)]; byh = collections.defaultdict(list)
for r in rows: byh[r["house"]].append(r)
P = []; G = []; H = []
for hd in sorted(glob.glob(os.path.join(root, "house_*"))):
    hn = os.path.basename(hd); g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    for r in byh.get(hn, []):
        m = live.get(int(r["t"]))
        if m and m.get("apos") and m.get("yaw") is not None: P.append([r["apos"][0], r["apos"][1], r["yaw"]]); G.append([m["apos"][0], m["apos"][1], m["yaw"]]); H.append(hn)
P = np.array(P, float); G = np.array(G, float); H = np.array(H); print("프레임 %d · 집 %d" % (len(P), len(set(H))))
def stats(px, pz, py):
    d = np.hypot(px - G[:, 0], pz - G[:, 1]); a = np.abs((py - G[:, 2] + 180) % 360 - 180); return d, a
d, a = stats(P[:, 0], P[:, 1], P[:, 2])
print("원본: 위치 ≤0.5 m %.2f (중앙 %.2f m) · yaw ≤10° %.2f (중앙 %.0f°)" % ((d <= .5).mean(), np.median(d), (a <= 10).mean(), np.median(a)))
print("yaw 오차 히스토그램(30° 구간):", np.histogram(a, bins=[0, 10, 30, 60, 90, 120, 150, 181])[0].tolist(), "← 90°·180° 봉우리면 규약")
print("집별 (위치 ≤0.5 · yaw ≤10 · n):"); 
for hn in sorted(set(H)): m = H == hn; print("  %s %.2f %.2f %d" % (hn, (d[m] <= .5).mean(), (a[m] <= 10).mean(), m.sum()))
best = []
for nm, (px, pz) in {"원본": (P[:, 0], P[:, 1]), "x반전": (-P[:, 0], P[:, 1]), "z반전": (P[:, 0], -P[:, 1]), "xz반전": (-P[:, 0], -P[:, 1]), "교환": (P[:, 1], P[:, 0]), "교환+x반전": (-P[:, 1], P[:, 0]), "교환+z반전": (P[:, 1], -P[:, 0])}.items():
    for ynm, py in {"yaw": P[:, 2], "-yaw": -P[:, 2], "yaw+90": P[:, 2] + 90, "yaw-90": P[:, 2] - 90, "yaw+180": P[:, 2] + 180, "180-yaw": 180 - P[:, 2], "90-yaw": 90 - P[:, 2], "-90-yaw": -90 - P[:, 2]}.items():
        dd, aa = stats(px, pz, py); best.append(((dd <= .5).mean(), (aa <= 10).mean(), nm, ynm))
best.sort(reverse=True); print("변환 상위 5 (위치≤0.5, yaw≤10, 위치변환, yaw변환):"); [print("  %.2f %.2f %s %s" % b) for b in best[:5]]
# 위치는 맞고 yaw 만 틀린 프레임 / 둘 다 틀린 프레임 비율
print("위치 OK·yaw NG %.2f · 위치 NG·yaw OK %.2f · 둘 다 NG %.2f" % (((d <= .5) & (a > 10)).mean(), ((d > .5) & (a <= 10)).mean(), ((d > .5) & (a > 10)).mean()))
