#!/usr/bin/env python3
"""기하 투영 국소화 — 앵커로 yaw 역산 → 타겟을 지도에 투영 → 방 폴리곤 판정.

    THOR_ROOT=data/thor7_t7view FRAME_W=768 python scripts/exp_georoom.py

§105 의 두 손실(방 안 이웃방 오답 95% · 문 너머 0.50)이 전부 "프레임 안 앵커로
방을 추론"하는 구조 탓 — 타겟의 위치 자체를 재면 둘 다 사라지는지 상한을 잰다.
GT(anch ctr·static pos·apos·dist)만 쓰므로 카메라 yaw 미기록(thor7)이어도 된다:
프레임의 정적 앵커(화면 x + 지도 좌표)로 yaw 를 역산한다.

내장 자기검증 (좌표 규약 오류는 여기서 터진다 — newsim 투영 교훈):
  ① apos→폴리곤 방 == live room GT 일치율 (폴리곤·좌표계 검증)
  ② yaw leave-one-out 픽셀 잔차 중앙값 (투영 규약 검증, 수십 px 이내여야)
"""
import glob, json, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
W = int(os.environ.get("FRAME_W", "768"))
F = W / 2.0                              # fov 90° → f = (W/2)/tan(45°)

def in_poly(p, poly):
    x, z = p; n = len(poly); c = False
    for i in range(n):
        x1, z1 = poly[i]; x2, z2 = poly[(i + 1) % n]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1:
            c = not c
    return c

def room_at(p, polys):
    for r, pl in polys.items():
        if in_poly(p, pl): return r
    best = (1e9, None)
    for r, pl in polys.items():
        d = min((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2 for v in pl)
        if d < best[0]: best = (d, r)
    return best[1]

def bearing(dx, dz): return np.degrees(np.arctan2(dx, dz))     # THOR yaw 규약: 0=+z, 90=+x
def pix_bear(cx): return np.degrees(np.arctan((cx - W / 2.0) / F))

agree_room = [0, 0]; loo_res = []
n_t = 0; hit_g = {"in": [0, 0], "out": [0, 0]}; hit_o = {"in": [0, 0], "out": [0, 0]}
nocov = 0
for hd in sorted(glob.glob(ROOT + "/house_*")):
    g = json.load(open(hd + "/gt.json"))
    sm = g["scene_meta"]; polys = sm["polys"]
    apos_st = {k: v["pos"] for k, v in sm["static"].items() if v.get("pos")}
    if not apos_st:
        print("static pos 없음(구판 생성) — 건너뜀:", os.path.basename(hd)); continue
    live = {m["t"]: m for m in g["live"]}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    mvs = {}
    for m in g["moves"]: mvs[m["oid"]] = m
    for oid, mv in mvs.items():
        v0 = g["gt0"].get(oid)
        if not v0 or not v0["room"] or cnt[v0["type"]] > 1: continue
        t0, tgt = mv["t"], mv["to"]
        sights = [m for m in g["live"]
                  if m["t"] > t0 and oid in (m.get("vis") or []) and (m.get("ctr") or {}).get(oid)]
        sights = sorted(sights, key=lambda m: -m["t"])[:3]
        if not sights: continue
        votes = []; obs_in = 0
        for m in sights:
            ap = m["apos"]
            # ① 자기검증: 폴리곤 좌표계
            agree_room[room_at(ap, polys) == m["room"]] += 1
            if m["room"] == tgt: obs_in += 1
            anc = [(a, c) for a, c in (m.get("anch") or {}).items()
                   if c and a in apos_st]
            if not anc: continue
            ys = []
            for a, c in anc:
                th = bearing(apos_st[a][0] - ap[0], apos_st[a][1] - ap[1])
                ys.append(np.radians(th - pix_bear(c[0])))
            yaw = np.degrees(np.arctan2(np.mean(np.sin(ys)), np.mean(np.cos(ys))))
            # ② 자기검증: LOO 픽셀 잔차 (앵커 2개 이상일 때)
            if len(anc) >= 2:
                for k, (a, c) in enumerate(anc):
                    ys2 = [y for j, y in enumerate(ys) if j != k]
                    y2 = np.degrees(np.arctan2(np.mean(np.sin(ys2)), np.mean(np.cos(ys2))))
                    th = bearing(apos_st[a][0] - ap[0], apos_st[a][1] - ap[1])
                    db = (th - y2 + 180) % 360 - 180
                    px = np.tan(np.radians(np.clip(db, -80, 80))) * F + W / 2.0
                    loo_res.append(abs(px - c[0]))
            d = (m.get("dist") or {}).get(oid)
            if d is None: continue
            b = yaw + pix_bear(m["ctr"][oid][0])
            pt = [ap[0] + d * np.sin(np.radians(b)), ap[1] + d * np.cos(np.radians(b))]
            votes.append(room_at(pt, polys))
        if not votes: nocov += 1; continue
        n_t += 1
        key = "in" if obs_in == len(sights) else "out"
        pred = Counter(votes).most_common(1)[0][0]
        hit_g[key][pred == tgt] += 1
        obs = Counter(m["room"] for m in sights).most_common(1)[0][0]   # 관측자 방 답변 기준선
        hit_o[key][obs == tgt] += 1

print("자기검증① apos→방 일치 %.3f (%d)  |  ② yaw LOO 잔차 중앙값 %.1fpx (n=%d)"
      % (agree_room[1] / max(sum(agree_room), 1), sum(agree_room),
         float(np.median(loo_res)) if loo_res else -1, len(loo_res)))
print("타겟 %d (yaw 미복구 %d)" % (n_t, nocov))
for key, lab in (("in", "전부 방 안"), ("out", "문 너머 포함")):
    tot = sum(hit_g[key])
    if tot:
        print("  %-10s n=%-4d  투영 %.3f  vs 관측자방 기준선 %.3f"
              % (lab, tot, hit_g[key][1] / tot, hit_o[key][1] / tot))
