#!/usr/bin/env python3
"""기하 투영 사다리 ② — yaw 를 GT 앵커가 아니라 **검출 앵커(ax 캐시)** 로 역산.

    THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ AX_PREFIX=/tmp/t7_x_ \\
      FRAME_W=768 python scripts/exp_georoom2.py

§106 사다리: ① 전부 GT 0.975~0.992 → ② 이 실험(yaw 검출·타겟 방위 OWL 패치·
거리만 GT) → ③ 거리 DA3 → ④ 앵커 3D initmap.
검출 재료: exemplar 앵커 점수(XSc>=TH_ANCH)의 패치 위치로 화면 x, scene_meta
좌표로 방위 → 원형평균(잔차 25° 초과 1회 탈락) yaw.
내장 교차검증: 같은 프레임에서 GT anch ctr 로 푼 yaw 와의 각도차 중앙값 —
캐시 인덱싱·규약 오류는 여기서 터진다.
"""
import glob, json, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
A3P = os.environ.get("A3_PREFIX", "/tmp/t7_a_")
AXP = os.environ.get("AX_PREFIX", "/tmp/t7_x_")
W = int(os.environ.get("FRAME_W", "768"))
TH_ANCH = float(os.environ.get("TH_ANCH", "0.15"))
F = W / 2.0

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

def bearing(dx, dz): return np.degrees(np.arctan2(dx, dz))
def pix_bear(cx): return np.degrees(np.arctan((cx - W / 2.0) / F))

def cmean(ys):
    return np.degrees(np.arctan2(np.mean(np.sin(ys)), np.mean(np.cos(ys))))

def solve_yaw(obs):
    """obs = [(화면x, 지도방위)] → yaw (원형평균 + 25° 초과 잔차 1회 탈락)"""
    ys = [np.radians(th - pix_bear(cx)) for cx, th in obs]
    y0 = cmean(ys)
    keep = [y for y in ys if abs((np.degrees(y) - y0 + 180) % 360 - 180) <= 25]
    return cmean(keep if keep else ys)

n_t = 0; hit = {"in": [0, 0], "out": [0, 0]}
nocov = 0; ddiff = []; cover = [0, 0]
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fx = A3P + hn + ".npz", AXP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fx)): continue
    g = json.load(open(hd + "/gt.json"))
    sm = g["scene_meta"]; polys = sm["polys"]
    stpos = {k: v["pos"] for k, v in sm["static"].items() if v.get("pos")}
    if not stpos:
        print("static pos 없음 — 건너뜀:", hn); continue
    za = np.load(fa, allow_pickle=True)
    ts, vocab, nT = za["ts"], list(za["vocab"]), int(za["nT"])
    P, ph, pw = za["p"], int(za["ph"]), int(za["pw"])
    ZX = np.load(fx, allow_pickle=True)
    an_ = list(ZX["anch"])
    cols = [k for k, a in enumerate(an_) if a in stpos]
    XS = ZX["s"][:, cols]
    XSc = XS - np.median(XS, axis=0, keepdims=True)
    XP = ZX["p"][:, cols]
    apos_c = [np.array(stpos[an_[k]]) for k in cols]
    live = {m["t"]: m for m in g["live"]}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    mvs = {}
    for m in g["moves"]: mvs[m["oid"]] = m
    tsl = [int(t) for t in ts]
    for oid, mv in mvs.items():
        v0 = g["gt0"].get(oid)
        if not v0 or not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        t0, tgt = mv["t"], mv["to"]
        ti = vocab.index(v0["type"])
        rows_i = [i for i, t in enumerate(tsl)
                  if t > t0 and oid in (live.get(t, {}).get("vis") or [])]
        rows_i = sorted(rows_i, key=lambda i: -tsl[i])[:3]
        if not rows_i: continue
        votes = []; obs_in = 0
        for i in rows_i:
            m = live[tsl[i]]; ap = m["apos"]
            if m["room"] == tgt: obs_in += 1
            on = np.where(XSc[i] >= TH_ANCH)[0]
            cover[len(on) > 0] += 1
            if not len(on): continue
            obs = []
            for k in on:
                cx = (XP[i, k] % pw + .5) / pw * W
                dp = apos_c[k] - np.array(ap)
                obs.append((cx, bearing(dp[0], dp[1])))
            yaw = solve_yaw(obs)
            # 교차검증: 같은 프레임 GT anch ctr 로 푼 yaw 와 비교
            gta = [(c[0], bearing(stpos[a][0] - ap[0], stpos[a][1] - ap[1]))
                   for a, c in (m.get("anch") or {}).items() if c and a in stpos]
            if gta:
                dd = abs((yaw - solve_yaw(gta) + 180) % 360 - 180)
                ddiff.append(dd)
            d = (m.get("dist") or {}).get(oid)
            if d is None: continue
            tx = (P[i, ti] % pw + .5) / pw * W
            b = yaw + pix_bear(tx)
            pt = [ap[0] + d * np.sin(np.radians(b)), ap[1] + d * np.cos(np.radians(b))]
            votes.append(room_at(pt, polys))
        if not votes: nocov += 1; continue
        n_t += 1
        key = "in" if obs_in == len(rows_i) else "out"
        hit[key][Counter(votes).most_common(1)[0][0] == tgt] += 1

print("교차검증 yaw(검출) vs yaw(GT ctr) 각도차 중앙값 %.1f° (n=%d) · 앵커 커버리지 %.2f"
      % (float(np.median(ddiff)) if ddiff else -1, len(ddiff),
         cover[1] / max(sum(cover), 1)))
print("타겟 %d (투영 불가 %d)" % (n_t, nocov))
for key, lab in (("in", "전부 방 안"), ("out", "문 너머 포함")):
    tot = sum(hit[key])
    if tot:
        print("  %-10s n=%-4d  투영(검출 yaw) %.3f" % (lab, tot, hit[key][1] / tot))
