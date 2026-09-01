#!/usr/bin/env python3
"""기하 투영 사다리 ②' — 검출 yaw 의 교정판: 가설 투표 + 타입 앵커 합류 + 기권.

    THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ AX_PREFIX=/tmp/t7_x_ \\
      FRAME_W=768 python scripts/exp_georoom3.py

②(원형평균, exemplar 앵커만)는 yaw 중앙오차 42.8°·커버리지 0.29·투영 ~0.5 로
꺾였다. 원인: argmax 패치는 앵커가 화면에 없으면 아무 데나 찍히고, 앵커가
한두 개면 오검출 하나가 평균을 끌고 간다. 교정 3종:
  1. **가설 투표** — 앵커마다 yaw 가설 1표, ±12° 합의 군집만 채택 (오검출은 흩어짐)
  2. **타입 앵커 합류** — a3 의 정적 타입 검출(소파·침대…)을 인스턴스 후보마다
     다중 가설로 (커버리지↑). 인스턴스 4개 초과 타입은 가설 홍수라 제외
  3. **기권** — 합의 가중 < SUPP 면 그 프레임은 투영하지 않는다. 목격 풀을
     3→8장으로 넓혀 기권을 흡수
교차검증(GT ctr yaw 와 각도차)은 ②와 동일하게 내장.
"""
import glob, json, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
A3P = os.environ.get("A3_PREFIX", "/tmp/t7_a_")
AXP = os.environ.get("AX_PREFIX", "/tmp/t7_x_")
W = int(os.environ.get("FRAME_W", "768"))
TH_ANCH = float(os.environ.get("TH_ANCH", "0.15"))
TH_TYPE = float(os.environ.get("TH_TYPE", "0.15"))
SUPP = float(os.environ.get("SUPP", "2.0"))
WIN = float(os.environ.get("WIN", "12.0"))
POOL = int(os.environ.get("POOL", "8"))
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

def vote_yaw(hyp, supp=SUPP, win=WIN):
    """hyp = [(yaw 가설 °, 가중)] → (yaw|None, 합의 가중)"""
    if not hyp: return None, 0.0
    best = (0.0, None)
    for y0, _ in hyp:
        w = sum(w2 for y2, w2 in hyp if abs((y2 - y0 + 180) % 360 - 180) <= win)
        if w > best[0]: best = (w, y0)
    if best[0] < supp: return None, best[0]
    ys = [(np.radians(y), w) for y, w in hyp if abs((y - best[1] + 180) % 360 - 180) <= win]
    sw = sum(w for _, w in ys)
    return float(np.degrees(np.arctan2(sum(np.sin(y) * w for y, w in ys) / sw,
                                       sum(np.cos(y) * w for y, w in ys) / sw))), best[0]

def cmean_deg(pairs):
    ys = [np.radians(th - pix_bear(cx)) for cx, th in pairs]
    return float(np.degrees(np.arctan2(np.mean(np.sin(ys)), np.mean(np.cos(ys)))))

n_t = 0; hit = {"in": [0, 0], "out": [0, 0]}
nocov = 0; ddiff = []; cover = [0, 0]; used_fr = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fx = A3P + hn + ".npz", AXP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fx)): continue
    g = json.load(open(hd + "/gt.json"))
    sm = g["scene_meta"]; polys = sm["polys"]
    stpos = {k: ([v["pos"][0], v["pos"][2]] if len(v["pos"]) == 3 else v["pos"]) for k, v in sm["static"].items() if v.get("pos")}
    if not stpos:
        print("static pos 없음 — 건너뜀:", hn); continue
    bytype = {}
    for k, v in sm["static"].items():
        if v.get("pos"): bytype.setdefault(v["type"], []).append(v["pos"])
    za = np.load(fa, allow_pickle=True)
    ts, vocab, nT = za["ts"], list(za["vocab"]), int(za["nT"])
    S, P, ph, pw = za["s"], za["p"], int(za["ph"]), int(za["pw"])
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
    def px_of(pidx): return (pidx % pw + .5) / pw * W
    for oid, mv in mvs.items():
        v0 = g["gt0"].get(oid)
        if not v0 or not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        t0, tgt = mv["t"], mv["to"]
        ti = vocab.index(v0["type"])
        rows_i = [i for i, t in enumerate(tsl)
                  if t > t0 and oid in (live.get(t, {}).get("vis") or [])]
        rows_i = sorted(rows_i, key=lambda i: -tsl[i])[:POOL]
        if not rows_i: continue
        votes = []; obs_in = 0; nfr = 0
        for i in rows_i:
            if len(votes) >= 3: break
            m = live[tsl[i]]; ap = np.array(m["apos"])
            hyp = []
            for k in np.where(XSc[i] >= TH_ANCH)[0]:      # exemplar 앵커 (개체 식별)
                dp = apos_c[k] - ap
                hyp.append((bearing(dp[0], dp[1]) - pix_bear(px_of(XP[i, k])), 2.0))
            for c in range(nT, len(vocab)):               # 타입 앵커 (다중 가설)
                t_ = vocab[c]
                inst = bytype.get(t_, [])
                if not inst or len(inst) > 4 or S[i, c] < TH_TYPE: continue
                cx = px_of(P[i, c])
                for pos in inst:
                    hyp.append((bearing(pos[0] - ap[0], pos[1] - ap[1]) - pix_bear(cx),
                                1.0 / len(inst)))
            yaw, _w = vote_yaw(hyp)
            cover[yaw is not None] += 1
            if yaw is None: continue
            gta = [(c_[0], bearing(stpos[a][0] - ap[0], stpos[a][1] - ap[1]))
                   for a, c_ in (m.get("anch") or {}).items() if c_ and a in stpos]
            if gta:
                ddiff.append(abs((yaw - cmean_deg(gta) + 180) % 360 - 180))
            d = (m.get("dist") or {}).get(oid)
            if d is None: continue
            if m["room"] == tgt: obs_in += 1
            b = yaw + pix_bear(px_of(P[i, ti]))
            pt = [ap[0] + d * np.sin(np.radians(b)), ap[1] + d * np.cos(np.radians(b))]
            votes.append(room_at(pt, polys)); nfr += 1
        if not votes: nocov += 1; continue
        n_t += 1; used_fr.append(nfr)
        key = "in" if obs_in == len(votes) else "out"
        hit[key][Counter(votes).most_common(1)[0][0] == tgt] += 1

print("교차검증 yaw(투표) vs yaw(GT ctr) 각도차 중앙값 %.1f° (n=%d)"
      % (float(np.median(ddiff)) if ddiff else -1, len(ddiff)))
print("프레임 커버리지(기권 반영) %.2f · 타겟당 사용 프레임 %.1f"
      % (cover[1] / max(sum(cover), 1), float(np.mean(used_fr)) if used_fr else 0))
print("타겟 %d (투영 불가 %d)" % (n_t, nocov))
for key, lab in (("in", "전부 방 안"), ("out", "문 너머 포함")):
    tot = sum(hit[key])
    if tot:
        print("  %-10s n=%-4d  투영(투표 yaw) %.3f" % (lab, tot, hit[key][1] / tot))
