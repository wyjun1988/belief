#!/usr/bin/env python3
"""초기맵 군집의 **방 라벨을 구성점 투표로 다시 붙인다** (2026-09-19).

왜: `build_initmap` 은 군집 중심 한 점을 평면도에 넣어 방을 정한다(`room_pt`). 그런데 실측에서
  ① 오답 143건 중 **10건이 "자리는 맞는데 방 라벨만 틀린" 경우**였고, 그 군집 중심은 GT 방
  경계에서 **중앙 0.16 m** 떨어져 있었다 — 사실상 선 위다. 점 하나로 방을 정하면 깨진다.
방법: `initmap_raw.json` 의 투영점 중 군집 중심 반경 R 안의 것들을 모아 **검출 점수 가중 다수결**.
  표가 비면 원래 라벨을 유지한다. `--margin` 으로 원 라벨을 이길 최소 표차를 줄 수 있다.

  python scripts/initmap_room_vote.py --root data/hssd_v2 --in initmap_k8.json --out initmap_k8_rv.json
"""
import argparse, glob, json, math, os, collections

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/hssd_v2")
ap.add_argument("--in", dest="inp", default="initmap_k8.json")
ap.add_argument("--raw", default="initmap_raw.json")
ap.add_argument("--out", default="initmap_k8_rv.json")
ap.add_argument("--radius", type=float, default=1.0, help="군집 중심에서 이 반경 안의 투영점만 센다")
ap.add_argument("--margin", type=float, default=0.0, help="원 라벨을 이기려면 이만큼 더 받아야 한다(점수 비율)")
a = ap.parse_args()

def pip(pt, poly):
    x, z = pt; ins = False; n = len(poly)
    for k in range(n):
        x1, z1 = poly[k][0], poly[k][-1]; x2, z2 = poly[(k+1) % n][0], poly[(k+1) % n][-1]
        if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12)+x1: ins = not ins
    return ins

def room_of(pt, polys):
    for r, pl in polys.items():
        pls = pl if (pl and isinstance(pl[0][0], (list, tuple))) else [pl]
        if any(pip(pt, q) for q in pls): return r
    return None

st = collections.Counter()
for hd in sorted(glob.glob(os.path.join(a.root, "house_*"))):
    ip, rp = os.path.join(hd, a.inp), os.path.join(hd, a.raw)
    if not (os.path.exists(ip) and os.path.exists(rp)):
        st["초기맵/raw 없음"] += 1; continue
    g = json.load(open(os.path.join(hd, "gt.json")))
    polys = (g.get("scene_meta") or {}).get("polys") or {}
    if not polys: st["평면도 없음"] += 1; continue
    im = json.load(open(ip)); raw = json.load(open(rp))
    out = []
    for it in im:
        c = it.get("pos"); pts = raw.get(it["type"]) or []
        if not c or not pts: out.append(it); st["투표 불가"] += 1; continue
        vote = collections.Counter()
        for p in pts:
            if math.hypot(p[0]-c[0], p[1]-c[1]) > a.radius: continue
            r = room_of((p[0], p[1]), polys)
            if r: vote[r] += float(p[2])
        if not vote: out.append(it); st["표 없음"] += 1; continue
        top, tw = vote.most_common(1)[0]
        cur = it.get("room"); cw = vote.get(cur, 0.0)
        if top != cur and tw >= cw * (1.0 + a.margin):
            it = dict(it, room=top, _room_old=cur, _vote=round(tw / max(sum(vote.values()), 1e-9), 3))
            st["라벨 바뀜"] += 1
        else: st["유지"] += 1
        out.append(it)
    json.dump(out, open(os.path.join(hd, a.out), "w"))
    st["집"] += 1
print("ROOM_VOTE_DONE", dict(st))
