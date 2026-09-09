#!/usr/bin/env python3
"""initmap_raw.json(투영점 원자료)에서 군집·순위 규칙을 바꿔 초기맵 파일을 다시 쓴다 (OWL 재실행 없음).
    python scripts/recluster_initmap.py <THOR_ROOT> --out initmap_owl_rc.json [--clu 2.0 --th 0.12 --rank max --minv 1 --maxi 3]
산출 형식은 build_initmap 과 같다: [{type, room, w, pos:[x,z], n}] (w = 군집 점수합, 순위는 --rank 로 정렬해 저장)."""
import argparse, json, glob, os, math, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", default="initmap_owl_rc.json"); ap.add_argument("--clu", type=float, default=2.0)
ap.add_argument("--raw", default="initmap_raw.json"); ap.add_argument("--th", type=float, default=0.12); ap.add_argument("--rank", default="max", choices=["sum", "max", "views", "views_sum", "nvw"]); ap.add_argument("--minv", type=int, default=1); ap.add_argument("--maxi", type=int, default=3)
a = ap.parse_args()
def _pip(pt, poly):
    x, z = pt; ins = False; n = len(poly)
    for k in range(n):
        x1, z1 = poly[k]; x2, z2 = poly[(k + 1) % n]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
    return ins
def room_of(pt, polys):
    for r, pl in polys.items():
        if _pip(pt, pl): return r
    return min(polys, key=lambda r: math.hypot(pt[0] - np.mean([p[0] for p in polys[r]]), pt[1] - np.mean([p[1] for p in polys[r]])))
KEY = {"sum": lambda c: c["w"], "max": lambda c: c["mx"], "views": lambda c: (c["nv"], c["mx"]), "views_sum": lambda c: c["nv"] * c["mx"] + 1e-3 * c["w"], "nvw": lambda c: math.sqrt(c["nv"]) * c["w"]}
n_h = 0
for hd in sorted(glob.glob(os.path.join(a.root, "house_*"))):
    hdr = os.path.realpath(hd); rf = os.path.join(hdr, a.raw)
    if not os.path.exists(rf): continue
    raw = json.load(open(rf)); polys = json.load(open(hdr + "/gt.json"))["scene_meta"]["polys"]; out = []
    for t, pts in raw.items():
        cl = []
        for x, z, s, ax, az, k in (p[:6] for p in sorted(pts, key=lambda p: -p[2])):     # raw 는 2026-09-10 부터 [.., top, margin] 두 필드가 더 붙는다
            if s < a.th: continue
            hit = next((i for i, c in enumerate(cl) if math.hypot(x - c["c"][0], z - c["c"][1]) <= a.clu), None)
            if hit is None: cl.append(dict(c=np.array([x, z]), w=s, n=1, vs=[(ax, az)], mx=s))
            else:
                c = cl[hit]; c["c"] = (c["c"] * c["w"] + np.array([x, z]) * s) / (c["w"] + s); c["w"] += s; c["n"] += 1; c["vs"].append((ax, az)); c["mx"] = max(c["mx"], s)
        for c in cl:
            u = []
            for v in c["vs"]:
                if all(math.hypot(v[0] - b[0], v[1] - b[1]) >= 1.0 for b in u): u.append(v)
            c["nv"] = len(u)
        cl = sorted([c for c in cl if c["nv"] >= a.minv], key=KEY[a.rank], reverse=True)[:a.maxi]
        for c in cl: out.append(dict(type=t, room=room_of(c["c"], polys), w=round(float(c["w"]), 3), pos=[round(float(c["c"][0]), 2), round(float(c["c"][1]), 2)], n=int(c["n"]), mx=round(float(c["mx"]), 4), nv=int(c["nv"])))
    json.dump(out, open(os.path.join(hdr, a.out), "w")); n_h += 1
print("재군집 %d채 → %s (clu %.1f · th %.2f · rank %s · minv %d · maxi %d)" % (n_h, a.out, a.clu, a.th, a.rank, a.minv, a.maxi))
