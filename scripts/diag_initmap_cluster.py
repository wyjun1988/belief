#!/usr/bin/env python3
"""초기맵 군집·순위 규칙 실험 — OWL 재실행 없이 initmap_raw.json(투영점 원자료 [x, z, 점수, 카메라x, 카메라z, 프레임])으로.
    python scripts/diag_initmap_cluster.py <THOR_ROOT>
① 타입단일 타겟(GT 는 채점에만)에 대해: 1위 인스턴스 오차·≤1 m 비율, 상위 3 안에 ≤1 m 인스턴스 존재 비율, 1위 방 정답."""
import json, glob, os, sys, math, collections, numpy as np
root = sys.argv[1]
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
def tri_pos(mem):
    """방위 교차(최소제곱): mem = [(x, z, s, ax, az)]. 시점이 2곳 이상·방위 분산이 있으면 교차점, 아니면 None."""
    A = np.zeros((2, 2)); b = np.zeros(2); W = 0.0
    for x, z, s, ax, az in mem:
        d = np.array([x - ax, z - az]); n = np.linalg.norm(d)
        if n < 0.2: continue
        u = d / n; Pm = np.eye(2) - np.outer(u, u); A += s * Pm; b += s * (Pm @ np.array([ax, az])); W += s
    if W <= 0: return None
    ev = np.linalg.eigvalsh(A / W)
    if ev[0] < 0.08: return None                       # 거의 한 방향(시점 1곳 또는 일직선) → 교차 불가
    p = np.linalg.solve(A, b)
    for x, z, s, ax, az in mem:                        # 카메라 뒤쪽 해는 버린다
        d = np.array([x - ax, z - az]);
        if np.dot(p - np.array([ax, az]), d) < 0: return None
    return p
def cluster(pts, clu, th, rank, minv, maxi=3, tri=False):
    """pts: [x, z, s, ax, az, k]. rank: sum|max|views|views_sum. tri: 군집 위치를 방위 교차로 재추정"""
    cl = []
    for x, z, s, ax, az, k in sorted(pts, key=lambda p: -p[2]):
        if s < th: continue
        hit = next((i for i, c in enumerate(cl) if math.hypot(x - c["c"][0], z - c["c"][1]) <= clu), None)
        if hit is None: cl.append(dict(c=np.array([x, z]), w=s, n=1, vs=[(ax, az)], mx=s, mem=[(x, z, s, ax, az)]))
        else:
            c = cl[hit]; c["c"] = (c["c"] * c["w"] + np.array([x, z]) * s) / (c["w"] + s); c["w"] += s; c["n"] += 1; c["vs"].append((ax, az)); c["mx"] = max(c["mx"], s); c["mem"].append((x, z, s, ax, az))
    if tri:
        for c in cl:
            p = tri_pos(c["mem"])
            if p is not None and math.hypot(p[0] - c["c"][0], p[1] - c["c"][1]) <= 3.0: c["c"] = p
    for c in cl:
        u = []
        for a in c["vs"]:
            if all(math.hypot(a[0] - b[0], a[1] - b[1]) >= 1.0 for b in u): u.append(a)
        c["nv"] = len(u)
    cl = [c for c in cl if c["nv"] >= minv]
    key = {"sum": lambda c: c["w"], "max": lambda c: c["mx"], "views": lambda c: (c["nv"], c["mx"]), "views_sum": lambda c: c["nv"] * c["mx"] + 1e-3 * c["w"], "nvw": lambda c: math.sqrt(c["nv"]) * c["w"]}[rank]
    return sorted(cl, key=key, reverse=True)[:maxi]
VARS = [("기준+tri", 2.0, 0.12, "sum", 1, True), ("th0.2+tri", 2.0, 0.2, "sum", 1, True), ("rank=max+tri", 2.0, 0.12, "max", 1, True), ("th0.2·max+tri", 2.0, 0.2, "max", 1, True),
        ("기준(2.0·0.12·sum·1)", 2.0, 0.12, "sum", 1), ("th0.2", 2.0, 0.2, "sum", 1), ("th0.3", 2.0, 0.3, "sum", 1), ("clu1.0", 1.0, 0.12, "sum", 1), ("rank=max", 2.0, 0.12, "max", 1),
        ("rank=views", 2.0, 0.12, "views", 1), ("rank=views_sum", 2.0, 0.12, "views_sum", 1), ("rank=nvw", 2.0, 0.12, "nvw", 1), ("minv2", 2.0, 0.12, "sum", 2), ("th0.2·clu1.0·views", 1.0, 0.2, "views", 1), ("th0.2·minv2·nvw", 2.0, 0.2, "nvw", 2)]
res = {v[0]: [] for v in VARS}; nh = 0
for hd in sorted(glob.glob(os.path.join(root, "house_*"))):
    hdr = os.path.realpath(hd); rf = hdr + "/initmap_raw.json"
    if not os.path.exists(rf): continue
    raw = json.load(open(rf)); g = json.load(open(hdr + "/gt.json")); polys = g["scene_meta"]["polys"]; nh += 1
    gf = hdr + "/room_groups.json"; gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}; grp = lambda r: gm.get(r, r) if r else r
    cnt = collections.Counter(v["type"] for v in g["gt0"].values()); mv = {m["oid"] for m in g["moves"]}
    for oid, v0 in g["gt0"].items():
        if oid in mv or cnt[v0["type"]] > 1 or not v0["room"] or v0["type"] not in raw: continue
        gs = (v0["pos"][0], v0["pos"][2])
        for name, clu, th, rank, minv, *tri in VARS:
            cl = cluster(raw[v0["type"]], clu, th, rank, minv, tri=bool(tri and tri[0]))
            if not cl: res[name].append(dict(miss=True)); continue
            errs = [math.hypot(c["c"][0] - gs[0], c["c"][1] - gs[1]) for c in cl]
            res[name].append(dict(miss=False, e1=errs[0], emin=min(errs), top3=min(errs) <= 1.0, room=grp(room_of(cl[0]["c"], polys)) == grp(v0["room"]), n_inst=len(cl)))
print("집 %d · 타겟 %d" % (nh, len(res[VARS[0][0]])))
print("%-22s %4s | 1위 오차 중앙 · ≤1m · ≤0.5m | 상위3 중 ≤1m | 1위 방 정답 | 인스턴스/타입" % ("규칙", "누락"))
for name, *_ in VARS:
    R = [r for r in res[name] if not r["miss"]]
    if not R: continue
    e1 = np.array([r["e1"] for r in R])
    print("%-22s %4d | %.2f m · %.2f · %.2f | %.2f | %.2f | %.1f" % (name, sum(r["miss"] for r in res[name]), np.median(e1), np.mean(e1 <= 1), np.mean(e1 <= 0.5), np.mean([r["top3"] for r in R]), np.mean([r["room"] for r in R]), np.mean([r["n_inst"] for r in R])))
