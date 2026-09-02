#!/usr/bin/env python3
"""GT 방 라벨을 폴리곤으로 재판정 — 생성기가 AABB 로 라벨링했던 데이터의 정정.

    python scripts/gt_relabel.py data/hssd20S2 [--apply]

gt0.room · moves.to/frm · live.room 을 저장된 좌표(gt0.pos / moves.pos / live.apos)로
다시 구한다. --apply 면 gt.json 을 덮어쓰고 원본은 gt.aabb.json 으로 남긴다.
"""
import glob, json, os, sys
root = sys.argv[1]; apply = "--apply" in sys.argv
def pip(pt, poly):
    x, z = float(pt[0]), float(pt[1]); ins = False; n = len(poly)
    for k in range(n):
        x1, z1 = poly[k]; x2, z2 = poly[(k + 1) % n]
        if (z1 > z) != (z2 > z):
            xi = x1 + (z - z1) * (x2 - x1) / ((z2 - z1) or 1e-9)
            if x < xi: ins = not ins
    return ins
tot = {"gt0": [0, 0], "moves": [0, 0], "live": [0, 0]}
for f in sorted(glob.glob(os.path.join(root, "house_*", "gt.json"))):
    g = json.load(open(f)); polys = g["scene_meta"]["polys"]
    def room_at(x, z):
        hits = [r for r, pl in polys.items() if pip((x, z), pl)]
        if hits: return min(hits, key=lambda r: (lambda P: abs(sum(P[k][0]*P[(k+1)%len(P)][1]-P[(k+1)%len(P)][0]*P[k][1] for k in range(len(P)))))(polys[r]))   # 겹치면 면적 작은 방
        return None   # 폴리곤 밖 → 기존 라벨 유지 (HSSD region 은 바닥 전체를 덮지 않는다)
    for oid, v in g["gt0"].items():
        r = room_at(v["pos"][0], v["pos"][2]) or v["room"]; tot["gt0"][1] += 1; tot["gt0"][0] += (r != v["room"])
        if apply: v["room"] = r
    for m in g["moves"]:
        if m.get("pos"):
            r = room_at(m["pos"][0], m["pos"][2]) or m["to"]; tot["moves"][1] += 1; tot["moves"][0] += (r != m["to"])
            if apply: m["to"] = r
    for l in g["live"]:
        r = room_at(l["apos"][0], l["apos"][1]) or l["room"]; tot["live"][1] += 1; tot["live"][0] += (r != l["room"])
        if apply: l["room"] = r
    if apply:
        last = {}
        for m in sorted(g["moves"], key=lambda m: m["t"]):
            m["frm"] = last.get(m["oid"], g["gt0"][m["oid"]]["room"]); last[m["oid"]] = m["to"]
        bak = f.replace("gt.json", "gt.aabb.json")
        if not os.path.exists(bak): os.rename(f, bak)
        json.dump(g, open(f, "w"))
for k, v in tot.items(): print("%-6s AABB≠폴리곤 %d/%d = %.1f%%" % (k, v[0], v[1], 100 * v[0] / max(v[1], 1)))
print("적용됨" if apply else "(보고만 — --apply 로 적용)")
