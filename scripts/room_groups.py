#!/usr/bin/env python3
"""열린 공간은 한 방 — 벽으로 나뉘지 않은 인접 방을 합친다 (사용자 결정 2026-09-04).
인접한 두 방 폴리곤의 공유 경계를 0.1m 간격으로 훑으며, 경계를 가로지르는 짧은 광선(±0.4m, 높이 1.2m·1.7m)이 막히지 않는
구간의 **최장 연속 길이 ≥ OPEN_MIN(2.0m)** 이면 합친다(문 ~0.9m 는 안 합쳐진다). union-find 전이 병합.

    HAB python scripts/room_groups.py --scene <id> --dataset <cfg> --house data/hssd20S2/house_0000 [--open-min 2.0]
출력: <house>/room_groups.json = {"groups": {room: group}, "open": [[A, B, 길이m]], "n_rooms", "n_groups"}
좌표: gt.json polys 는 x 미러 프레임 → 시뮬 좌표는 (-x, z).
"""
import argparse, json, os, sys
import numpy as np, habitat_sim, magnum as mn

ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--dataset", required=True)
ap.add_argument("--house", required=True); ap.add_argument("--open-min", type=float, default=2.0); ap.add_argument("--step", type=float, default=0.1)
ap.add_argument("--reach", type=float, default=0.4)
a = ap.parse_args()
g = json.load(open(os.path.join(a.house, "gt.json"))); polys = g["scene_meta"]["polys"]
cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = a.scene; cfg.scene_dataset_config_file = a.dataset; cfg.enable_physics = True
agc = habitat_sim.agent.AgentConfiguration(); agc.sensor_specifications = []
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agc]))
if not sim.pathfinder.is_loaded:
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); sim.recompute_navmesh(sim.pathfinder, ns)

def pip(pt, poly):
    x, z = pt; ins = False; n = len(poly)
    for k in range(n):
        x1, z1 = poly[k]; x2, z2 = poly[(k + 1) % n]
        if (z1 > z) != (z2 > z):
            if x < x1 + (z - z1) * (x2 - x1) / ((z2 - z1) or 1e-9): ins = not ins
    return ins
def floor_y(x, z):
    p = sim.pathfinder.snap_point(np.array([-x, 0.0, z]))     # 시뮬 좌표 (x 미러)
    return None if np.isnan(p[0]) else float(p[1])
def clear(p, q, y):
    """gt 프레임 점 p→q 사이가 높이 y 에서 막히지 않는가 (시뮬 좌표로 광선)"""
    P = np.array([-p[0], y, p[1]]); Q = np.array([-q[0], y, q[1]]); d = Q - P; L = float(np.linalg.norm(d))
    ray = habitat_sim.geo.Ray(mn.Vector3(*P), mn.Vector3(*(d / L)))
    res = sim.cast_ray(ray, max_distance=L)
    return not res.has_hits()

rooms = list(polys); parent = {r: r for r in rooms}
def find(r):
    while parent[r] != r: parent[r] = parent[parent[r]]; r = parent[r]
    return r
opens = []
for i, A in enumerate(rooms):
    pa = polys[A]; n = len(pa)
    for B in rooms[i + 1:]:
        pb = polys[B]; best = 0.0
        for k in range(n):                                       # A 의 각 변을 훑는다
            x1, z1 = pa[k]; x2, z2 = pa[(k + 1) % n]; L = float(np.hypot(x2 - x1, z2 - z1))
            if L < a.step: continue
            ex, ez = (x2 - x1) / L, (z2 - z1) / L; nx, nz = -ez, ex     # 변 방향·법선
            run = 0.0
            for s in np.arange(0.05, L, a.step):
                px, pz = x1 + ex * s, z1 + ez * s
                # 법선 방향으로 A 안 / B 안 인 쪽을 정한다
                side = None
                for sg in (1, -1):
                    pin = (px - sg * nx * 0.25, pz - sg * nz * 0.25); pout = (px + sg * nx * 0.25, pz + sg * nz * 0.25)
                    if pip(pin, pa) and pip(pout, pb): side = sg; break
                ok = False
                if side is not None:
                    p = (px - side * nx * a.reach, pz - side * nz * a.reach); q = (px + side * nx * a.reach, pz + side * nz * a.reach)
                    y0 = floor_y(*p); y1 = floor_y(*q)
                    if y0 is not None and y1 is not None and abs(y0 - y1) < 0.3:
                        ok = clear(p, q, y0 + 1.2) and clear(p, q, y0 + 1.7)
                run = run + a.step if ok else 0.0
                best = max(best, run)
        if best >= a.open_min:
            opens.append([A, B, round(best, 2)]); parent[find(A)] = find(B)
        elif best > 0.5:
            opens.append([A, B, round(best, 2), "door"])
groups = {}
for r in rooms:
    root = find(r); groups.setdefault(root, []).append(r)
gmap = {}
for root, mem in groups.items():
    name = "+".join(sorted(mem)) if len(mem) > 1 else mem[0]
    for r in mem: gmap[r] = name
out = dict(groups=gmap, open=opens, n_rooms=len(rooms), n_groups=len(groups), open_min=a.open_min)
json.dump(out, open(os.path.join(a.house, "room_groups.json"), "w"), indent=1, ensure_ascii=False)
print("%s 방 %d → 그룹 %d · 합침 %s" % (os.path.basename(a.house), len(rooms), len(groups),
      [(o[0], o[1], o[2]) for o in opens if len(o) == 3]), flush=True)
sim.close()
