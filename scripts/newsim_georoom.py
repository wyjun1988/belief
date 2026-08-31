#!/usr/bin/env python3
"""투영 국소화의 newsim 이식 — §106-111 스택(광선 역산·삼각측량·방 폴리곤 판정).

    python scripts/newsim_georoom.py --ep <원본 에피소드 dir> --gt data/newsim/ep1/gt.json

THOR 판과의 차이:
  · 포즈: 앵커 투표 불필요 — 에피소드 포즈 스트림(apos·yaw·pitch, 어댑터 v2)이
    상한용. 시스템측 포즈는 SfM 사슬 몫(사다리 ④ — 연속 영상이라 실사슬 성립).
  · 광선: pitch 가 가파를 수 있어(-38° 관측) 수평 근사 대신 **검증된 3D 규약**
    (newsim_project, 체스판 5px)을 역방향으로 쓴다: dir = F + x̂·R + ŷ·U.
  · 삼각측량: 3D 광선 쌍의 최근접점 중점.

이 스크립트의 selftest 는 실측이 아니라 **이식 검증**이다: 정적 물체를 투영해
얻은 픽셀을 다시 역산해 위치·방이 복원되는지(왕복), 두 프레임 광선의 교점이
실위치에 근접하는지(삼각측량). OWL 검출 픽셀 실측은 에피소드 캐시가 생기면
같은 함수로 돈다 (loc_ray / loc_tri 그대로).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newsim_project import cam_axes, project  # 검증된 규약 (체스판 5px)

CM = 0.01
W, H = 760, 570


def ray_of(cam, u, v, fov_deg=90.0):
    """픽셀 → 세계 광선 (원점, 단위방향). project() 의 정확한 역."""
    loc, fwd, right, up = cam_axes(cam)
    f = (W / 2) / np.tan(np.radians(fov_deg / 2))
    d = fwd + ((u - W / 2) / f) * right + ((H / 2 - v) / f) * up
    return np.asarray(loc, float), d / np.linalg.norm(d)


def loc_ray(cam, u, v, dist_cm):
    """단일 프레임: 광선 + 거리 → 3D 점 (cm)."""
    o, d = ray_of(cam, u, v)
    return o + d * dist_cm


def loc_tri(cam1, uv1, cam2, uv2):
    """두 프레임 광선의 최근접점 중점 (cm). 근평행·후방이면 None."""
    o1, d1 = ray_of(cam1, *uv1); o2, d2 = ray_of(cam2, *uv2)
    c = np.dot(d1, d2)
    if abs(c) > 0.966: return None            # 각도차 < 15°
    b = o2 - o1
    t1 = (np.dot(b, d1) - c * np.dot(b, d2)) / (1 - c * c)
    t2 = (c * np.dot(b, d1) - np.dot(b, d2)) / (1 - c * c)
    if not (30 < t1 < 1200 and 30 < t2 < 1200): return None   # 0.3~12 m
    return (o1 + t1 * d1 + o2 + t2 * d2) / 2


def room_at(xy_m, polys):
    x, z = xy_m
    for r, pl in polys.items():
        n = len(pl); c2 = False
        for i in range(n):
            x1, z1 = pl[i]; x2, z2 = pl[(i + 1) % n]
            if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1:
                c2 = not c2
        if c2: return r
    return min(polys, key=lambda r: min((x - v[0]) ** 2 + (z - v[1]) ** 2 for v in polys[r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True, help="원본 에피소드 dir (scene_graph·observed 스트림)")
    ap.add_argument("--gt", required=True, help="어댑터 v2 산출 gt.json")
    args = ap.parse_args()
    g = json.load(open(args.gt))
    polys = g["scene_meta"]["polys"]
    sg = json.load(open(os.path.join(args.ep, "scene_graph.json")))
    objs = {o["id"]: o for o in sg["objects"] if o.get("aabb_world")}
    import collections
    cams = {}
    for line in open(os.path.join(args.ep, "observed_graph_updates.jsonl")):
        d = json.loads(line)
        for c in d.get("cameras", []):
            if c.get("source") == "ego": cams[int(d["time_s"])] = c

    # ── 이식 검증 1: 왕복 — 정적 물체 투영 픽셀 → 역산 → 위치·방 복원 ──
    stat = {k: v for k, v in g["scene_meta"]["static"].items() if k in objs}
    errs = []; rooms_ok = [0, 0]; obs = collections.defaultdict(list)
    for t, cam in sorted(cams.items()):
        for oid, v in stat.items():
            o = objs[oid]
            lo, hi = o["aabb_world"]["min"], o["aabb_world"]["max"]
            ctr3 = [(lo[k] + hi[k]) / 2 for k in range(3)]
            pr = project(ctr3, cam, W, H)
            if pr is None or not (20 < pr[0] < W - 20 and 20 < pr[1] < H - 20): continue
            u, v2, _z = pr
            dist = float(np.linalg.norm(np.asarray(ctr3) - np.asarray(cam["location"])))
            pt = loc_ray(cam, u, v2, dist)
            errs.append(float(np.linalg.norm(pt - ctr3)) * CM)
            rm = room_at((pt[0] * CM, pt[1] * CM), polys)
            rooms_ok[rm == v["room"]] += 1
            obs[oid].append((t, cam, (u, v2), ctr3))
    print("왕복: 관측 %d · 위치오차 중앙값 %.3fm · 방 복원 %.3f"
          % (len(errs), float(np.median(errs)) if errs else -1,
             rooms_ok[1] / max(sum(rooms_ok), 1)))

    # ── 이식 검증 2: 삼각측량 — 두 프레임 광선 교점 vs 실위치 ──
    terr = []; trm = [0, 0]
    for oid, os_ in obs.items():
        if len(os_) < 2: continue
        for a in range(0, len(os_) - 1, 2):
            (t1, c1, uv1, ctr3), (t2, c2_, uv2, _) = os_[a], os_[a + 1]
            pt = loc_tri(c1, uv1, c2_, uv2)
            if pt is None: continue
            terr.append(float(np.linalg.norm(pt - ctr3)) * CM)
            trm[room_at((pt[0] * CM, pt[1] * CM), polys) == stat[oid]["room"]] += 1
    print("삼각측량: 교점 %d · 위치오차 중앙값 %.3fm · 방 복원 %.3f"
          % (len(terr), float(np.median(terr)) if terr else -1,
             trm[1] / max(sum(trm), 1)))
    print("(이식 검증 — OWL 검출 픽셀 실측은 에피소드 캐시 도착 후 같은 함수로)")


if __name__ == "__main__":
    main()
