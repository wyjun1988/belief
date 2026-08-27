#!/usr/bin/env python3
"""새 시뮬 카메라 투영 — **검증된 규약** (frame1 체스판 5px 오차로 확정, §키트).

  FWD = (cosP·cosY, cosP·sinY, sinP)   RIGHT = (−sinY, cosY, 0)   UP = R×F
  u = W/2 + f·(d·R)/(d·F)              v = H/2 − f·(d·U)/(d·F)
  (Unreal cm 좌표 그대로 · pitch 음수 = 아래 · yaw 그대로)

⚠️ 검증 이력: 회귀·화면안비율 검증은 표본 부족/거울상 모호로 실패했고,
**실제 프레임의 알려진 물체 배치와 직접 대조**가 결정타였다. 새 exporter 버전이
오면 같은 방법으로 재검증할 것 (scripts 끝의 selftest).
"""
import json, os
import numpy as np


def cam_axes(cam):
    p, y, _ = np.radians(cam["rotation_pyr_deg"])
    fwd = np.array([np.cos(p)*np.cos(y), np.cos(p)*np.sin(y), np.sin(p)])
    right = np.array([-np.sin(y), np.cos(y), 0.0])
    up = np.cross(right, fwd)
    return np.array(cam["location"]), fwd, right, up


def project(pt, cam, W=760, H=570, fov_deg=None):
    loc, fwd, right, up = cam_axes(cam)
    f = (W / 2) / np.tan(np.radians((fov_deg or cam.get("fov_deg", 90.0)) / 2))
    d = np.asarray(pt, float) - loc
    z = d @ fwd
    if z <= 5: return None
    return float(W/2 + f*(d@right)/z), float(H/2 - f*(d@up)/z), float(z)


def project_aabb(aabb, cam, W=760, H=570):
    """AABB 8꼭짓점 투영 → 화면 bbox (일부만 앞이면 그 부분으로)."""
    lo, hi = aabb["min"], aabb["max"]
    us, vs = [], []
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                r = project((cx, cy, cz), cam, W, H)
                if r: us.append(r[0]); vs.append(r[1])
    if len(us) < 4: return None
    x0, x1 = max(0, min(us)), min(W, max(us))
    y0, y1 = max(0, min(vs)), min(H, max(vs))
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    return [int(x0), int(y0), int(x1), int(y1)]


def selftest(ep):
    sg = json.load(open(os.path.join(ep, "scene_graph.json")))
    chess = next(o for o in sg["objects"] if "chess" in o["id"].lower())
    for k, line in enumerate(open(os.path.join(ep, "observed_graph_updates.jsonl"))):
        if k == 20:
            cam = next(c for c in json.loads(line)["cameras"] if c["source"] == "ego")
            break
    r = project(chess["transform"]["location"], cam)
    print("체스판 투영 %s — 기대 (385, 259) 근방" % (tuple(round(x) for x in r[:2]),))


if __name__ == "__main__":
    import sys
    selftest(sys.argv[1])
