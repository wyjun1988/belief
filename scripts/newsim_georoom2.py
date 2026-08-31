#!/usr/bin/env python3
"""newsim 투영 예행 ② — GT 픽셀이 아니라 **OWL 검출 픽셀**로 방 판정. (사다리 ② newsim판)

    A3_PREFIX=/tmp/ns_a_ GT=~/work/ns_root/house_0000/gt.json \\
      python scripts/newsim_georoom2.py

단일 인스턴스 정적 타입만: 검출 패치 픽셀(a3) + 포즈(GT, live yaw/pitch/apos)
+ GT 거리 → 3D 점 → 방 폴리곤. 정답 = 그 정적 물체의 방.
OWL 패치 좌표는 **패딩 정방(760) 기준** — 세로는 570 안쪽만 유효.
THOR ②'(투표 yaw 0.739/0.670)의 newsim 대응 수치를 만든다 — 에피소드가 오면
이동 타겟에 같은 사슬을 적용한다.
"""
import json, os
import numpy as np

A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "/tmp/ns_a_"))
GTP = os.path.expanduser(os.environ.get("GT", "~/work/ns_root/house_0000/gt.json"))
W, H, S = 760, 570, 760.0                 # S = 패딩 정방 변
TH = float(os.environ.get("TH", "0.20"))
CM = 1.0                                   # 어댑터 gt 는 이미 m


def cam_axes(apos, yaw, pitch):
    p, y = np.radians(pitch), np.radians(yaw)
    fwd = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    right = np.array([-np.sin(y), np.cos(y), 0.0])
    up = np.cross(right, fwd)
    return np.array([apos[0], apos[1], 1.6]), fwd, right, up   # 카메라 높이 ~1.6m


def ray_pt(apos, yaw, pitch, u, v, dist_h):
    """dist_h = 수평 거리 — 광선의 수평 성분으로 정규화해 경사 미달을 막는다."""
    loc, fwd, right, up = cam_axes(apos, yaw, pitch)
    f = (W / 2) / np.tan(np.radians(45.0))
    d = fwd + ((u - W / 2) / f) * right + ((H / 2 - v) / f) * up
    hz = np.hypot(d[0], d[1])
    if hz < 1e-6: return loc
    return loc + d * (dist_h / hz)


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
    return min(polys, key=lambda r: min((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2
                                        for v in polys[r]))


g = json.load(open(GTP))
polys = g["scene_meta"]["polys"]
live = {m["t"]: m for m in g["live"]}
za = np.load(A3P + "house_0000.npz", allow_pickle=True)
Sm, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
vocab, nT = list(za["vocab"]), int(za["nT"])

from collections import Counter
def base(t):                      # armchair_01 → armchair · ArmChair → armchair
    return "".join(c for c in t if c.isalpha()).lower()
byt = {}
for k, v in g["scene_meta"]["static"].items():
    if v.get("pos"): byt.setdefault(base(v["type"]), []).append((v["pos"], v["room"]))
singles = {t: inst[0] for t, inst in byt.items() if len(inst) == 1}

ok = [0, 0]; miss_pose = 0; n_det = 0
for c in range(nT, len(vocab)):
    t = base(vocab[c])
    if t not in singles: continue
    (spos, srm) = singles[t]
    for i in range(len(ts)):
        if Sm[i, c] < TH: continue
        m = live.get(int(ts[i]))
        if not m or m.get("yaw") is None: miss_pose += 1; continue
        u = (P[i, c] % pw + .5) / pw * S
        v = (P[i, c] // pw + .5) / ph * S
        if v >= H: continue                       # 패딩 영역
        d = float(np.hypot(spos[0] - m["apos"][0], spos[1] - m["apos"][1]))
        if not (0.3 < d < 8): continue
        pt = ray_pt(m["apos"], m["yaw"], m.get("pitch", 0.0), u, v, d)
        n_det += 1
        ok[room_at((pt[0], pt[1]), polys) == srm] += 1
print("단일 정적 타입 %d · 검출 투영 %d건 (포즈 결손 %d)" % (len(singles), n_det, miss_pose))
if sum(ok):
    print("검출 픽셀 투영 방 정답률: %.3f" % (ok[1] / sum(ok)))
