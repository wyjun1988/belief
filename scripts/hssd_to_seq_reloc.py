#!/usr/bin/env python3
"""[reloc_hloc 용 어댑터 — 기존 hssd_to_seq.py(§136 포즈 파이프라인)와 별개] HSSD 집(map/·live/·gt.json)을 data/seq 규약(rgb/·pose/poses.txt·camera_info.json)으로 — reloc_hloc.py 를 시뮬에 그대로 적용하려고.
    python scripts/hssd_to_seq_reloc.py data/hssd20_c3/house_0000 data/seq/hssd20c3_h0000 [--mirror 1]
rgb/ = map 프레임(스캔) 이어서 live 프레임 — --scan-end 는 map 장수. **하드링크**(심볼릭 링크면 pycolmap 이 대상 경로를 이름으로 저장해 이름이 어긋난다). 포즈 = gt.json 의 apos(x,z)·높이 1.5·yaw(·pitch) → T_world_camera(COLMAP 축: x 우·y 하·z 전방).
gt.json 좌표는 x 미러 프레임(room_groups.py 주석) → 기본 --mirror 1 로 x 를 뒤집어 오른손 좌표계로 만든다. 규약이 틀리면 GT 삼각측량 지도의
점 수·재투영 오차가 무너지므로 그것으로 검증한다."""
import argparse, json, os, sys, numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("out"); ap.add_argument("--mirror", type=int, default=1); ap.add_argument("--height", type=float, default=1.5)
ap.add_argument("--yaw-sign", type=float, default=1.0)
a = ap.parse_args(); hd = os.path.realpath(a.house); g = json.load(open(os.path.join(hd, "gt.json")))
maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg")); lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg"))
assert len(g["map"]) == len(maps); live = {m["t"]: m for m in g["live"]} if isinstance(g["live"], list) else g["live"]
W, H = Image.open(os.path.join(hd, "map", maps[0])).size; fx = W / 2.0
os.makedirs(os.path.join(a.out, "rgb"), exist_ok=True); os.makedirs(os.path.join(a.out, "pose"), exist_ok=True)
def Twc(apos, yaw_deg, pitch_deg=0.0):
    x, z = apos; sx = -1.0 if a.mirror else 1.0; x = sx * x
    y = np.radians(a.yaw_sign * yaw_deg); p = np.radians(pitch_deg)
    f = np.array([sx * np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])       # 전방(sfm_reloc to_ours: yaw = atan2(v_x, v_z))
    f /= np.linalg.norm(f); up = np.array([0, 1.0, 0]); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)   # r 우, u 상
    Rwc = np.stack([r, -u, f], 1)                                                     # COLMAP 카메라: x 우 · y 하 · z 전방
    T = np.eye(4); T[:3, :3] = Rwc; T[:3, 3] = [x, a.height, z]; return T
rows = []; k = 0
for f, m in zip(maps, g["map"]):
    os.path.exists(os.path.join(a.out, "rgb", "%06d.jpg" % k)) or os.link(os.path.join(hd, "map", f), os.path.join(a.out, "rgb", "%06d.jpg" % k))
    rows.append(Twc(m["apos"], m["yaw"], m.get("pitch", 0.0))); k += 1
n_map = k
for f in lives:
    t = int(f[:-4]); m = live.get(t)
    if m is None: continue
    os.path.exists(os.path.join(a.out, "rgb", "%06d.jpg" % k)) or os.link(os.path.join(hd, "live", f), os.path.join(a.out, "rgb", "%06d.jpg" % k))
    rows.append(Twc(m["apos"], m["yaw"], m.get("pitch", 0.0))); k += 1
open(os.path.join(a.out, "pose", "poses.txt"), "w").write("\n".join(" ".join("%.9g" % v for v in T.reshape(-1)) for T in rows) + "\n")
json.dump(dict(width=W, height=H, fps=1.0, fx=fx, fy=fx, cx=W / 2.0, cy=H / 2.0, distortion_model="none", source="hssd gt.json (apos, yaw, pitch) mirror=%d" % a.mirror, n_map=n_map, house=os.path.basename(hd)),
          open(os.path.join(a.out, "camera_info.json"), "w"), indent=1)
print("→ %s · 스캔(map) %d · 라이브 %d · fx %.0f · --scan-end %d" % (a.out, n_map, k - n_map, fx, n_map))
