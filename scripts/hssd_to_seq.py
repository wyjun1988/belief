#!/usr/bin/env python3
"""HSSD 에피소드 → 우리 포즈 파이프라인의 seq 형식 (rgb/ · camera_info.json · pose/poses.txt).

    python scripts/hssd_to_seq.py data/hssd20S2/house_0000 data/seq/hs2_house_0000

포즈 파이프라인(kx/depth/da3_runner → pose_stitch → build_graph/incremental_sfm)은 실사 seq
형식에 묶여 있어 THOR/HSSD 평가에는 붙지 않았다(2026-09-03). 이 어댑터가 첫 단계:
  rgb/%06d.jpg      live 프레임 심볼릭 링크 (1fps 그대로)
  camera_info.json  hfov 90° 핀홀 (fx = W/2)
  pose/poses.txt    GT c2w 4x4 (ATE 대조용. DA3 anchor/nopose 모드는 포즈를 읽지 않는다 —
                    프레임 수 확인과 window 모드 조건화에만 쓰인다)
좌표: GT apos=[x,z](미러 규약)·yaw(0°=+z, 시계 증가) → c2w 는 y-up, 카메라 +z 전방·+x 우.
"""
import glob, json, os, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
g = json.load(open(os.path.join(src, "gt.json")))
frames = sorted(glob.glob(os.path.join(src, "live", "*.jpg")))
assert frames, "live 프레임 없음"
from PIL import Image
W, H = Image.open(frames[0]).size
os.makedirs(os.path.join(dst, "rgb"), exist_ok=True); os.makedirs(os.path.join(dst, "pose"), exist_ok=True)
for k, f in enumerate(frames):
    lnk = os.path.join(dst, "rgb", "%06d.jpg" % k)
    if not os.path.exists(lnk): os.symlink(os.path.abspath(f), lnk)
fx = W / 2.0
json.dump(dict(width=W, height=H, fps=1.0, intrinsics=[[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]],
               fx=fx, fy=fx, cx=W / 2.0, cy=H / 2.0, distortion_model="none",
               distortion_coefficients=[0, 0, 0, 0, 0], source="habitat rgb hfov90", hfov_deg=90.0),
          open(os.path.join(dst, "camera_info.json"), "w"), indent=1)
live = {m["t"]: m for m in g["live"]}
rows = []
for k, f in enumerate(frames):
    t = int(os.path.basename(f)[:-4]); m = live[t]
    x, z = m["apos"]; yaw = np.radians(m["yaw"]); eye = 1.5
    fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)]); up = np.array([0.0, 1.0, 0.0]); right = np.cross(fwd, up)
    R = np.stack([right, up, fwd], 1)                   # 카메라 축(x우, y상, z전방) → 월드
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x, eye, z]
    rows.append(T.reshape(-1))
np.savetxt(os.path.join(dst, "pose", "poses.txt"), np.array(rows))
print("seq → %s · 프레임 %d · %dx%d · fx %.0f" % (dst, len(frames), W, H, fx))
