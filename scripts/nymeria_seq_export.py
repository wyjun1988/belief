#!/usr/bin/env python3
"""Nymeria 시퀀스를 data/seq 규약(rgb/%06d.jpg · pose/poses.txt · camera_info.json)으로 — reloc_pnp.py 실사 검증용.
    python scripts/nymeria_seq_export.py data/nymeria/loc49/<seq> data/nymeria/loc49_rgb/<seq>.mp4 data/seq/nym_<tag> [--every 30]
· RGB 1408² fisheye624 → 704²·f350 선형(이 시퀀스의 online_calibration 으로 정확 보정, aria_undistort.build_maps)
· 포즈 = MPS closed_loop_trajectory(T_world_device, 1 kHz) 를 프레임 시각에 최근접 → × T_device_camera(camera-rgb) = T_world_camera
· ⚠️ 시각 정합 근사: mp4 프레임 k 의 시각 = 궤적 첫 행 + k/30 s (nymeria_graph.py 와 같은 가정). ±수백 ms 오차 → 보행 중 수십 cm 오차 가능.
  위치 오차 분포를 읽을 때 이 한계를 함께 본다(정지 구간에서는 무시 가능)."""
import argparse, json, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aria_undistort import rgb_fisheye_params, build_maps
ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("mp4"); ap.add_argument("out")
ap.add_argument("--every", type=int, default=30); ap.add_argument("--size", type=int, default=704); ap.add_argument("--focal", type=float, default=350.0)
ap.add_argument("--max", type=int, default=0)
a = ap.parse_args()
cal = os.path.join(a.seq, "recording_head", "mps", "slam", "online_calibration.jsonl")
P = rgb_fisheye_params(cal); mx, my = build_maps(P, 1408, 1408, a.size, a.focal)
c0 = json.loads(open(cal).readline()); rgb = [c for c in c0["CameraCalibrations"] if c["Label"].lower() == "camera-rgb"][0]
w, (qx, qy, qz) = rgb["T_Device_Camera"]["UnitQuaternion"]; tx, ty, tz = rgb["T_Device_Camera"]["Translation"]
def quat2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
T_dc = np.eye(4); T_dc[:3, :3] = quat2R(w, qx, qy, qz); T_dc[:3, 3] = [tx, ty, tz]
import pandas as pd
tr = pd.read_csv(os.path.join(a.seq, "recording_head", "mps", "slam", "closed_loop_trajectory.csv"),
                 usecols=["tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device", "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"])
ts = tr["tracking_timestamp_us"].values.astype(np.int64); t0 = ts[0]
cap = cv2.VideoCapture(a.mp4); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
os.makedirs(os.path.join(a.out, "rgb"), exist_ok=True); os.makedirs(os.path.join(a.out, "pose"), exist_ok=True)
poses = []; k = 0; j = 0
while True:
    ok = cap.grab()
    if not ok: break
    if k % a.every == 0:
        ok, img = cap.retrieve()
        if not ok: break
        t_us = t0 + int(round(k / fps * 1e6)); i = int(np.searchsorted(ts, t_us)); i = min(max(i, 0), len(ts) - 1)
        if abs(int(ts[i]) - t_us) > 20000: k += 1; continue                    # 궤적 밖(>20 ms)
        r = tr.iloc[i]; Twd = np.eye(4); Twd[:3, :3] = quat2R(r["qw_world_device"], r["qx_world_device"], r["qy_world_device"], r["qz_world_device"]); Twd[:3, 3] = [r["tx_world_device"], r["ty_world_device"], r["tz_world_device"]]
        Twc = Twd @ T_dc
        cv2.imwrite(os.path.join(a.out, "rgb", "%06d.jpg" % j), cv2.remap(img, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        poses.append(" ".join("%.9g" % v for v in Twc.reshape(-1))); j += 1
        if a.max and j >= a.max: break
    k += 1
cap.release()
open(os.path.join(a.out, "pose", "poses.txt"), "w").write("\n".join(poses) + "\n")
json.dump(dict(width=a.size, height=a.size, fps=fps / a.every, fx=a.focal, fy=a.focal, cx=(a.size - 1) / 2.0, cy=(a.size - 1) / 2.0, distortion_model="none",
               source="nymeria aria_rgb_fisheye624 -> linear (seq calib) · pose=MPS closed_loop × T_device_camera · time=video k/fps from trajectory start (approx)",
               seq=os.path.basename(a.seq.rstrip("/")), every=a.every, n_video_frames=nfr), open(os.path.join(a.out, "camera_info.json"), "w"), indent=1)
print("→ %s · 프레임 %d (영상 %d장 중 %d장마다) · 궤적 %.0f s" % (a.out, j, nfr, a.every, (ts[-1] - t0) / 1e6))
