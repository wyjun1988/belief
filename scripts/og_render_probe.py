#!/usr/bin/env python3
"""Isolation probe for the Isaac 5.1 headless renderer crash.

Varies one factor at a time: which annotators are attached, where the camera
looks, how many render ticks per pose, and whether get_obs() is called at all.
Prints one line per pose so the crash frame index is unambiguous.
"""
import argparse, math, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="Rs_int")
ap.add_argument("--mods", default="", help="comma list added on top of rgb")
ap.add_argument("--pose", default="horiz", choices=("horiz", "down", "fixed_horiz", "none"))
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--renders", type=int, default=4)
ap.add_argument("--getobs", type=int, default=1)
ap.add_argument("--sensor", default="viewer", choices=("viewer", "new"))
ap.add_argument("--res", default="", help="WxH for --sensor new")
ap.add_argument("--clip", type=float, default=0.0, help="far clipping range override")
ap.add_argument("--tag", default="probe")
a = ap.parse_args()

import numpy as np
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True

def gpumem():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().replace("\n", " | ")
        return o
    except Exception as e:
        return str(e)

env = og.Environment(configs=dict(
    scene=dict(type="InteractiveTraversableScene", scene_model=a.scene),
    robots=[], env=dict(action_frequency=30, physics_frequency=30)))
sc = env.scene

mods = [m for m in a.mods.split(",") if m]
if a.sensor == "viewer":
    cam = og.sim.viewer_camera
    for m in mods:
        cam.add_modality(m)
else:
    from omnigibson.sensors import VisionSensor
    w, h = (int(x) for x in a.res.split("x")) if a.res else (1280, 720)
    cam = VisionSensor(relative_prim_path="/og_render_probe", name="og_render_probe",
                       modalities=["rgb"] + mods, image_height=h, image_width=w)
    cam.load(None); cam.initialize()
if a.clip > 0:
    cam.clipping_range = (0.01, a.clip)
for _ in range(4):
    og.sim.render()
print(f"[{a.tag}] READY mods={['rgb']+mods} pose={a.pose} sensor={a.sensor} renders={a.renders} "
      f"getobs={a.getobs} device={og.sim.device} mem={gpumem()}", flush=True)

# USD cameras look down local -Z with local +Y up.  In OmniGibson's z-up stage an
# identity quaternion therefore stares at the floor -- which is exactly what the
# old [0,0,sin,cos] pose produced.  BASE maps local -Z to world +X and local +Y to
# world +Z, so yaw=0 is a true horizontal heading along +X.
BASE = np.array([0.5, -0.5, -0.5, 0.5])          # (x, y, z, w)

def qmul(p, q):
    x1, y1, z1, w1 = p; x2, y2, z2, w2 = q
    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2,
                     w1*w2 - x1*x2 - y1*y2 - z1*z2])

def horiz_quat(yaw_deg, pitch_deg=0.0):
    h = math.radians(yaw_deg) / 2
    yq = np.array([0.0, 0.0, math.sin(h), math.cos(h)])
    q = qmul(yq, BASE)
    if pitch_deg:
        p = math.radians(pitch_deg) / 2
        q = qmul(q, np.array([math.sin(p), 0.0, 0.0, math.cos(p)]))
    return q

def down_quat(yaw_deg):
    h = math.radians(yaw_deg) / 2
    return np.array([0.0, 0.0, math.sin(h), math.cos(h)])

def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

rng = np.random.default_rng(0)
pts = []
while len(pts) < 8:
    _, p = sc.get_random_point(floor=0)
    pts.append(arr(p).astype(float)[:2])
fixed = pts[0]

for i in range(a.n):
    yaw = (i * 37) % 360
    p = fixed if a.pose in ("fixed_horiz", "none") else pts[i % len(pts)]
    if a.pose != "none":
        q = down_quat(yaw) if a.pose == "down" else horiz_quat(yaw, -5.0)
        cam.set_position_orientation(position=[float(p[0]), float(p[1]), 1.5],
                                     orientation=[float(v) for v in q])
    for _ in range(a.renders):
        og.sim.render()
    nseg = -1
    if a.getobs:
        ob, inf = cam.get_obs()
        if "seg_instance" in ob:
            nseg = int(len(np.unique(arr(ob["seg_instance"]))))
        elif "seg_semantic" in ob:
            nseg = int(len(np.unique(arr(ob["seg_semantic"]))))
    tail = f" mem={gpumem()}" if i % 20 == 0 else ""
    print(f"[{a.tag}] POSE {i} yaw={yaw} nseg={nseg}{tail}", flush=True)

print(f"[{a.tag}] DONE n={a.n} mem={gpumem()}", flush=True)
