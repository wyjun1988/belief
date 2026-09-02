#!/usr/bin/env python3
"""Close the coordinate convention numerically (SPEC section 5) with RGB+depth only.

Three independent closures, all reported as numbers:
  1. projection-vs-depth: the analytic pixel of an object centre must land on the
     rendered depth of that object (+-tol) whenever a PhysX ray says it is visible.
  2. anchor yaw inversion: recover yaw from (world anchor, screen x) and compare to
     the yaw we commanded -- median absolute error must be ~0 deg.
  3. readback: the quaternion the sim reports must equal the one we set.
Also dumps annotated frames so the horizon can be eyeballed.
"""
import argparse, json, math, os

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="Rs_int")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--pitch", type=float, default=0.0)
ap.add_argument("--tol", type=float, default=0.30)
ap.add_argument("--out", default="/mnt/ssd2/wooyeol/work/og_diag_20260902/pose_verify")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import numpy as np
from PIL import Image, ImageDraw
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True

env = og.Environment(configs=dict(
    scene=dict(type="InteractiveTraversableScene", scene_model=a.scene),
    robots=[], env=dict(action_frequency=30, physics_frequency=30)))
sc = env.scene
from omnigibson.utils.sampling_utils import raytest

cam = og.sim.viewer_camera
cam.add_modality("depth_linear")          # rgb+depth is the stable pair (og_render_probe A/B)
for _ in range(4):
    og.sim.render()


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


BASE = np.array([0.5, -0.5, -0.5, 0.5])   # local -Z -> world +X, local +Y -> world +Z


def qmul(p, q):
    x1, y1, z1, w1 = p
    x2, y2, z2, w2 = q
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def cam_quat(yaw_deg, pitch_deg=0.0):
    h = math.radians(yaw_deg) / 2
    q = qmul(np.array([0.0, 0.0, math.sin(h), math.cos(h)]), BASE)
    if pitch_deg:
        p = math.radians(pitch_deg) / 2
        q = qmul(q, np.array([math.sin(p), 0.0, 0.0, math.cos(p)]))
    return q


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


W, H = int(cam.image_width), int(cam.image_height)
FOC, HAP = float(cam.focal_length), float(cam.horizontal_aperture)
FX = W * FOC / HAP
FY = FX                                   # square pixels: vertical aperture = HAP*H/W
CX, CY = W / 2.0, H / 2.0
print("intrinsics W=%d H=%d focal=%s h_aperture=%s fx=%.3f" % (W, H, FOC, HAP, FX), flush=True)

SKIP = {"walls", "floors", "ceilings", "wall", "floor", "ceiling", "background",
        "door", "window", "roof", "lawn", "driveway"}
objs = {}
for o in sc.objects:
    cat = str(getattr(o, "category", o.name)).lower()
    if cat in SKIP:
        continue
    try:
        lo, hi = o.aabb
        c = (arr(lo).astype(float) + arr(hi).astype(float)) / 2
    except Exception:
        try:
            p, _ = o.get_position_orientation()
            c = arr(p).astype(float)
        except Exception:
            continue
    objs[str(o.name)] = dict(ctr3=c, cat=cat)
print("objects considered %d" % len(objs), flush=True)


def visible_ray(eye, tgt, name):
    """First PhysX hit along eye->tgt must be the target itself."""
    r = raytest([float(v) for v in eye], [float(v) for v in tgt])
    L = float(np.linalg.norm(np.asarray(tgt, float) - np.asarray(eye, float)))
    if not r.get("hit"):
        return True
    d = float(arr(r["distance"]))
    body = str(r.get("rigidBody", "")) + "|" + str(r.get("collision", ""))
    return (name in body) or (abs(d - L) < 0.15)


pts = []
while len(pts) < a.n:
    _, p = sc.get_random_point(floor=0)
    pts.append(arr(p).astype(float)[:2])

proj_ok = proj_n = 0
resid, samples, readback_err = [], [], []
for k in range(a.n):
    p = pts[k]
    yaw = float((k * 37) % 360)
    q = cam_quat(yaw, a.pitch)
    eye = np.array([p[0], p[1], 1.5])
    cam.set_position_orientation(position=[float(v) for v in eye],
                                 orientation=[float(v) for v in q])
    for _ in range(4):
        og.sim.render()
    _, gq = cam.get_position_orientation()
    gq = arr(gq).astype(float)
    readback_err.append(float(min(np.linalg.norm(gq - q), np.linalg.norm(gq + q))))
    ob, _ = cam.get_obs()
    rgb = arr(ob["rgb"][..., :3]).astype(np.uint8)
    dep = arr(ob["depth_linear"]).astype(float)
    R = quat_to_mat(q)                    # camera-to-world
    fwd, right, up = -R[:, 2], R[:, 0], R[:, 1]
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for nm, o in objs.items():
        d3 = o["ctr3"] - eye
        zc = float(d3 @ fwd)
        if zc < 0.3 or zc > 20:
            continue
        u = CX + FX * float(d3 @ right) / zc
        v = CY - FY * float(d3 @ up) / zc
        if not (8 <= u < W - 8 and 8 <= v < H - 8):
            continue
        if not visible_ray(eye, o["ctr3"], nm):
            continue
        patch = dep[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            continue
        err = float(np.median(patch)) - zc
        resid.append(err)
        proj_n += 1
        good = abs(err) <= a.tol
        proj_ok += good
        dr.ellipse([u - 9, v - 9, u + 9, v + 9],
                   outline=(0, 255, 0) if good else (255, 0, 0), width=3)
        # anchor yaw inversion, exactly the SPEC section-5 check
        samples.append((math.degrees(math.atan2(o["ctr3"][1] - eye[1], o["ctr3"][0] - eye[0])),
                        math.degrees(math.atan((u - CX) / FX)), yaw))
    if k < 6:
        dr.line([0, CY, W, CY], fill=(255, 255, 0), width=1)
        dr.text((10, 10), "yaw=%.0f pitch=%.0f" % (yaw, a.pitch), fill=(255, 255, 0))
        im.save(os.path.join(a.out, "%03d_yaw%03d.jpg" % (k, int(yaw))), quality=88)
    if (k + 1) % 10 == 0:
        print("  pose %d/%d proj %d/%d" % (k + 1, a.n, proj_ok, proj_n), flush=True)

best = None
for sign in (-1, 1):
    qq = np.array([((b - sign * s - y + 180) % 360) - 180 for b, s, y in samples])
    if not len(qq):
        continue
    off = float(np.median(qq))
    for _ in range(3):
        r = ((qq - off + 180) % 360) - 180
        off = float(((off + np.median(r) + 180) % 360) - 180)
    r = ((qq - off + 180) % 360) - 180
    s_ = float(np.median(abs(r)))
    if best is None or s_ < best[0]:
        best = (s_, sign, off, float(np.median(r)))

rep = dict(scene=a.scene, poses=a.n, pitch=a.pitch,
           intrinsics=dict(W=W, H=H, focal=FOC, h_aperture=HAP, fx=round(FX, 3)),
           quat_readback_max_err=round(max(readback_err), 6),
           projection_vs_depth=dict(
               n=proj_n, within_tol=proj_ok, frac=round(proj_ok / max(proj_n, 1), 4),
               median_signed_err_m=round(float(np.median(resid)), 4) if resid else None,
               median_abs_err_m=round(float(np.median(np.abs(resid))), 4) if resid else None),
           yaw_audit=dict(anchors=len(samples),
                          screen_x_sign=best[1] if best else None,
                          yaw_offset_deg=round(best[2], 4) if best else None,
                          median_abs_error_deg=round(best[0], 4) if best else None,
                          median_signed_error_deg=round(best[3], 4) if best else None))
json.dump(rep, open(os.path.join(a.out, "report_%s_pitch%d.json" % (a.scene, int(a.pitch))), "w"), indent=1)
print(json.dumps(rep), flush=True)
print("POSE_VERIFY_DONE", flush=True)
