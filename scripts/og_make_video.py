#!/usr/bin/env python3
"""Render an episode to mp4 with the GT drawn on top, so the walk and the analytic
projection can be checked by eye in one pass.

    python scripts/og_make_video.py data/og20/house_0000 out.mp4 [--fps 8]

Overlay: frame index / room / yaw, one box+label per visible object (green = the
rendered depth agrees with the projected depth, amber = it does not), a red banner
on the frames where a move fires, and a small top-down map with the walked path.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("house")
ap.add_argument("out")
ap.add_argument("--fps", type=int, default=8)
ap.add_argument("--tol", type=float, default=0.6)
ap.add_argument("--map-frames", action="store_true", help="append the mapping walk")
a = ap.parse_args()

gt = json.load(open(os.path.join(a.house, "gt.json")))
live = gt["live"]
types = {n: v["type"] for n, v in gt["gt0"].items()}
moves = {m["t"]: m for m in gt["moves"]}
path = np.array([f["apos"] for f in live], float)
lo, hi = path.min(0) - 1.0, path.max(0) + 1.0
MW = 240


def to_map(p):
    s = (MW - 20) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    return (10 + (p[0] - lo[0]) * s, MW - 10 - (p[1] - lo[1]) * s)


def draw(frame, src, idx):
    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    for n, b in frame.get("box", {}).items():
        err = frame.get("occl", {}).get(n)
        ok = err is None or abs(err) <= a.tol
        col = (60, 230, 90) if ok else (245, 180, 40)
        d.rectangle(b, outline=col, width=2)
        lab = "%s %.1fm" % (types.get(n, n), frame["dist"].get(n, 0))
        d.rectangle([b[0], max(0, b[1] - 14), b[0] + 8 * len(lab), max(0, b[1] - 14) + 14],
                    fill=(0, 0, 0))
        d.text((b[0] + 2, max(0, b[1] - 13)), lab, fill=col)
    # SPEC 4-4.4: the source is 1 fps, so playback speed is a multiplier -- say so on
    # every frame, otherwise the walk reads as far faster than it is.
    hdr = "t=%d  %s  yaw=%.0f  vis=%d  [%s]   |  1fps source @ %dfps = %dx speed" % (
        frame["t"], frame["room"], frame["yaw"], len(frame["vis"]), frame.get("tag", ""),
        a.fps, a.fps)
    d.rectangle([0, 0, 8 * len(hdr) + 12, 22], fill=(0, 0, 0))
    d.text((6, 5), hdr, fill=(255, 255, 0))
    m = moves.get(frame["t"])
    if m:
        txt = "MOVE %s: %s  %s -> %s  (%s)" % (
            m["case"], types.get(m["oid"], m["oid"]), m["frm"], m["to"], m["kind"])
        d.rectangle([0, 26, 10 * len(txt) + 12, 52], fill=(180, 0, 0))
        d.text((6, 32), txt, fill=(255, 255, 255))
    # top-down map with the path so far
    W, H = im.size
    ox, oy = W - MW - 10, H - MW - 10
    d.rectangle([ox, oy, ox + MW, oy + MW], fill=(12, 12, 12), outline=(90, 90, 90))
    pts = [to_map(p) for p in path[:idx + 1]]
    if len(pts) > 1:
        d.line([(ox + x, oy + y) for x, y in pts], fill=(80, 160, 255), width=2)
    cx, cy = to_map(frame["apos"])
    d.ellipse([ox + cx - 4, oy + cy - 4, ox + cx + 4, oy + cy + 4], fill=(255, 60, 60))
    yr = math.radians(frame["yaw"])
    d.line([ox + cx, oy + cy, ox + cx + 18 * math.cos(yr), oy + cy - 18 * math.sin(yr)],
           fill=(255, 60, 60), width=2)
    return im


with tempfile.TemporaryDirectory() as tmp:
    k = 0
    if a.map_frames:
        for i, mp in enumerate(gt["map"]):
            src = os.path.join(a.house, "map", "%04d.jpg" % i)
            if not os.path.exists(src):
                continue
            im = Image.open(src).convert("RGB")
            d = ImageDraw.Draw(im)
            for n, b in mp.get("box", {}).items():
                d.rectangle(b, outline=(120, 200, 255), width=2)
                d.text((b[0] + 2, max(0, b[1] - 13)), types.get(n, n), fill=(120, 200, 255))
            hdr = "MAPPING WALK  %s  yaw=%.0f" % (mp["room"], mp["yaw"])
            d.rectangle([0, 0, 8 * len(hdr) + 12, 22], fill=(0, 0, 90))
            d.text((6, 5), hdr, fill=(255, 255, 255))
            im.save(os.path.join(tmp, "%06d.jpg" % k), quality=90)
            k += 1
    for i, f in enumerate(live):
        src = os.path.join(a.house, "live", "%06d.jpg" % f["t"])
        if not os.path.exists(src):
            continue
        draw(f, src, i).save(os.path.join(tmp, "%06d.jpg" % k), quality=90)
        k += 1
    if not k:
        raise SystemExit("no frames found under " + a.house)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", os.path.join(tmp, "%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                   check=True)
print("wrote %s (%d frames @ %d fps)" % (a.out, k, a.fps))
