#!/usr/bin/env python3
"""Cheap scene pre-screen (no Isaac): room roster from the layout maps + unique-type
object count from the scene json.  Keeps scenes that could satisfy
"4-8 rooms including a kitchen and a bathroom" before the live navgrid pass runs.
"""
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np

DATA = os.environ.get("OMNIGIBSON_DATA_PATH",
                      "/mnt/ssd2/wooyeol/work/behavior1k_latest/og_data_2026")
ASSETS = os.path.join(DATA, "behavior-1k-assets")
SCENES = os.path.join(ASSETS, "scenes")
CATS = [l.rstrip() for l in open(os.path.join(ASSETS, "metadata", "room_categories.txt"))]
STRUCT = {"walls", "wall", "floors", "floor", "ceilings", "ceiling", "door", "window",
          "background", "groundplane", "unlabelled", "stairs", "staircase", "railing",
          "roof", "lawn", "driveway", "fence", "curtain", "rug", "carpet"}

rows = []
for sd in sorted(os.listdir(SCENES)):
    lay = os.path.join(SCENES, sd, "layout")
    if not os.path.isdir(lay):
        continue
    ins = cv2.imread(os.path.join(lay, "floor_insseg_0.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(os.path.join(lay, "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
    if ins is None or sem is None:
        continue
    per, names = {}, []
    for iid in sorted(int(v) for v in np.unique(ins) if v):
        ys, xs = np.where(ins == iid)
        cat = CATS[int(sem[ys[0], xs[0]]) - 1]
        names.append("%s_%d" % (cat, per.get(cat, 0)))
        per[cat] = per.get(cat, 0) + 1
    types = set(per)
    j = glob.glob(os.path.join(SCENES, sd, "json", "*_best.json"))
    uniq = mov = 0
    if j:
        try:
            ii = json.load(open(j[0]))["objects_info"]["init_info"]
            cnt = Counter(str(v.get("args", {}).get("category", k)).lower()
                          for k, v in ii.items()
                          if str(v.get("args", {}).get("category", k)).lower() not in STRUCT)
            uniq = sum(1 for c, k in cnt.items() if k == 1)
            mov = sum(1 for k, v in ii.items()
                      if not v.get("args", {}).get("fixed_base")
                      and str(v.get("args", {}).get("category", k)).lower() not in STRUCT
                      and cnt[str(v.get("args", {}).get("category", k)).lower()] == 1)
        except Exception:
            pass
    rows.append(dict(scene=sd, n_rooms=len(names), rooms=names,
                     kitchen="kitchen" in types,
                     bathroom="bathroom" in types,
                     outdoor=sorted(t for t in types
                                    if any(h in t for h in ("garage", "garden", "porch",
                                                            "yard", "balcony", "patio"))),
                     uniq_type=uniq, movable_uniq=mov))

keep = [r for r in rows if r["kitchen"] and r["bathroom"] and 4 <= r["n_rooms"] <= 12]
keep.sort(key=lambda r: (-r["movable_uniq"], r["n_rooms"]))
print("%-28s %6s %-9s %-9s %8s %8s  %s"
      % ("scene", "rooms", "kitchen", "bathroom", "uniqTyp", "movUniq", "outdoor"))
for r in keep:
    print("%-28s %6d %-9s %-9s %8d %8d  %s"
          % (r["scene"], r["n_rooms"], r["kitchen"], r["bathroom"],
             r["uniq_type"], r["movable_uniq"], ",".join(r["outdoor"]) or "-"))
print("\ncandidates with kitchen+bathroom and 4..12 rooms: %d / %d" % (len(keep), len(rows)))
print("\nrejected (no kitchen or no bathroom or room count out of range):")
for r in rows:
    if r not in keep:
        why = []
        if not r["kitchen"]:
            why.append("no kitchen")
        if not r["bathroom"]:
            why.append("no bathroom")
        if not (4 <= r["n_rooms"] <= 12):
            why.append("rooms=%d" % r["n_rooms"])
        print("  %-28s %s" % (r["scene"], "; ".join(why)))
json.dump(rows, open(os.environ.get("OUT", "/mnt/ssd2/wooyeol/work/og_diag_20260902/prescreen.json"), "w"), indent=1)
