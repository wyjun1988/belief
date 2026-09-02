#!/usr/bin/env python3
"""Navmesh connectivity audit for BEHAVIOR scenes -- no Isaac Sim needed.

OmniGibson only ever loads floor_trav_<f>.png (doors as authored, usually shut) or
floor_trav_no_obj_<f>.png.  The assets also ship floor_trav_open_door_<f>.png.
This measures, per scene, how many room instances land outside the largest
connected component of each variant, which is what makes get_shortest_path()
return None between two rooms.
"""
import os, sys, json
import cv2, numpy as np

DATA = os.environ.get("OMNIGIBSON_DATA_PATH", "/mnt/ssd2/wooyeol/work/behavior1k_latest/og_data_2026")
ASSETS = os.path.join(DATA, "behavior-1k-assets")
SCENES = os.path.join(ASSETS, "scenes")
RES, DEFAULT_RES = 0.1, 0.01
ROOM_CATS = [l.rstrip() for l in open(os.path.join(ASSETS, "metadata", "room_categories.txt"))]
VARIANTS = ("floor_trav_0.png", "floor_trav_open_door_0.png",
            "floor_trav_no_door_0.png", "floor_trav_no_obj_0.png")

def load(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    img[img < 255] = 0
    # OmniGibson's _erode_trav_map builds a 0x0 kernel when default_erosion_radius==0,
    # which OpenCV treats as the default 3x3 rect -- reproduce that exactly.
    return cv2.erode(img, np.ones((3, 3), np.uint8))

def audit(scene):
    lay = os.path.join(SCENES, scene, "layout")
    base = cv2.imread(os.path.join(lay, "floor_trav_0.png"), cv2.IMREAD_GRAYSCALE)
    if base is None:
        return None
    size = int(base.shape[0] * DEFAULT_RES / RES)
    ins = cv2.resize(cv2.imread(os.path.join(lay, "floor_insseg_0.png"), cv2.IMREAD_GRAYSCALE),
                     (size, size), interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(cv2.imread(os.path.join(lay, "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE),
                     (size, size), interpolation=cv2.INTER_NEAREST)
    names, per_sem = {}, {}
    for iid in sorted(int(v) for v in np.unique(ins) if v != 0):
        ys, xs = np.where(ins == iid)
        sid = int(sem[ys[0], xs[0]])
        cat = ROOM_CATS[sid - 1]
        names[iid] = "%s_%d" % (cat, per_sem.get(cat, 0))
        per_sem[cat] = per_sem.get(cat, 0) + 1
    out = dict(scene=scene, n_rooms=len(names), rooms=sorted(names.values()))
    for v in VARIANTS:
        trav = load(os.path.join(lay, v), size)
        if trav is None:
            out[v] = None
            continue
        n, lab = cv2.connectedComponents(trav, connectivity=4)
        sizes = np.bincount(lab[trav > 0].ravel(), minlength=n)
        # The pixel-largest component is usually the EXTERIOR ground, not the interior:
        # Rs_int and Wainscott_0_int both score largest_frac ~0.5 with zero rooms in it.
        # Anchor on the component that covers the most room instances instead.
        per_room = {}
        for iid, nm in names.items():
            m = (ins == iid) & (trav > 0)
            per_room[nm] = np.bincount(lab[m].ravel(), minlength=n) if m.sum() >= 5 else np.zeros(n, int)
        cover = np.zeros(n, int)
        for c in per_room.values():
            cover += (c >= 5).astype(int)
        cover[0] = 0
        best = int(np.argmax(cover))
        reach = sorted(nm for nm, c in per_room.items() if c[best] >= 5)
        tiny = sorted(nm for nm, c in per_room.items() if c.sum() == 0)
        iso = sorted(set(names.values()) - set(reach) - set(tiny))
        out[v] = dict(components=int((sizes[1:] > 20).sum()), reachable=len(reach),
                      isolated=iso, no_free_space=tiny, best_component=best,
                      best_frac=round(float(sizes[best]) / max(int(trav.sum() // 255), 1), 3),
                      largest_frac=round(float(sizes[1:].max()) / max(int(trav.sum() // 255), 1), 3)
                      if len(sizes) > 1 else 0.0)
    return out

if __name__ == "__main__":
    todo = sys.argv[1:] or sorted(os.listdir(SCENES))
    rows = []
    for s in todo:
        try:
            r = audit(s)
        except Exception as e:
            print("%-28s ERROR %s" % (s, e)); continue
        if r is None:
            continue
        rows.append(r)
        pieces = []
        for v in VARIANTS:
            d = r[v]
            pieces.append("%s: -" % v.replace("floor_trav", "").replace("_0.png", "").strip("_") if d is None
                          else "%s: %d/%d ok comp=%d" % (v.replace("floor_trav", "").replace("_0.png", "").strip("_") or "base",
                                                          d["reachable"], r["n_rooms"], d["components"]))
        print("%-26s rooms=%-3d %s" % (s, r["n_rooms"], " | ".join(pieces)), flush=True)
    json.dump(rows, open(os.environ.get("OUT", "/mnt/ssd2/wooyeol/work/og_diag_20260902/navmesh_audit.json"), "w"), indent=1)
