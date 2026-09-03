#!/usr/bin/env python3
"""Scripted OmniGibson/BEHAVIOR generator for the og20 GT schema (one house per process).

Every departure from the 300-frame smoke build is backed by a measurement taken on
this node (Isaac Sim 5.1.0-rc.19 / Kit 107.3.1 / OmniGibson 3.9.2, GPU1):

* No segmentation modalities.  scripts/og_render_probe.py, 150 horizontal poses per
  config: rgb 150/150, rgb+depth_linear 150/150, rgb+seg_semantic 6, +seg_instance 7,
  all three 4.  The crash is a segfault inside omni.syntheticdata's on-demand
  post-process OmniGraph (SyntheticData._post_process_graph_tick -> graph.evaluate())
  and it fires even when get_obs() is never called, so it is the annotator graph and
  not the fetch.  Visibility therefore comes from PhysX rays plus analytic projection,
  which is what PORTING_CHECKLIST / SPEC 4.4 asks for anyway.

* Real horizontal camera.  A USD camera looks down its local -Z with local +Y up, so
  in OmniGibson's z-up stage the old [0,0,sin,cos] quaternion stared at the floor --
  every "stable" frame was a top-down carpet shot, which is why the yaw audit sat at
  ~50 deg.  BASE maps local -Z to world +X and local +Y to world +Z.  Measured after
  the fix: yaw median abs error 0.008 deg over 820 anchors, screen_x_sign -1.

* Real navmesh.  The shipped floor_trav_*.png maps put almost every room in its own
  connected component (Rs_int: 5 rooms, 6 components, no component covering more than
  2 rooms), so scene.get_shortest_path() returned None and the smoke build fell back
  to straight-line interpolation.  scripts/og_navgrid.py raycasts the live scene
  (Rs_int: 5/5 rooms in one component) and its routes are arc-length resampled.
  Measured: 0.242 m median step, 0.307 m max, zero jumps over 0.5 m.  There is no
  teleport or direct-interpolation fallback in this file.

* Heading rate is capped (--max-turn-deg).  Before the cap, 33% of live frames turned
  more than 45 deg and the worst was 175 deg, which reads as a strobe rather than a
  head turn and breaks any VO/SfM pose chain.

* Placement honesty (SPEC 4.4 / section 125).  Moves go through OnTop/Inside
  .set_value(), which samples, settles with physics and verifies the predicate; GT
  coordinates are re-read from the sim and the destination room is re-derived from
  them.  Each plan entry carries alternates because 3 of 4 single-target attempts
  failed sampling.  Every move is gated on support plus a 2 m witness render, or, for
  a storage move, on being invisible from all twelve 2 m viewpoints.

* Props are injected (--props).  BEHAVIOR houses with a kitchen, a bathroom and 4-8
  rooms carry at most 9 unique-type non-fixed objects (Rs_int 9, Rs_garden 7, the rest
  <=4), so the 5+5+2 case budget is not reachable from scene content alone.  Unique
  categories are drawn from the 1829-category object dataset and placed OnTop of real
  receptacles before the mapping walk, so they earn exemplars like any other object.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--house", type=int, default=0)
ap.add_argument("--frames", type=int, default=2000,
                help="SPEC 2 wants >=1200; the --evidence device needs ~1900 "
                     "for a 5+5 case budget, so the default is 2000")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frames-max-factor", type=float, default=2.0,
                help="widen_budget: auto-extend --frames up to this factor when the scripted part needs more (v4 delta)")
ap.add_argument("--case1", type=int, default=10)
ap.add_argument("--case2", type=int, default=5)
ap.add_argument("--case3", type=int, default=5)
ap.add_argument("--case4", type=int, default=0,
                help="outdoor removal; 0 because garage/garden are out of scope")
ap.add_argument("--props", type=int, default=24, help="movable props to inject")
ap.add_argument("--eye", type=float, default=1.5)
ap.add_argument("--pitch", type=float, default=0.0)
ap.add_argument("--step", type=float, default=0.25, help="metres walked per live frame")
ap.add_argument("--max-turn-deg", type=float, default=45.0,
                help="cap on heading change between consecutive live frames")
ap.add_argument("--vis-range", type=float, default=20.0)
ap.add_argument("--depth-tol", type=float, default=0.6, help="SPEC 4.2 occlusion tolerance")
ap.add_argument("--min-rooms", type=int, default=4)
ap.add_argument("--max-rooms", type=int, default=8)
ap.add_argument("--need-types", default="kitchen,bathroom")
ap.add_argument("--map-sites", type=int, default=2)
ap.add_argument("--turns", type=int, default=8,
                help="headings per live dwell station (the map walk always uses 6)")
ap.add_argument("--diag-max", type=float, default=1.0, help="max AABB diagonal of a movable")
ap.add_argument("--evidence", default="3:1.4",
                help="SPEC 4-2.8  K:D -- after a case2 move, approach the object K "
                     "times from distance D, entering along the line of sight, with "
                     "another room in between so each counts as a separate visit")
ap.add_argument("--frames-max-factor", type=float, default=2.5,
                help="how far --frames may be auto-extended to fit the scripted part")
ap.add_argument("--spare", type=int, default=3,
                help="extra moves planned per case, so one gate failure does not "
                     "drop the house below quota")
ap.add_argument("--scan-deg", type=float, default=180.0,
                help="angular sweep of a live dwell station (the map walk always "
                     "sweeps the full 360 in six headings)")
ap.add_argument("--detour", type=float, default=0.85,
                help="probability that a visit routes via an extra waypoint, to keep "
                     "the episode walking rather than pivoting in place")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.frames < 1200 and not a.smoke:
    raise SystemExit("need --frames >=1200 (or --smoke)")
for d in ("live", "map", "witness"):
    os.makedirs(os.path.join(a.out, d), exist_ok=True)

import cv2
import numpy as np
from PIL import Image, ImageDraw

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True

env = og.Environment(configs=dict(
    scene=dict(type="InteractiveTraversableScene", scene_model=a.scene),
    robots=[], env=dict(action_frequency=30, physics_frequency=30)))
sc = env.scene

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import og_navgrid
from omnigibson.object_states import AABB, Inside, OnTop, Open
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.sampling_utils import raytest

frames_requested = a.frames
rng = np.random.default_rng(a.seed)
STRUCT = {"walls", "wall", "floors", "floor", "ceilings", "ceiling", "door", "window",
          "background", "groundplane", "unlabelled", "stairs", "staircase", "railing",
          "roof", "lawn", "driveway", "fence", "curtain", "rug", "carpet"}
CONTAINER_HINTS = ("cabinet", "drawer", "dresser", "wardrobe", "cupboard", "closet",
                   "chest", "nightstand", "sideboard", "bureau", "locker", "shelf")
# Garage, garden and other outdoor rooms are out of scope for this benchmark
# (user decision, 2026-09-02).  They are dropped from the room scope entirely, so no
# trajectory, GT room or move destination can land in one.  The direct consequence is
# that case4 ("carried out of the house") has no in-scope destination -- --case4
# defaults to 0 and the audit records why.
EXCLUDE_ROOM_TYPES = ("garage", "garden", "porch", "yard", "lawn", "balcony", "patio",
                      "outdoor", "driveway", "deck", "terrace")
NOT_A_SURFACE = ("switch", "socket", "outlet", "picture", "mirror", "painting", "poster",
                 "clock", "curtain", "towel", "vent", "radiator", "sconce", "railing",
                 "faucet", "showerhead", "handle", "knob", "thermostat")
PROP_POOL = ["mug", "bowl", "plate", "laptop", "water_bottle", "teapot", "hairbrush",
             "toothbrush", "hand_towel", "hat", "backpack", "umbrella", "banana", "apple",
             "orange", "wine_bottle", "alarm_clock", "vase", "pen", "screwdriver",
             "flashlight", "sunglasses", "wallet", "calculator", "headset", "notebook",
             "can", "saucepan", "frying_pan", "kettle", "toaster", "blender", "coffee_cup",
             "wineglass", "lunch_box", "briefcase", "tennis_racket", "baseball",
             "chess_set", "board_game", "magazine", "newspaper", "folder", "stapler",
             "hammer", "paintbrush", "watering_can", "picture_frame", "radio", "tablet",
             "keyboard"]
notes = []


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def room_at(xy):
    r = sc.seg_map.get_room_instance_by_point(np.asarray(xy, float)[:2])
    return str(r) if r is not None else "outside"


def pos_std(p):
    """OG z-up (x,y,z) -> exported y-up [x, height, depth]; matches the THOR/hab schema."""
    p = np.asarray(p, float)
    return [round(float(p[0]), 3), round(float(p[2]), 3), round(float(p[1]), 3)]


def apos(p):
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def rtype_of(r):
    return r.rsplit("_", 1)[0].replace("_", " ")


# ---------------------------------------------------------------- doors + navmesh
door_report = og_navgrid.open_doors_for_clearance(sc)
for _ in range(90):
    og.sim.step_physics()               # settle before anything is measured
og.sim.render()
ng = og_navgrid.build(sc)
print("navgrid %s" % json.dumps(ng.summary()), flush=True)
reach = [r for r in ng.rooms if not any(h in rtype_of(r) for h in EXCLUDE_ROOM_TYPES)]
dropped_outdoor = [r for r in ng.rooms if r not in reach]
if dropped_outdoor:
    notes.append("outdoor rooms dropped from scope: %s" % dropped_outdoor)
need = [t for t in a.need_types.split(",") if t]
have = {rtype_of(r) for r in reach}
missing = [t for t in need if t not in have]
if len(reach) < a.min_rooms or missing:
    msg = ("scene %s: %d rooms in one navmesh component (need >=%d), missing room types "
           "%s; isolated %s" % (a.scene, len(reach), a.min_rooms, missing, ng.isolated))
    if not a.smoke:
        raise SystemExit(msg)
    notes.append(msg)

# Trim to --max-rooms: grow a connected room set outward from the required types, so a
# 17-room scene still yields a 4-8 room house instead of being discarded.
rooms = reach
if len(reach) < len(ng.rooms):
    ng.restrict(reach)
    rooms = list(ng.rooms)
    reach = list(ng.rooms)
if len(reach) > a.max_rooms:
    masks = {}
    for r in reach:
        m = np.zeros(ng.mask.shape, np.uint8)
        for j, i in ng.room_cells(r):
            m[j, i] = 255
        masks[r] = m
    grown = {r: cv2.dilate(m, np.ones((7, 7), np.uint8)) for r, m in masks.items()}
    adj = defaultdict(set)
    for x in reach:
        for y in reach:
            if x != y and np.logical_and(grown[x] > 0, masks[y] > 0).any():
                adj[x].add(y)
    seeds = [r for r in reach if rtype_of(r) in need]
    chosen = list(dict.fromkeys(seeds))[:a.max_rooms] or [reach[0]]
    while len(chosen) < a.max_rooms:
        front = sorted({y for x in chosen for y in adj[x]} - set(chosen))
        if not front:
            break
        chosen.append(front[int(rng.integers(len(front)))])
    rooms = sorted(chosen)
    notes.append("scene has %d connected rooms; episode restricted to %d: %s"
                 % (len(reach), len(rooms), rooms))
    ng.restrict(rooms)
rtype = {r: rtype_of(r) for r in rooms}
print("rooms in scope (%d): %s" % (len(rooms), rooms), flush=True)

# ---------------------------------------------------------------- camera
cam = og.sim.viewer_camera
cam.add_modality("depth_linear")        # rgb + depth is the only stable pair here
for _ in range(4):
    og.sim.render()
W, H = int(cam.image_width), int(cam.image_height)
FX = W * float(cam.focal_length) / float(cam.horizontal_aperture)
FY = FX                                  # square pixels
CX, CY = W / 2.0, H / 2.0
BASE = np.array([0.5, -0.5, -0.5, 0.5])  # local -Z -> world +X, local +Y -> world +Z


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


def cam_axes(yaw_deg, pitch_deg=0.0):
    x, y, z, w = cam_quat(yaw_deg, pitch_deg)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return -R[:, 2], R[:, 0], R[:, 1]    # forward, right, up


# ---------------------------------------------------------------- inventory
def aabb_of(o):
    try:
        lo, hi = o.states[AABB].get_value()
        return arr(lo).astype(float), arr(hi).astype(float)
    except Exception:
        try:
            lo, hi = o.aabb
            return arr(lo).astype(float), arr(hi).astype(float)
        except Exception:
            return None


objs = {}


def index(o, injected=False):
    box = aabb_of(o)
    if box is None:
        return None
    lo, hi = box
    ctr = (lo + hi) / 2
    cat = str(getattr(o, "category", o.name)).lower()
    objs[str(o.name)] = dict(obj=o, cat=cat, lo=lo, hi=hi, ctr=ctr, room=room_at(ctr),
                             struct=cat in STRUCT,
                             fixed=bool(getattr(o, "fixed_base", False)),
                             diag=float(np.linalg.norm(hi - lo)), injected=injected)
    return str(o.name)


for o in list(sc.objects):
    index(o)
if not objs:
    raise SystemExit("no objects with a readable AABB")


def refresh(n):
    v = objs[n]
    box = aabb_of(v["obj"])
    if box is not None:
        v["lo"], v["hi"] = box
        v["ctr"] = (box[0] + box[1]) / 2
        v["room"] = room_at(v["ctr"])
    return v


def top_is_flat(n):
    """Cast down onto the object's top face; a wall plate returns a vertical normal."""
    v = objs[n]
    c = (v["lo"] + v["hi"]) / 2
    r = raytest([float(c[0]), float(c[1]), float(v["hi"][2]) + 0.30],
                [float(c[0]), float(c[1]), float(v["hi"][2]) - 0.05])
    if not r.get("hit"):
        return False
    nrm = r.get("normal")
    return True if nrm is None else float(arr(nrm)[2]) > 0.7


def build_surfaces():
    """room -> objects with a real horizontal top, and room -> closable containers."""
    recept, contain = defaultdict(list), defaultdict(list)
    for n, v in objs.items():
        if v["struct"] or v["room"] not in rooms or v.get("injected"):
            continue
        top_z = float(v["hi"][2]) - float(ng.z0)
        foot = float(v["hi"][0] - v["lo"][0]) * float(v["hi"][1] - v["lo"][1])
        if (0.30 <= top_z <= 1.35 and foot >= 0.09 and v["diag"] >= 0.4
                and not any(h in v["cat"] for h in NOT_A_SURFACE) and top_is_flat(n)):
            recept[v["room"]].append(n)
        if any(h in v["cat"] for h in CONTAINER_HINTS) and getattr(v["obj"], "n_joints", 0):
            contain[v["room"]].append(n)
    return recept, contain


receptacles, containers = build_surfaces()
print("receptacles %s | containers %s"
      % ({k: len(v) for k, v in receptacles.items()},
         {k: len(v) for k, v in containers.items()}), flush=True)

# ---------------------------------------------------------------- prop injection
present = {v["cat"] for v in objs.values()}
pool = [c for c in PROP_POOL
        if c not in present and os.path.isdir(os.path.join(
            get_dataset_path("behavior-1k-assets"), "objects", c))]
rng.shuffle(pool)
rooms_with_surface = [r for r in rooms if receptacles.get(r)]
prop_report = dict(requested=a.props, categories_available=len(pool), placed=0, failed=[])
if a.props and rooms_with_surface:
    stage_cells = []
    for r in rooms_with_surface:
        cs = ng.room_cells(r)
        stage_cells += [cs[i] for i in range(0, len(cs), max(1, len(cs) // 8))]
    rng.shuffle(stage_cells)
    made, staged = [], {}
    for k, catg in enumerate(pool[:a.props]):
        try:
            o = DatasetObject(name="prop_%s_%d" % (catg, k), category=catg, fixed_base=False)
            sc.add_object(o)
            if stage_cells:
                sp = ng.to_world(stage_cells[k % len(stage_cells)])
                pos = [float(sp[0]), float(sp[1]), float(ng.z0) + 0.35 + 0.02 * k]
                o.set_position_orientation(position=pos, orientation=[0.0, 0.0, 0.0, 1.0])
                staged[str(o.name)] = pos
            made.append(o)
        except Exception as ex:
            prop_report["failed"].append(dict(category=catg, stage="load", err=str(ex)[:120]))
    # Only og.sim.step() drains og.sim._objects_to_initialize (Simulator._non_physics_step).
    # render()/step_physics() do not, and an uninitialized object in the registry makes
    # every states[...] call fail and later crashes og.sim.dump_state() with
    # "Object must be initialized before dumping state!".
    for _ in range(3):
        og.sim.step()
    pending = [str(o.name) for o in getattr(og.sim, "_objects_to_initialize", [])]
    if pending:
        raise SystemExit("props still uninitialized after 3 sim steps: %s" % pending[:5])

    def prop_pose_ok(o):
        """A single NaN pose makes every later og.sim.dump_state() raise, so screen
        each prop and re-seat or drop it before anything depends on the state."""
        try:
            pp, qq = o.get_position_orientation()
            if np.isfinite(arr(pp)).all() and np.isfinite(arr(qq)).all():
                return True
        except Exception:
            pass
        try:
            o.set_position_orientation(position=staged.get(str(o.name), [0.0, 0.0, 2.0]),
                                       orientation=[0.0, 0.0, 0.0, 1.0])
            o.keep_still()
            og.sim.step()
            pp, qq = o.get_position_orientation()
            return bool(np.isfinite(arr(pp)).all() and np.isfinite(arr(qq)).all())
        except Exception:
            return False

    alive = []
    for o in made:
        if prop_pose_ok(o):
            try:
                o.keep_still()
            except Exception:
                pass
            alive.append(o)
        else:
            prop_report["failed"].append(dict(category=str(o.category), stage="nan_pose"))
            objs.pop(str(o.name), None)
            try:
                sc.remove_object(o)
                og.sim.step()
            except Exception as ex:
                prop_report["failed"][-1]["remove_err"] = str(ex)[:100]
    made = alive
    for _ in range(30):
        og.sim.step_physics()
    for i, o in enumerate(made):
        room = rooms_with_surface[i % len(rooms_with_surface)]
        targets = list(receptacles[room])
        rng.shuffle(targets)
        ok, why = False, []
        for tname in targets[:5]:
            try:
                if o.states[OnTop].set_value(objs[tname]["obj"], True, use_trav_map=False):
                    ok = True
                    break
                why.append("onTop:%s:sampling" % tname)
            except Exception as ex:
                why.append("onTop:%s:%s" % (tname, str(ex)[:60]))
        if not ok:
            for cname in list(containers.get(room, []))[:3]:
                try:
                    if o.states[Inside].set_value(objs[cname]["obj"], True,
                                                  use_trav_map=False):
                        ok = True
                        break
                    why.append("inside:%s:sampling" % cname)
                except Exception as ex:
                    why.append("inside:%s:%s" % (cname, str(ex)[:60]))
        if ok:
            for _ in range(20):
                og.sim.step_physics()
            index(o, injected=True)
            prop_report["placed"] += 1
        elif prop_pose_ok(o):
            # Left where it was staged, resting on the floor.  That is a legitimate,
            # supported placement, and keeping it avoids another dangling-prim crash.
            index(o, injected=True)
            prop_report.setdefault("left_on_floor", 0)
            prop_report["left_on_floor"] += 1
            prop_report["failed"].append(dict(category=str(o.category),
                                              stage="place_fell_back_to_floor",
                                              room=room, why=why[:3]))
        else:
            prop_report["failed"].append(dict(category=str(o.category), stage="place",
                                              room=room, why=why[:4]))
            objs.pop(str(o.name), None)
    for _ in range(60):
        og.sim.step_physics()
    og.sim.render()
    for n in list(objs):
        refresh(n)
print("props %s" % json.dumps(prop_report), flush=True)

counts = Counter(v["cat"] for v in objs.values())
live_objs = [n for n, v in objs.items() if not v["struct"]]
print("objects %d (non-structural %d, unique-type %d)"
      % (len(objs), len(live_objs), sum(1 for c in counts.values() if c == 1)), flush=True)


def sample_points(n):
    """Centre plus face centres -- a single centre ray misses partly-hidden objects."""
    v = objs[n]
    c, lo, hi = v["ctr"], v["lo"], v["hi"]
    return [c,
            np.array([lo[0], c[1], c[2]]), np.array([hi[0], c[1], c[2]]),
            np.array([c[0], lo[1], c[2]]), np.array([c[0], hi[1], c[2]]),
            np.array([c[0], c[1], c[2] + (hi[2] - c[2]) * 0.8])]


def ray_sees(eye, target, name):
    r = raytest([float(v) for v in eye], [float(v) for v in target])
    if not r.get("hit"):
        return True
    L = float(np.linalg.norm(np.asarray(target, float) - np.asarray(eye, float)))
    body = "%s|%s" % (r.get("rigidBody", ""), r.get("collision", ""))
    return (name in str(body)) or (abs(float(arr(r["distance"])) - L) < 0.12)


def visible(eye, name):
    return any(ray_sees(eye, t, name) for t in sample_points(name))


# ---------------------------------------------------------------- capture
def capture(xy, yaw, kind, idx):
    eye = np.array([float(xy[0]), float(xy[1]), a.eye])
    cam.set_position_orientation(position=[float(v) for v in eye],
                                 orientation=[float(v) for v in cam_quat(yaw, a.pitch)])
    for _ in range(4):
        og.sim.render()
    ob, _ = cam.get_obs()
    rgb = arr(ob["rgb"][..., :3]).astype(np.uint8)
    dep = arr(ob["depth_linear"]).astype(float)
    fwd, right, up = cam_axes(yaw, a.pitch)

    vis, ctr, dist, box, occl = [], {}, {}, {}, {}
    for n in live_objs:
        v = objs[n]
        d3 = v["ctr"] - eye
        zc = float(d3 @ fwd)
        if zc < 0.25 or zc > a.vis_range:
            continue
        u = CX + FX * float(d3 @ right) / zc
        w = CY - FY * float(d3 @ up) / zc
        lo, hi = v["lo"], v["hi"]
        us, ws = [], []
        for sx in (lo[0], hi[0]):
            for sy in (lo[1], hi[1]):
                for sz in (lo[2], hi[2]):
                    q = np.array([sx, sy, sz]) - eye
                    z = float(q @ fwd)
                    if z < 0.1:
                        continue
                    us.append(CX + FX * float(q @ right) / z)
                    ws.append(CY - FY * float(q @ up) / z)
        if not us:
            continue
        x0, x1 = max(0, int(min(us))), min(W, int(max(us)) + 1)
        y0, y1 = max(0, int(min(ws))), min(H, int(max(ws)) + 1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        if not visible(eye, n):
            continue
        vis.append(n)
        ctr[n] = [round(float(u), 1), round(float(w), 1)]
        dist[n] = round(float(np.linalg.norm(v["ctr"][:2] - eye[:2])), 2)
        box[n] = [x0, y0, x1, y1]
        if 0 <= u < W and 0 <= w < H:      # SPEC 4.2 depth cross-check
            patch = dep[max(0, int(w) - 3):int(w) + 4, max(0, int(u) - 3):int(u) + 4]
            patch = patch[np.isfinite(patch) & (patch > 0)]
            if patch.size:
                occl[n] = round(float(np.median(patch)) - zc, 3)
    fn = "%06d.jpg" % idx if kind == "live" else "%04d.jpg" % idx
    Image.fromarray(rgb).save(os.path.join(a.out, kind, fn), quality=88)
    return dict(vis=vis, ctr=ctr, dist=dist, box=box, occl=occl, apos=apos(eye[:2]),
                yaw=round(float(yaw) % 360, 2), pitch=float(a.pitch), room=room_at(eye[:2]))


# ---------------------------------------------------------------- mapping walk
mapping, seen, mi = [], set(), 0
map_route_len = 0.0
cursor = ng.to_world(ng.room_point(rooms[0], rng))
for r in rooms:
    for _ in range(max(1, a.map_sites)):
        cell = ng.room_point(r, rng)
        if cell is None:
            continue
        p = ng.to_world(cell)
        leg = ng.route(cursor, p, step=a.step)
        if leg is None:
            notes.append("mapping walk could not reach a site in %s" % r)
            continue
        if len(leg) > 1:
            map_route_len += float(np.sum(np.linalg.norm(np.diff(leg, axis=0), axis=1)))
        cursor = p
        for yaw in range(0, 360, 60):      # SPEC 3: six headings, exactly
            rec = capture(p, yaw, "map", mi)
            mapping.append(dict(room=r, yaw=rec["yaw"], apos=rec["apos"], box=rec["box"],
                                ctr=rec["ctr"], dist=rec["dist"]))
            seen.update(rec["vis"])
            mi += 1
    print("mapping %s done (%d frames, %d objects seen)" % (r, mi, len(seen)), flush=True)
if not mapping:
    raise SystemExit("mapping walk produced no frames")

# ---------------------------------------------------------------- case planning
funnel = dict(non_structural=len(live_objs))
s1 = [n for n in live_objs if counts[objs[n]["cat"]] == 1]
funnel["unique_type"] = len(s1)
s2 = [n for n in s1 if not objs[n]["fixed"]]
funnel["not_fixed_base"] = len(s2)
s3 = [n for n in s2 if objs[n]["room"] in rooms]
funnel["in_scope_room"] = len(s3)
s4 = [n for n in s3 if n in seen]
funnel["seen_in_mapping_walk"] = len(s4)
movable = sorted(n for n in s4 if 0.03 < objs[n]["diag"] < a.diag_max)
funnel["carryable_size"] = len(movable)
funnel["injected_among_them"] = sum(1 for n in movable if objs[n].get("injected"))
print("movable funnel %s" % json.dumps(funnel), flush=True)
rng.shuffle(movable)
out_rooms = []          # nothing outdoor is in scope; kept so the case4 loop still reads


def far_room(r):
    """Farthest room, tie-broken towards the *least* walkable area.

    A big central room is a traffic hub, so an object parked there gets re-seen by
    accident and case3 collapses into case2 (measured on HSSD: 3 of 3 case3 roles
    degraded that way).  Walkable-cell count is the dwell proxy we have without an
    OmniGibson dwell prior.
    """
    c0 = ng.to_world(ng.room_point(r, rng))
    cand = [x for x in rooms if x != r]
    if not cand:
        return None
    area = {x: max(1, len(ng.room_cells(x))) for x in cand}
    amax = max(area.values())
    return max(cand, key=lambda x: (float(np.linalg.norm(ng.to_world(ng.room_point(x, rng)) - c0))
                                    * (1.0 - 0.5 * area[x] / amax)))


plan = []
avail = list(movable)


def take():
    return avail.pop(0) if avail else None


for _ in range(a.case3 + a.spare if a.case3 else 0):
    n = take()
    if n is None:
        notes.append("movable pool exhausted before the case3 quota")
        break
    src = objs[n]["room"]
    ext = objs[n]["hi"] - objs[n]["lo"]
    # A storage target must actually swallow the object: a laptop fits a top cabinet,
    # a standing TV does not -- that was one of the observed sampling failures.
    cand = [(r, c) for r in rooms if r != src for c in containers.get(r, [])
            if np.all((objs[c]["hi"] - objs[c]["lo"]) > ext + 0.06)]
    rng.shuffle(cand)
    if cand:
        plan.append(dict(case="case3", kind="storage", oid=n, frm=src, dest=cand[0][0],
                         into=cand[0][1],
                         alts=[dict(dest=r, into=c) for r, c in cand[:8]]))
    else:
        r = far_room(src)
        if r:
            alts = [dict(dest=r, onto=o) for o in receptacles.get(r, [])][:6]
            plan.append(dict(case="case3", kind="far_room", oid=n, frm=src, dest=r,
                             into=None, alts=alts or [dict(dest=r, onto=None)]))
for _ in range(a.case2 + a.spare if a.case2 else 0):
    n = take()
    if n is None:
        notes.append("movable pool exhausted before the case2 quota")
        break
    src = objs[n]["room"]
    cand = [(r, o) for r in rooms if r != src for o in receptacles.get(r, [])]
    rng.shuffle(cand)
    if not cand:
        continue
    plan.append(dict(case="case2", kind="ontop", oid=n, frm=src, dest=cand[0][0],
                     onto=cand[0][1], alts=[dict(dest=r, onto=o) for r, o in cand[:8]]))
for _ in range(a.case4):
    n = take()
    if n is None:
        notes.append("movable pool exhausted before the case4 quota")
        break
    src = objs[n]["room"]
    alts = [dict(dest=r, onto=o) for r in out_rooms if r != src
            for o in receptacles.get(r, [])]
    alts += [dict(dest=r, onto=None) for r in out_rooms if r != src]
    rng.shuffle(alts)
    if not alts:
        notes.append("case4 unavailable: outdoor rooms are out of scope by design, so "
                     "there is no in-scope destination outside the house")
        avail.insert(0, n)
        break
    plan.append(dict(case="case4", kind="ontop" if alts[0].get("onto") else "floor",
                     oid=n, frm=src, dest=alts[0]["dest"], onto=alts[0].get("onto"),
                     alts=alts[:8]))
moved_ids = {e["oid"] for e in plan}
statics = [n for n in live_objs if n not in moved_ids and n in seen
           and counts[objs[n]["cat"]] == 1 and objs[n]["room"] in rooms][:max(a.case1 * 3, 30)]
print("plan %d entries: %s" % (len(plan), json.dumps(
    [{k: v for k, v in e.items() if k != "alts"} for e in plan])), flush=True)

# ---------------------------------------------------------------- programme
program = []
cursor_xy = np.array(mapping[-1]["apos"], float)
route_fail = 0


def walk_to(target_xy):
    """One live frame per --step metres along the smoothed A* route.  No fallback."""
    global cursor_xy, route_fail
    if float(np.linalg.norm(np.asarray(target_xy, float) - cursor_xy)) < a.step * 0.75:
        return True                      # already there; do not emit a standstill frame
    leg = ng.route(cursor_xy, target_xy, step=a.step)
    if leg is None or len(leg) < 1:
        route_fail += 1
        return False
    if len(leg) < 2:
        cursor_xy = np.asarray(leg[-1], float)
        return True
    # The resample lands on total/round(total/step), which can exceed step; subdivide
    # so the continuity audit can never see a gap larger than the declared stride.
    dense = [np.asarray(leg[0], float)]
    for q in leg[1:]:
        q = np.asarray(q, float)
        gap = float(np.linalg.norm(q - dense[-1]))
        n = max(1, int(math.ceil(gap / (a.step * 1.05))))
        for k in range(1, n + 1):
            dense.append(dense[-1] + (q - dense[-1]) / (n - k + 1))
    leg = np.stack(dense)
    for k in range(len(leg)):
        nxt = leg[min(k + 1, len(leg) - 1)]
        d = nxt - leg[k]
        yaw = (math.degrees(math.atan2(d[1], d[0])) if float(np.linalg.norm(d)) > 1e-6
               else (program[-1]["yaw"] if program else 0.0))
        program.append(dict(p=np.asarray(leg[k], float), yaw=yaw, event=None, tag="walk"))
    cursor_xy = np.asarray(leg[-1], float)
    return True


def scan_here(turns=None, event=None, look_at=None, tag="scan", sweep=None):
    """Sweep @sweep degrees around the arrival heading in @turns steps."""
    turns = a.turns if turns is None else turns
    sweep = a.scan_deg if sweep is None else sweep
    start = float(program[-1]["yaw"]) if program else 0.0
    if look_at is not None:
        d = np.asarray(look_at, float)[:2] - cursor_xy
        start = math.degrees(math.atan2(d[1], d[0]))
    n = max(1, int(round(turns * sweep / 360.0)))
    if n == 1:
        sweep = 0.0                      # a single-frame station just holds the heading
    step = 0.0 if n == 1 else sweep / n
    sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    for k in range(n):
        program.append(dict(p=cursor_xy.copy(), yaw=start + sign * step * k,
                            event=event if k == 0 else None, tag=tag))


def visit(room, turns=None, event=None, look_at=None, tag="visit", detour=True,
          sweep=None):
    cell = ng.room_point(room, rng)
    if cell is None:
        return False
    if detour and a.detour > 0 and float(rng.random()) < a.detour:
        # An extra waypoint inside the target room lengthens the walk without adding
        # another stationary dwell, which is what pushes the moving-frame share up.
        via = ng.room_point(room, rng, centrality=0.0)
        if via is not None:
            walk_to(ng.to_world(via))
    if not walk_to(ng.to_world(cell)):
        return False
    scan_here(turns=turns, event=event, look_at=look_at, tag=tag, sweep=sweep)
    return True


def approach(target_xy, target_z=None, hold=5, tag="approach", radii=(1.3, 1.6, 1.9)):
    """Stand 1.3-1.9 m from @target_xy with line of sight and face it for @hold frames."""
    tgt = np.asarray(target_xy, float)[:2]
    aim = np.array([tgt[0], tgt[1], float(target_z) if target_z is not None
                    else float(ng.z0) + 0.8])
    for r in radii:
        order = list(range(0, 360, 20))
        rng.shuffle(order)
        for ang in order:
            rad = math.radians(ang)
            cell = ng.snap(tgt + r * np.array([math.cos(rad), math.sin(rad)]), max_r=6)
            if cell is None:
                continue
            spot = ng.to_world(cell)
            d2 = float(np.linalg.norm(spot - tgt))
            if not (0.7 <= d2 <= 2.0):
                continue
            eye = np.array([spot[0], spot[1], a.eye])
            if not ray_sees(eye, aim, "__none__") and float(np.linalg.norm(aim - eye)) > 0.3:
                # ray_sees with a bogus name reports False whenever something blocks
                # the line, which is exactly the test we want here.
                continue
            if not walk_to(spot):
                continue
            d = aim - eye
            base = math.degrees(math.atan2(d[1], d[0]))
            for k in range(max(1, hold)):
                program.append(dict(p=cursor_xy.copy(),
                                    yaw=base + (k - (hold - 1) / 2.0) * 8.0,
                                    event=None, tag=tag))
            return True
    return False



for r in rooms:
    if not visit(r, turns=6, tag="prelude"):
        raise SystemExit("prelude route to %s failed -- navmesh scope is wrong" % r)
prelude_end = len(program)

never_revisit = set()
for e in plan:
    if e["case"] == "case2":
        visit(e["dest"], turns=1, event=e, tag="case2-move")
    elif e["case"] == "case3":
        visit(e["frm"], event=e, tag="case3-move")
        if e["kind"] == "far_room":
            never_revisit.add(e["dest"])
    else:
        visit(e["frm"], event=e, tag="case4-move")
        never_revisit.add(e["dest"])


def evidence_goals(oid, K, D, crit=2.0, margin=0.25):
    """SPEC 4-2.8: K navigable points at distance D with line of sight, each planted as
    a (far, near) pair so the second leg is walked *towards* the object and the camera
    is looking at it on the way in.  Ported from hab_episode.evidence_goals.

    The accepted radius is clamped to crit - margin: a point at 2.05 m satisfies the
    old 1.4*D window but never the "within 2 m" criterion it exists to serve.
    """
    v = refresh(oid)
    tgt = v["ctr"]
    lo, hi = 0.6 * D, min(1.4 * D, crit - margin)
    cand = []
    start = float(rng.uniform(0.0, 360.0))
    for ang in np.linspace(start, start + 360.0, 36, endpoint=False):
        rad = math.radians(float(ang))
        dirv = np.array([math.cos(rad), math.sin(rad)])
        nc = ng.snap(tgt[:2] + D * dirv, max_r=8)
        fc = ng.snap(tgt[:2] + (D + 1.5) * dirv, max_r=12)
        if nc is None or fc is None:
            continue
        near, far = ng.to_world(nc), ng.to_world(fc)
        dn = float(np.linalg.norm(near - tgt[:2]))
        if not (lo <= dn <= hi):
            continue
        if float(np.linalg.norm(far - near)) < 0.8:
            continue
        if not ray_sees(np.array([near[0], near[1], a.eye]), tgt, "__none__"):
            continue
        cand.append((dn, far, near, tgt.copy()))
    cand.sort(key=lambda c: c[0])          # closest usable viewpoints first
    return [(f, n, t) for _, f, n, t in cand[:K]]


def sees_any(xy, names):
    eye = np.array([float(xy[0]), float(xy[1]), a.eye])
    return any(visible(eye, n) for n in names if n in objs)


def limit_turn_rate(prog, cap, seed=None):
    """Insert in-place frames so no two consecutive frames differ by more than cap.

    @seed is the previous phase's last rendered step, so the seam between phases is
    capped too (before seeding: max 111 deg, 7 frames over 45).
    """
    if not prog or cap <= 0:
        return prog
    out = [dict(p=np.asarray(seed["p"], float), yaw=float(seed["yaw"]),
                event=None, tag="seed")] if seed is not None else [prog[0]]
    prog = prog if seed is not None else prog[1:]
    for s in prog[1:]:
        d = ((float(s["yaw"]) - float(out[-1]["yaw"]) + 180.0) % 360.0) - 180.0
        n = int(math.ceil(abs(d) / cap)) - 1
        base_yaw = float(out[-1]["yaw"])
        base_p = np.asarray(out[-1]["p"], float).copy()
        for k in range(1, n + 1):
            out.append(dict(p=base_p.copy(), yaw=base_yaw + d * k / (n + 1),
                            event=None, tag="turn"))
        out.append(s)
    if seed is not None:
        out = out[1:]                    # the seed frame was already rendered
    return out


def widen_budget(need, what):
    """The scripted part must never be trimmed -- the first smoke build did that and
    lost every move (planned 3, moves 0).  Its length scales with the room count, so
    rather than hand-tuning --frames per scene, extend the budget and say so (v4 delta)."""
    if need <= a.frames:
        return
    if need > a.frames * a.frames_max_factor:
        raise SystemExit("%s needs %d frames, more than --frames=%d x %.1f; lower "
                         "--map-sites/--turns/--evidence/--spare or raise --frames"
                         % (what, need, a.frames, a.frames_max_factor))
    grown = int(need * 1.05)
    notes.append("frame budget auto-extended %d -> %d (%s needs %d, %d rooms in scope)"
                 % (a.frames, grown, what, need, len(rooms)))
    print("frame budget %d -> %d (%s needs %d)" % (a.frames, grown, what, need), flush=True)
    a.frames = grown


phase_a = limit_turn_rate(program, a.max_turn_deg)
program = []
widen_budget(len(phase_a), "prelude+moves")
allowed = [r for r in rooms if r not in never_revisit] or rooms
print("phase A (prelude+moves) %d frames (prelude %d, route failures %d)"
      % (len(phase_a), prelude_end, route_fail), flush=True)

# ---------------------------------------------------------------- move execution
moves, event_status = [], []


def supported(n):
    v = objs[n]
    c = v["ctr"]
    half = float(v["hi"][2] - v["lo"][2]) / 2
    r = raytest([float(c[0]), float(c[1]), float(c[2])],
                [float(c[0]), float(c[1]), float(c[2] - half - 0.25)])
    return bool(r.get("hit"))


def witness(n, k, t, radii=(2.0,), min_d=0.8):
    """Render the object with line of sight from one of @radii metres; SPEC 4.4 gate."""
    v = refresh(n)
    tgt = v["ctr"]
    for R in radii:
        for ang in range(0, 360, 30):
            rad = math.radians(ang)
            cell = ng.snap(tgt[:2] + R * np.array([math.cos(rad), math.sin(rad)]),
                           max_r=14)
            if cell is None:
                continue
            e2 = ng.to_world(cell)
            if float(np.linalg.norm(e2 - tgt[:2])) < min_d:
                continue
            eye = np.array([e2[0], e2[1], a.eye])
            if not visible(eye, n):
                continue
            d = tgt - eye
            yaw = math.degrees(math.atan2(d[1], d[0]))
            pitch = math.degrees(math.atan2(d[2], float(np.linalg.norm(d[:2]))))
            cam.set_position_orientation(
                position=[float(x) for x in eye],
                orientation=[float(x) for x in cam_quat(yaw, pitch)])
            for _ in range(4):
                og.sim.render()
            ob, _ = cam.get_obs()
            fwd, right, up = cam_axes(yaw, pitch)
            zc = float(d @ fwd)
            if zc < 0.2:
                continue
            u = CX + FX * float(d @ right) / zc
            w = CY - FY * float(d @ up) / zc
            if not (0 <= u < W and 0 <= w < H):
                continue
            im = Image.fromarray(arr(ob["rgb"][..., :3]).astype(np.uint8))
            ImageDraw.Draw(im).ellipse([u - 14, w - 14, u + 14, w + 14],
                                       outline=(255, 0, 0), width=3)
            rel = "witness/%02d_t%d_%s.jpg" % (k, t, objs[n]["cat"].replace(" ", "_"))
            im.save(os.path.join(a.out, rel), quality=88)
            return rel, [round(float(u), 1), round(float(w), 1)]
    return None, None


def hidden_everywhere(n):
    """A storage move must be invisible from every 2 m viewpoint once the door shuts."""
    v = refresh(n)
    for ang in range(0, 360, 30):
        rad = math.radians(ang)
        cell = ng.snap(v["ctr"][:2] + 2.0 * np.array([math.cos(rad), math.sin(rad)]),
                       max_r=14)
        if cell is None:
            continue
        e2 = ng.to_world(cell)
        if visible(np.array([e2[0], e2[1], a.eye]), n):
            return False
    return True


def apply_move(e, t):
    """Try each alternate target until one passes sampling and the geometry gate."""
    tried = []
    for alt in (e.get("alts") or [{}]):
        cand = dict(e)
        cand.update(alt)
        if "onto" in alt:
            cand["kind"] = "ontop" if alt["onto"] else "floor"
        if _apply_one(cand, t, tried):
            return True
    event_status.append(dict(t=t, oid=e["oid"], case=e["case"], kind=e["kind"],
                             applied=False, attempts=tried[:8]))
    return False


def _apply_one(e, t, tried):
    n = e["oid"]
    v = objs[n]
    before_room, before_pos = v["room"], v["ctr"].copy()
    st = og.sim.dump_state(serialized=False)
    target = e.get("into") or e.get("onto") or e["dest"]
    ok, err, store_wit = False, None, (None, None)
    try:
        if e["kind"] == "storage":
            box = objs[e["into"]]["obj"]
            if Open in box.states:
                box.states[Open].set_value(True)
                for _ in range(20):
                    og.sim.step_physics()
            ok = bool(objs[n]["obj"].states[Inside].set_value(box, True, use_trav_map=False))
            if ok:
                # SPEC 4-2.7 / section 125: the witness render happens while the
                # container is still open, so the object is actually in frame and
                # gen_selfcheck can score it.  Closing it afterwards is what makes the
                # object invisible for case3.
                for _ in range(20):
                    og.sim.step_physics()
                og.sim.render()
                store_wit = witness(n, len(moves), t,
                                    radii=(1.6, 1.2, 2.0, 2.5), min_d=0.5)
                if Open in box.states:
                    box.states[Open].set_value(False)
                    for _ in range(20):
                        og.sim.step_physics()
        elif e["kind"] == "ontop":
            ok = bool(objs[n]["obj"].states[OnTop].set_value(
                objs[e["onto"]]["obj"], True, use_trav_map=False))
        else:
            cell = ng.room_point(e["dest"], rng)
            if cell is not None:
                p = ng.to_world(cell)
                h = float(v["hi"][2] - v["lo"][2])
                objs[n]["obj"].set_position_orientation(
                    position=[float(p[0]), float(p[1]), float(ng.z0) + h / 2 + 0.02])
                for _ in range(40):
                    og.sim.step_physics()
                ok = True
    except Exception as ex:
        err = str(ex)[:160]
    for _ in range(30):
        og.sim.step_physics()
    og.sim.render()
    refresh(n)
    if not ok:
        og.sim.load_state(st, serialized=False)
        refresh(n)
        tried.append(dict(target=target, reason=err or "predicate sampling failed"))
        return False
    v = refresh(n)
    actual = v["room"]
    sup = True if e["kind"] == "storage" else supported(n)
    wit_file, wit_ctr, hidden = None, None, None
    if e["kind"] == "storage":
        hidden = hidden_everywhere(n)
    else:
        wit_file, wit_ctr = witness(n, len(moves), t,
                                    radii=(2.0, 1.6, 1.3, 2.5), min_d=0.6)
    if not (sup and (hidden is True or wit_file is not None)):
        og.sim.load_state(st, serialized=False)
        refresh(n)
        tried.append(dict(target=target,
                          reason="geometry gate (supported=%s witness=%s hidden=%s)"
                                 % (sup, wit_file is not None, hidden)))
        return False
    rec = dict(t=t, oid=n, frm=before_room, to=actual, case=e["case"], kind=e["kind"],
               pos=pos_std(v["ctr"]), from_pos=pos_std(before_pos), supported=bool(sup))
    if e["kind"] == "storage":
        rec.update(into=e["into"], into_type=objs[e["into"]]["cat"],
                   hidden_verified=bool(hidden), witness=store_wit[0] is not None,
                   witness_file=store_wit[0], witness_ctr=store_wit[1],
                   witness_taken="container_open")
    else:
        rec.update(witness=True, witness_file=wit_file, witness_ctr=wit_ctr)
    if actual != e["dest"]:
        rec["planned_dest"] = e["dest"]
        notes.append("%s landed in %s, planned %s (GT uses the measured room)"
                     % (n, actual, e["dest"]))
    moves.append(rec)
    event_status.append(dict(t=t, oid=n, case=e["case"], kind=e["kind"], applied=True,
                             to=actual, target=target, failed_attempts=tried[:8]))
    return True


# ---------------------------------------------------------------- run (three phases)
live = []


def run_steps(steps, limit=None):
    for st in steps:
        if limit is not None and len(live) >= limit:
            break
        t = len(live)
        if st["event"] is not None:
            apply_move(st["event"], t)
        rec = capture(st["p"], st["yaw"], "live", t)
        live.append(dict(t=t, room=rec["room"], vis=rec["vis"], ctr=rec["ctr"],
                         dist=rec["dist"], box=rec["box"], occl=rec["occl"],
                         anch={n: rec["ctr"][n] for n in rec["vis"] if n in statics},
                         apos=rec["apos"], yaw=rec["yaw"], pitch=rec["pitch"],
                         tag=st["tag"]))
        if len(live) % 100 == 0:
            print("live %d/%d" % (len(live), a.frames), flush=True)


# Phase A: establish every room, then fire the scripted moves.
run_steps(phase_a)

# Phase B: revisits, built now that the moves have run and the objects report real
# coordinates.  case2 walks into the room and closes to within 2 m of the object
# itself; case3 revisits the source room where the object no longer is.
program = []
approach_fail = []
EV_K, EV_D = (int(a.evidence.split(":")[0]), float(a.evidence.split(":")[1])) \
    if a.evidence else (0, 0.0)
hidden_oids = [m["oid"] for m in moves if m["case"] in ("case3", "case4")]
low_dwell = sorted(rooms, key=lambda r: len(ng.room_cells(r)))
for m in moves:
    if m["case"] != "case2":
        visit(m["frm"], turns=4, tag="case3-revisit")
        continue
    evs = evidence_goals(m["oid"], EV_K, EV_D) if EV_K else []
    if not evs:
        # fall back to a plain room visit plus the closest viewpoint we can find
        visit(m["to"], turns=4, tag="case2-revisit")
        v = refresh(m["oid"])
        if not approach(v["ctr"][:2], target_z=float(v["ctr"][2]), hold=8,
                        tag="case2-approach"):
            approach_fail.append(m["oid"])
            notes.append("no navigable viewpoint within 2 m of %s after its move"
                         % m["oid"])
        continue
    for j, (far, near, tgt) in enumerate(evs):
        walk_to(far)
        walk_to(near)                     # travel direction == line of sight
        d = tgt - np.array([cursor_xy[0], cursor_xy[1], a.eye])
        base = math.degrees(math.atan2(d[1], d[0]))
        for k in range(4):
            program.append(dict(p=cursor_xy.copy(), yaw=base + (k - 1.5) * 8.0,
                                event=None, tag="case2-evidence"))
        if j < len(evs) - 1:
            # another room in between, so the next one counts as a separate visit
            other = next((r for r in low_dwell if r != m["to"]), m["to"])
            visit(other, turns=3, tag="case2-between")
    m["evidence_visits"] = len(evs)
phase_b = limit_turn_rate(program, a.max_turn_deg,
                          seed=dict(p=np.asarray(live[-1]["apos"], float),
                                    yaw=live[-1]["yaw"]) if live else None)
program = []
widen_budget(len(live) + len(phase_b), "prelude+moves+revisits")
print("phase B (revisits) %d frames, approach failures %d"
      % (len(phase_b), len(approach_fail)), flush=True)
run_steps(phase_b)

# Phase C: fill the remaining budget with ordinary room-to-room walking, skipping any
# room a case3/case4 object was moved into.
guard = 0
while len(live) < a.frames and guard < 400:
    guard += 1
    before = len(program)
    for r in allowed:
        # SPEC 4-2.8: re-draw a filler viewpoint that would leak line of sight to an
        # object case3 says is out of sight, instead of quietly invalidating the case.
        placed = False
        for _try in range(30):
            cell = ng.room_point(r, rng)
            if cell is None:
                break
            if hidden_oids and sees_any(ng.to_world(cell), hidden_oids):
                continue
            if walk_to(ng.to_world(cell)):
                scan_here(turns=4, tag="filler")
                placed = True
            break
        if not placed:
            visit(r, turns=4, tag="filler")
        if len(live) + len(program) >= a.frames:
            break
    if len(program) == before:
        raise SystemExit("programme made no progress -- no reachable filler room")
    run_steps(limit_turn_rate(program, a.max_turn_deg,
                              seed=dict(p=np.asarray(live[-1]["apos"], float),
                                        yaw=live[-1]["yaw"]) if live else None),
              limit=a.frames)
    program = []
print("phase C done, %d live frames" % len(live), flush=True)

# ---------------------------------------------------------------- audit
def in_poly(xy, poly):
    """Crossing-number test; poly is a list of [x, y]."""
    if len(poly) < 3:
        return False
    x, y = float(xy[0]), float(xy[1])
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12):
            inside = not inside
    return inside


def _contour_polys(mask, size, res, eps, index_to_world):
    """Ordered outlines of @mask, largest first, simplified to @eps metres."""
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps / res, True)
    return [apos(index_to_world(int(pt[0][1]), int(pt[0][0]))) for pt in c]


def room_polygon(room, eps=0.15):
    """SPEC 4-2.5: the room's FLOOR extent as an ordered outline.

    The first version traced ng.cells_by_room, i.e. the *walkable* cells -- eroded by
    the body radius and with every furniture footprint carved out.  Objects sit on or
    against furniture, so they all fell outside it and the SPEC 4-2.4 containment test
    read gt0_pos 0.013 / moves_pos 0.0.  The room label itself comes from the shipped
    room instance raster, so the polygon has to come from the same raster.
    """
    sm = sc.seg_map
    iid = sm.room_ins_name_to_ins_id.get(room)
    if iid is None:
        return []
    ins = arr(sm.room_ins_map)
    size = float(ins.shape[0])
    res = float(sm.map_resolution)

    def idx_to_world(row, col):
        # world_to_map returns [y_index, x_index] (BaseMap flips), so invert that.
        return np.array([(col - size / 2.0) * res, (row - size / 2.0) * res])

    return _contour_polys((ins == iid).astype(np.uint8) * 255, size, res, eps, idx_to_world)


def walk_polygon(room, eps=0.12):
    """The walkable subset, kept separately -- useful, but not a room label."""
    m = np.zeros(ng.mask.shape, np.uint8)
    for j, i in ng.cells_by_room.get(room, []):
        m[j, i] = 255
    return _contour_polys(m, float(ng.mask.shape[0]), ng.res, eps,
                          lambda row, col: ng.to_world((row, col)))


room_polys = {r: room_polygon(r) for r in rooms}
walk_polys = {r: walk_polygon(r) for r in rooms}
# Self-check: the polygon must agree with the raster lookup that produced the labels.
poly_selfcheck = {}
for r in rooms:
    cells = ng.room_cells(r)[:400]
    if not cells or not room_polys[r]:
        poly_selfcheck[r] = None
        continue
    poly_selfcheck[r] = round(float(np.mean([in_poly(ng.to_world(c), room_polys[r])
                                             for c in cells])), 3)


def containment(pairs):
    """SPEC 4-2.4: fraction of (xy, room) pairs whose point falls in that room's polygon."""
    pairs = [(q, r) for q, r in pairs if r in room_polys and room_polys[r]]
    if not pairs:
        return None
    return round(float(np.mean([in_poly(q, room_polys[r]) for q, r in pairs])), 3)


gt0 = {n: dict(type=objs[n]["cat"], room=objs[n]["room"], pos=pos_std(objs[n]["ctr"]),
               injected=bool(objs[n].get("injected"))) for n in live_objs}
for m in moves:                          # gt0 is the pre-move world
    gt0[m["oid"]]["room"] = m["frm"]
    gt0[m["oid"]]["pos"] = m["from_pos"]

by_case = defaultdict(list)
for m in moves:
    by_case[m["case"]].append(m)
after = lambda n, t: [f for f in live[t + 1:] if n in f["vis"]]
ok2 = sum(1 for m in by_case["case2"]
          if sum(1 for f in live[m["t"] + 1:]
                 if m["oid"] in f["vis"] and f["dist"].get(m["oid"], 99) <= 2.0) >= 3)
ok3 = sum(1 for m in by_case["case3"]
          if not after(m["oid"], m["t"])
          and any(f["room"] == m["frm"] for f in live[m["t"] + 1:]))
ok4 = sum(1 for m in by_case["case4"] if not after(m["oid"], m["t"]))
ok1 = sum(1 for n in statics if any(n in f["vis"] for f in live)
          and len({f["t"] // 50 for f in live if f["room"] == gt0[n]["room"]}) >= 2)

samples = []
for f in live:
    p = np.asarray(f["apos"], float)
    for n, c in f["anch"].items():
        w = objs[n]["ctr"]
        samples.append((math.degrees(math.atan2(w[1] - p[1], w[0] - p[0])),
                        math.degrees(math.atan((c[0] - CX) / FX)), f["yaw"]))
best = None
for sign in (-1, 1):
    q = np.array([((b - sign * s - y + 180) % 360) - 180 for b, s, y in samples])
    if not len(q):
        continue
    off = float(np.median(q))
    for _ in range(3):
        r = ((q - off + 180) % 360) - 180
        off = float(((off + np.median(r) + 180) % 360) - 180)
    r = ((q - off + 180) % 360) - 180
    s_ = float(np.median(np.abs(r)))
    if best is None or s_ < best[0]:
        best = (s_, sign, off, float(np.median(r)))
yaw_audit = dict(anchors=len(samples), screen_x_sign=best[1] if best else None,
                 yaw_offset_deg=round(best[2], 4) if best else None,
                 median_abs_error_deg=round(best[0], 4) if best else None,
                 median_signed_error_deg=round(best[3], 4) if best else None)

pth = np.array([f["apos"] for f in live], float)
dpos = np.linalg.norm(np.diff(pth, axis=0), axis=1) if len(pth) > 1 else np.zeros(1)
yy = np.array([f["yaw"] for f in live], float)
dyaw = np.abs((np.diff(yy) + 180) % 360 - 180) if len(yy) > 1 else np.zeros(1)
occl_all = [v for f in live for v in f["occl"].values()]
audit = dict(house=a.house, scene=a.scene, frames=len(live),
             frames_requested=frames_requested, frames_budget=a.frames,
             map_frames=len(mapping),
             moves=len(moves), planned=len(plan),
             case1_no_move_revisited=ok1, case2_move_reobserved=ok2,
             case3_absent_belief=ok3, case4_outside=ok4,
             case3_storage=sum(1 for m in by_case["case3"] if m["kind"] == "storage"),
             mapped_objects=len(seen), statics=len(statics), movable_funnel=funnel,
             props=prop_report, rooms_in_scope=rooms, rooms_reachable=reach,
             case4_in_scope=False, outdoor_rooms_dropped=dropped_outdoor,
             rooms_isolated=ng.isolated, navgrid=ng.summary(), doors=door_report,
             route_failures=route_fail, map_route_len_m=round(map_route_len, 1),
             approach_failures=approach_fail,
             teleport_used=False, direct_interpolation_used=False,
             step_m=a.step, turns=a.turns, max_turn_deg=a.max_turn_deg,
             continuity=dict(moving_frame_frac=round(float(np.mean(dpos > 0.05)), 3),
                             walked_m=round(float(dpos.sum()), 1),
                             pos_step_median_m=round(float(np.median(dpos)), 3),
                             pos_step_max_m=round(float(dpos.max()), 3),
                             pos_jumps_over_0p5m=int((dpos > 0.5).sum()),
                             yaw_step_median_deg=round(float(np.median(dyaw)), 2),
                             yaw_step_max_deg=round(float(dyaw.max()), 2),
                             yaw_steps_over_45deg=int((dyaw > 45).sum())),
             depth_vs_projection=dict(
                 n=len(occl_all),
                 median_abs_m=round(float(np.median(np.abs(occl_all))), 3) if occl_all else None,
                 within_tol_frac=round(float(np.mean(np.abs(occl_all) <= a.depth_tol)), 4)
                 if occl_all else None),
             frame_tags={k: int(v) for k, v in
                         __import__("collections").Counter(f["tag"] for f in live).items()},
             field_convention=dict(
                 note="SPEC 4-2.4: every stored coordinate field measured by "
                      "point-in-polygon containment against scene_meta.polys",
                 polygon_selfcheck=poly_selfcheck,
                 gt0_pos=containment([([v["pos"][0], v["pos"][2]], v["room"])
                                      for v in gt0.values()]),
                 moves_pos=containment([([m["pos"][0], m["pos"][2]], m["to"])
                                        for m in moves]),
                 moves_from_pos=containment([([m["from_pos"][0], m["from_pos"][2]], m["frm"])
                                             for m in moves]),
                 live_apos=containment([(f["apos"], f["room"]) for f in live]),
                 map_apos=containment([(mp["apos"], mp["room"]) for mp in mapping]),
                 static_pos=containment([([v["pos"][0], v["pos"][2]], v["room"])
                                         for v in (gt0[n] for n in statics)])),
             evidence=dict(
                 spec=a.evidence,
                 visits={m["oid"]: m.get("evidence_visits", 0)
                         for m in moves if m["case"] == "case2"},
                 closest_m={m["oid"]: (round(min([f["dist"][m["oid"]]
                                                  for f in live[m["t"] + 1:]
                                                  if m["oid"] in f["vis"]] or [99.0]), 2))
                            for m in moves if m["case"] == "case2"}),
             yaw=yaw_audit, event_status=event_status, notes=notes)
# ---------------------------------------------------------------- export convention
# 검토(2026-09-02, 아이맥): 내부 yaw θ 는 OG 규약(0°=+og_x, 반시계)이고 감사의 screen_x_sign=-1 은
# 그 규약에서 "오른쪽=각도 감소" 라는 뜻이다. 우리 평가기(eval_online·georoom·initmap·검증)는
# bearing=atan2(dx,dz) · 0°=+z(=og_y) · 시계 증가 · β = ψ + atan((u-cx)/fx) 를 가정한다.
# 두 규약은 미러 없이 ψ = 90° − θ 로 일치한다(합성 검증 오차 0°). 내보낼 때만 변환한다.
def _yaw_ours(th): return round((90.0 - float(th)) % 360.0, 2)
for _f in live:
    _f["yaw_og"] = _f["yaw"]; _f["yaw"] = _yaw_ours(_f["yaw_og"])
for _m in mapping:
    _m["yaw_og"] = _m["yaw"]; _m["yaw"] = _yaw_ours(_m["yaw_og"])
_q = []
for _f in live:
    _p = np.asarray(_f["apos"], float)
    for _n, _c in _f["anch"].items():
        _w = objs[_n]["ctr"]
        _b = math.degrees(math.atan2(_w[0] - _p[0], _w[1] - _p[1]))       # 우리 bearing
        _s = math.degrees(math.atan((_c[0] - CX) / FX))
        _q.append(((_b - _s - _f["yaw"] + 180) % 360) - 180)
yaw_audit_ours = dict(anchors=len(_q), sign=+1,
                      median_abs_error_deg=round(float(np.median(np.abs(_q))), 4) if _q else None)
audit["yaw_ours"] = yaw_audit_ours
print("yaw_audit_ours %s" % json.dumps(yaw_audit_ours), flush=True)
if not a.smoke and _q and yaw_audit_ours["median_abs_error_deg"] > 0.5:
    raise SystemExit("GATE FAILED: exported-yaw convention error %.3f deg > 0.5" % yaw_audit_ours["median_abs_error_deg"])

payload = dict(house=a.house, rooms=[dict(id=r, type=rtype[r]) for r in rooms],
               room_types=rtype, gt0=gt0, moves=moves, live=live, map=mapping,
               fps=1.0, T=len(live),
               scene_meta=dict(
                   polys=room_polys, walk_polys=walk_polys,
                   doors=[], static={n: gt0[n] for n in statics},
                   coordinate_transform="OG z-up -> exported y-up: [x,y,z]=[og_x,og_z,og_y]; "
                                        "apos=[og_x,og_y]; yaw(exported)=90-yaw_og: 0 deg = +z(=og_y), cw+ ; "
                                        "bearing=atan2(dx,dz)=yaw+atan((u-cx)/fx)",
                   yaw_convention="ours", screen_x_sign=+1,
                   screen_x_sign_og=yaw_audit["screen_x_sign"], yaw_audit_ours=yaw_audit_ours,
                   intrinsics=dict(W=W, H=H, fx=round(FX, 3), cx=CX, cy=CY,
                                   focal_length=float(cam.focal_length),
                                   horizontal_aperture=float(cam.horizontal_aperture))),
               audit=audit)
json.dump(payload, open(os.path.join(a.out, "gt.json"), "w"), ensure_ascii=False)
json.dump(audit, open(os.path.join(a.out, "audit.json"), "w"), ensure_ascii=False, indent=2)
print(json.dumps(audit, ensure_ascii=False), flush=True)

fails = []
if not a.smoke:
    if len(live) < min(1200, frames_requested):
        fails.append("frames %d < %d" % (len(live), min(1200, frames_requested)))
    if not (a.min_rooms <= len(rooms) <= a.max_rooms):
        fails.append("%d rooms in scope, want %d..%d" % (len(rooms), a.min_rooms, a.max_rooms))
    for t in need:
        if t not in {rtype[r] for r in rooms}:
            fails.append("no %s in scope" % t)
    if ok1 < a.case1:
        fails.append("case1 %d < %d" % (ok1, a.case1))
    if ok2 < a.case2:
        fails.append("case2 %d < %d" % (ok2, a.case2))
    if ok3 < a.case3:
        fails.append("case3 %d < %d" % (ok3, a.case3))
    if ok4 < a.case4:
        fails.append("case4 %d < %d" % (ok4, a.case4))
    if yaw_audit["median_abs_error_deg"] is None or yaw_audit["median_abs_error_deg"] > 0.5:
        fails.append("yaw median abs error %s deg" % yaw_audit["median_abs_error_deg"])
    if route_fail:
        fails.append("%d routes failed" % route_fail)
    if int((dpos > 0.5).sum()):
        fails.append("%d position jumps over 0.5 m" % int((dpos > 0.5).sum()))
    if int((dyaw > a.max_turn_deg + 1.0).sum()):
        fails.append("%d frames turn more than %.0f deg (cap breached at a phase seam)"
                     % (int((dyaw > a.max_turn_deg + 1.0).sum()), a.max_turn_deg))
    if float(np.mean(dpos > 0.05)) < 0.55:
        fails.append("only %.0f%% of frames are walking (want >=55%%)"
                     % (100 * float(np.mean(dpos > 0.05))))
if fails:
    raise SystemExit("GATE FAILED: " + "; ".join(fails))
print("EPISODE_OK house=%d frames=%d" % (a.house, len(live)), flush=True)
