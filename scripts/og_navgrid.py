#!/usr/bin/env python3
"""Live traversability grid for BEHAVIOR scenes, built with PhysX raycasts.

The shipped floor_trav_*.png maps put nearly every room in its own connected
component (Rs_int: 5 rooms / 6 components, only 2 rooms coverable by any single
component), so scene.get_shortest_path() cannot route between rooms and the
generator falls back to teleporting.  Raycasting the live physics scene instead
recovers the real walkable region (Rs_int: 5/5 rooms in one component).

Importable helper -- `build(scene)` returns a NavGrid used by scripts/og_episode.py.
Run directly to screen scenes:  python scripts/og_navgrid.py [scene ...]
"""
import heapq
import math

import cv2
import numpy as np

import omnigibson as og
from omnigibson.utils.sampling_utils import raytest

DEFAULT_RES = 0.05
DEFAULT_HEAD = 1.75        # standing clearance the walker needs
DEFAULT_CLEARANCE = 0.25   # lateral body radius
FLOOR_TOL = 0.12


def _arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class NavGrid:
    """Walkable occupancy grid + A* over the component that covers the most rooms."""

    def __init__(self, scene, res=DEFAULT_RES, head=DEFAULT_HEAD,
                 clearance=DEFAULT_CLEARANCE, floor=0):
        self.scene = scene
        self.res = float(res)
        self.floor = int(floor)
        tm = scene.trav_map
        self.half = tm.map_size * tm.map_resolution / 2.0
        self.z0 = float(tm.floor_heights[self.floor])
        self.xs = np.arange(-self.half, self.half, self.res)
        self.ys = np.arange(-self.half, self.half, self.res)

        free = np.zeros((len(self.ys), len(self.xs)), np.uint8)
        for j, y in enumerate(self.ys):
            for i, x in enumerate(self.xs):
                # One ray down from head height: it reaches the floor only when nothing
                # (wall, furniture, closed or swung-open door leaf) occupies the column.
                r = raytest([float(x), float(y), self.z0 + head],
                            [float(x), float(y), self.z0 - 0.4])
                if r.get("hit") and abs(float(_arr(r["position"])[2]) - self.z0) <= FLOOR_TOL:
                    free[j, i] = 255
        self.free = free
        k = max(1, int(round(clearance / self.res)) * 2 + 1)
        self.walk = cv2.erode(free, np.ones((k, k), np.uint8))

        # Room label per walkable cell, straight from the shipped room instance map.
        self.room_of = {}
        self.cells_by_room = {}
        for j, y in enumerate(self.ys):
            for i, x in enumerate(self.xs):
                if not self.walk[j, i]:
                    continue
                r = scene.seg_map.get_room_instance_by_point(np.array([float(x), float(y)]))
                if r is None:
                    continue
                self.room_of[(j, i)] = str(r)
                self.cells_by_room.setdefault(str(r), []).append((j, i))

        n, lab = cv2.connectedComponents(self.walk, connectivity=4)
        self.ncomp, self.lab = n, lab
        cover = np.zeros(n, int)
        self.room_comp_hist = {}
        for room, cells in self.cells_by_room.items():
            hist = np.bincount(np.array([lab[j, i] for j, i in cells]), minlength=n)
            self.room_comp_hist[room] = hist
            cover += (hist >= 4).astype(int)
        cover[0] = 0
        # The pixel-largest component is often the exterior ground: anchor on the one
        # that covers the most room instances instead.
        self.comp = int(np.argmax(cover)) if n > 1 else 0
        self.rooms = sorted(r for r, h in self.room_comp_hist.items() if h[self.comp] >= 4)
        self.isolated = sorted(set(self.room_comp_hist) - set(self.rooms))
        self.mask = (self.walk > 0) & (lab == self.comp)

    # -- coordinate helpers -------------------------------------------------
    def to_cell(self, xy):
        i = int(round((float(xy[0]) + self.half) / self.res))
        j = int(round((float(xy[1]) + self.half) / self.res))
        return (max(0, min(len(self.ys) - 1, j)), max(0, min(len(self.xs) - 1, i)))

    def to_world(self, cell):
        j, i = cell
        return np.array([self.xs[i], self.ys[j]], float)

    def snap(self, xy, max_r=40):
        """Nearest cell inside the chosen component."""
        j0, i0 = self.to_cell(xy)
        if self.mask[j0, i0]:
            return (j0, i0)
        for r in range(1, max_r):
            js = slice(max(0, j0 - r), min(self.mask.shape[0], j0 + r + 1))
            is_ = slice(max(0, i0 - r), min(self.mask.shape[1], i0 + r + 1))
            sub = self.mask[js, is_]
            if not sub.any():
                continue
            cj, ci = np.where(sub)
            d = (cj + js.start - j0) ** 2 + (ci + is_.start - i0) ** 2
            k = int(np.argmin(d))
            return (int(cj[k] + js.start), int(ci[k] + is_.start))
        return None

    def room_cells(self, room):
        return [c for c in self.cells_by_room.get(room, []) if self.mask[c[0], c[1]]]

    def room_point(self, room, rng, centrality=0.6):
        """A navigable point in @room, biased toward cells far from the room boundary."""
        cells = self.room_cells(room)
        if not cells:
            return None
        m = np.zeros(self.mask.shape, np.uint8)
        for j, i in cells:
            m[j, i] = 255
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        vals = np.array([dist[j, i] for j, i in cells])
        thr = np.quantile(vals, centrality) if len(vals) > 4 else 0.0
        pool = [c for c, v in zip(cells, vals) if v >= thr] or cells
        return pool[int(rng.integers(len(pool)))]

    # -- A* -----------------------------------------------------------------
    def path(self, start_xy, goal_xy):
        """8-connected A* inside the chosen component.  Returns (N,2) world waypoints."""
        s, g = self.snap(start_xy), self.snap(goal_xy)
        if s is None or g is None:
            return None
        if s == g:
            return np.array([self.to_world(s)])
        H, Wd = self.mask.shape
        nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        gscore = {s: 0.0}
        came = {}
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        pq = [(h(s), s)]
        seen = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == g:
                break
            cj, ci = cur
            for dj, di, w in nb:
                nj, ni = cj + dj, ci + di
                if not (0 <= nj < H and 0 <= ni < Wd) or not self.mask[nj, ni]:
                    continue
                if dj and di and not (self.mask[cj, ni] and self.mask[nj, ci]):
                    continue        # no corner-cutting through a diagonal gap
                ng = gscore[cur] + w
                if ng < gscore.get((nj, ni), 1e18):
                    gscore[(nj, ni)] = ng
                    came[(nj, ni)] = cur
                    heapq.heappush(pq, (ng + h((nj, ni)), (nj, ni)))
        if g not in came and g != s:
            return None
        out = [g]
        while out[-1] != s:
            out.append(came[out[-1]])
        out.reverse()
        return np.array([self.to_world(c) for c in out])

    def restrict(self, keep_rooms):
        """Narrow the walkable mask to @keep_rooms plus unlabelled cells (doorways,
        corridors), so an episode scoped to 4-8 rooms never routes through a room it
        does not report.  Falls back to the full component if that would disconnect."""
        keep = set(keep_rooms)
        sub = self.mask.copy()
        for room, cells in self.cells_by_room.items():
            if room in keep:
                continue
            for j, i in cells:
                sub[j, i] = False
        n, lab = cv2.connectedComponents(sub.astype(np.uint8) * 255, connectivity=4)
        cover = np.zeros(n, int)
        for room in keep:
            cells = [c for c in self.cells_by_room.get(room, []) if sub[c[0], c[1]]]
            if len(cells) >= 4:
                cover += (np.bincount(np.array([lab[j, i] for j, i in cells]),
                                      minlength=n) >= 4).astype(int)
        cover[0] = 0
        best = int(np.argmax(cover)) if n > 1 else 0
        got = sorted(r for r in keep
                     if sum(1 for c in self.cells_by_room.get(r, [])
                            if sub[c[0], c[1]] and lab[c[0], c[1]] == best) >= 4)
        if len(got) < len(keep):
            self.restricted = False       # keep the full component; caller is warned
            return sorted(keep)
        self.mask = sub & (lab == best)
        self.rooms = got
        self.restricted = True
        return got

    def route(self, start_xy, goal_xy, step=0.25, smooth_passes=2):
        """A* path, lightly smoothed, then resampled by arc length at @step metres.

        The raw A* output is one point per 0.05 m cell with 45-degree staircase
        jitter; walking it directly produced 0.05 m per live frame (a 5-room prelude
        cost 335 frames) and a heading that swung +-45 degrees between frames.
        """
        poly = self.path(start_xy, goal_xy)
        if poly is None or len(poly) < 2:
            return poly
        for _ in range(smooth_passes):
            out = poly.copy()
            for k in range(1, len(poly) - 1):
                cand = (poly[k - 1] + 2 * poly[k] + poly[k + 1]) / 4.0
                c = self.to_cell(cand)
                if self.mask[c[0], c[1]]:      # never smooth a corner out of the component
                    out[k] = cand
            poly = out
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(cum[-1])
        if total < 1e-6:
            return poly[:1]
        n = max(1, int(round(total / float(step))))
        want = np.linspace(0.0, total, n + 1)
        return np.stack([np.interp(want, cum, poly[:, 0]),
                         np.interp(want, cum, poly[:, 1])], axis=1)

    def summary(self):
        return dict(res=self.res, grid=list(self.mask.shape),
                    free_cells=int((self.free > 0).sum()),
                    walk_cells=int((self.walk > 0).sum()),
                    component_cells=int(self.mask.sum()),
                    components=int(self.ncomp - 1),
                    restricted=bool(getattr(self, "restricted", False)),
                    rooms_reachable=self.rooms, rooms_isolated=self.isolated)


def open_doors_for_clearance(scene, res=DEFAULT_RES, head=DEFAULT_HEAD, floor=0, radius=1.6):
    """Set each door joint to the limit that actually clears its doorway.

    Open().set_value(True) picks a direction from asset metadata; on several BEHAVIOR
    scenes that swings the leaf *into* the passage and makes connectivity worse
    (measured: Beechwood_0_int 5/11 -> 3/11, Rs_int 5/5 -> 4/5).  So try each joint
    extreme plus fully-shut and keep whichever frees the most standing space around
    the door, scored with the same downward ray the grid uses.
    """
    z0 = float(scene.trav_map.floor_heights[int(floor)])
    report = []
    for o in scene.objects:
        cat = str(getattr(o, "category", "")).lower()
        if "door" not in cat or not getattr(o, "n_joints", 0):
            continue
        try:
            p, _ = o.get_position_orientation()
            p = _arr(p).astype(float)
        except Exception:
            continue
        joints = list(o.joints.values())
        gx = np.arange(p[0] - radius, p[0] + radius + 1e-9, res)
        gy = np.arange(p[1] - radius, p[1] + radius + 1e-9, res)

        def clearance():
            n = 0
            for y in gy:
                for x in gx:
                    r = raytest([float(x), float(y), z0 + head],
                                [float(x), float(y), z0 - 0.4])
                    if r.get("hit") and abs(float(_arr(r["position"])[2]) - z0) <= FLOOR_TOL:
                        n += 1
            return n

        best = None
        for tag in ("shut", "lower", "upper"):
            try:
                for jt in joints:
                    if tag == "shut":
                        v = 0.0
                    elif tag == "lower":
                        v = float(jt.lower_limit)
                    else:
                        v = float(jt.upper_limit)
                    if not math.isfinite(v):
                        v = 0.0
                    jt.set_pos(v)
                og.sim.step()
            except Exception:
                continue
            c = clearance()
            if best is None or c > best[1]:
                best = (tag, c, [float(jt.lower_limit) if tag == "lower"
                                 else float(jt.upper_limit) if tag == "upper" else 0.0
                                 for jt in joints])
        if best is None:
            continue
        for jt, v in zip(joints, best[2]):
            jt.set_pos(v)
        report.append(dict(door=str(o.name), category=cat, mode=best[0], free_cells=best[1]))
    og.sim.step()
    return report


def build(scene, **kw):
    return NavGrid(scene, **kw)


if __name__ == "__main__":
    import json
    import os
    import sys
    import time

    from omnigibson.macros import gm
    gm.HEADLESS = True

    ASSETS = os.path.join(os.environ["OMNIGIBSON_DATA_PATH"], "behavior-1k-assets", "scenes")
    todo = sys.argv[1:] or sorted(d for d in os.listdir(ASSETS)
                                  if os.path.isdir(os.path.join(ASSETS, d, "layout")))
    out = os.environ.get("OUT", "/mnt/ssd2/wooyeol/work/og_diag_20260902/scene_screen.json")
    rows = []
    for name in todo:
        t0 = time.time()
        try:
            env = og.Environment(configs=dict(
                scene=dict(type="InteractiveTraversableScene", scene_model=name),
                robots=[], env=dict(action_frequency=30, physics_frequency=30)))
            ng = NavGrid(env.scene)
            row = dict(scene=name, ok=True, secs=round(time.time() - t0, 1), **ng.summary())
            row["n_reach"] = len(ng.rooms)
            row["n_rooms"] = len(ng.room_comp_hist)
            print("%-28s %2d/%-2d rooms in one component  (%s)  %.0fs"
                  % (name, row["n_reach"], row["n_rooms"], ",".join(ng.rooms[:6]), row["secs"]),
                  flush=True)
            env.close()
        except Exception as e:
            row = dict(scene=name, ok=False, error=str(e)[:200])
            print("%-28s ERROR %s" % (name, str(e)[:120]), flush=True)
        rows.append(row)
        json.dump(rows, open(out, "w"), indent=1)
    good = [r for r in rows if r.get("ok") and r.get("n_reach", 0) >= 4]
    print("\nusable (>=4 connected rooms): %d/%d" % (len(good), len(rows)))
    for r in sorted(good, key=lambda r: -r["n_reach"]):
        print("  %-28s %2d rooms" % (r["scene"], r["n_reach"]))
