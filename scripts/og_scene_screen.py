#!/usr/bin/env python3
"""Scene screening with the generator's own scope rules (2026-09-02 범위 결정).

Builds the live raycast navgrid, drops outdoor room types, and reports whether what is
left satisfies "4-8 rooms including a kitchen and a bathroom".  This is the last input
needed to fix the 20-house scene/seed list (handover 10-3).

Excluding garage/garden is a *room type* rule, not a *scene* rule -- a garden scene
still qualifies if its indoor rooms do.

One scene per invocation: OmniGibson 3.9 refuses a second InteractiveTraversableScene in
the same process ("Simulator must be stopped before loading scene!") and its teardown
segfaults anyway, so the driver loops in shell.

    python scripts/og_scene_screen.py Rs_int
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True

import og_navgrid

EXCLUDE = ("garage", "garden", "porch", "yard", "lawn", "balcony", "patio", "outdoor",
           "driveway", "deck", "terrace")
NEED = ("kitchen", "bathroom")
MIN_ROOMS, MAX_ROOMS = 4, 8


def rtype_of(r):
    return r.rsplit("_", 1)[0].replace("_", " ")


if len(sys.argv) != 2:
    sys.exit("usage: og_scene_screen.py <scene_model>   (one scene per process)")
name = sys.argv[1]
out = os.environ.get("OUT", "/mnt/ssd2/wooyeol/work/og_diag_20260902/scene_screen.jsonl")

t0 = time.time()
row = dict(scene=name)
try:
    env = og.Environment(configs=dict(
        scene=dict(type="InteractiveTraversableScene", scene_model=name),
        robots=[], env=dict(action_frequency=30, physics_frequency=30)))
    doors = og_navgrid.open_doors_for_clearance(env.scene)
    for _ in range(30):
        og.sim.step_physics()
    ng = og_navgrid.build(env.scene)
    reach = list(ng.rooms)
    indoor = [r for r in reach if not any(h in rtype_of(r) for h in EXCLUDE)]
    outdoor = [r for r in reach if r not in indoor]
    types = {rtype_of(r) for r in indoor}
    missing = [t for t in NEED if t not in types]
    row.update(ok=True, secs=round(time.time() - t0, 1), doors=len(doors),
               n_reachable=len(reach), n_indoor=len(indoor),
               indoor=sorted(indoor), outdoor=sorted(outdoor),
               isolated=ng.isolated, missing=missing,
               walk_cells=ng.summary()["component_cells"],
               qualifies=bool(not missing and len(indoor) >= MIN_ROOMS),
               needs_trim=len(indoor) > MAX_ROOMS)
    print("SCREEN %-26s indoor %2d/%-2d  kitchen+bath %-14s %-8s %s"
          % (name, len(indoor), len(reach),
             "yes" if not missing else "NO:" + ",".join(missing),
             "QUALIFY" if row["qualifies"] else "reject",
             ("(trim to %d)" % MAX_ROOMS) if row["needs_trim"] else ""), flush=True)
    print("   indoor: %s" % ", ".join(sorted(indoor)), flush=True)
    if outdoor:
        print("   outdoor dropped: %s" % ", ".join(sorted(outdoor)), flush=True)
    if ng.isolated:
        print("   isolated: %s" % ", ".join(ng.isolated), flush=True)
except Exception as e:
    row.update(ok=False, error=str(e)[:200])
    print("SCREEN %-26s ERROR %s" % (name, str(e)[:120]), flush=True)

with open(out, "a") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print("SCREEN_DONE", flush=True)
