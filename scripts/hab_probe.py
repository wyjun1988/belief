#!/usr/bin/env python3
"""Habitat 렌더 프로브 — 장면에서 무작위 시점 렌더 + 물체 라벨·중심·거리 덤프.

    ~/miniforge3/envs/hab/bin/python scripts/hab_probe.py \\
      --scene <scene.glb 또는 scene_dataset_config 의 장면 id> \\
      --dataset <scene_dataset_config.json> --frames 200 --out /tmp/hab_probe

시맨틱 센서로 인스턴스 id → 카테고리, 뎁스로 물체 중심 거리.
출력은 probe_pairs.py 공통 형식. (ReplicaCAD 로 기동 확인 → HSSD 로 본시험)
"""
import argparse, json, os
import numpy as np
from PIL import Image
import habitat_sim

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--dataset", default=None)
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--out", required=True)
ap.add_argument("--w", type=int, default=768)
ap.add_argument("--h", type=int, default=768)
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
rng = np.random.default_rng(0)

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = args.scene
if args.dataset: cfg.scene_dataset_config_file = args.dataset
cfg.enable_physics = False

def sensor(uuid, stype):
    sp = habitat_sim.CameraSensorSpec()
    sp.uuid = uuid; sp.sensor_type = stype
    sp.resolution = [args.h, args.w]; sp.hfov = 90
    sp.position = [0.0, 1.5, 0.0]
    return sp

ag = habitat_sim.agent.AgentConfiguration()
ag.sensor_specifications = [
    sensor("rgb", habitat_sim.SensorType.COLOR),
    sensor("sem", habitat_sim.SensorType.SEMANTIC),
    sensor("dep", habitat_sim.SensorType.DEPTH)]
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))

# 인스턴스 id → 카테고리 이름
cat = {}
for obj in sim.semantic_scene.objects or []:
    if obj is None or obj.category is None: continue
    try: cat[int(obj.semantic_id)] = obj.category.name()
    except Exception: pass
print("시맨틱 인스턴스 %d" % len(cat), flush=True)

SKIP = {"wall", "floor", "ceiling", "door", "window", "misc", "unknown", "", "root",
        "stair", "column", "beam"}
meta = []
made = 0; tries = 0
while made < args.frames and tries < args.frames * 10:
    tries += 1
    pt = sim.pathfinder.get_random_navigable_point()
    if not np.isfinite(pt).all(): continue
    st = habitat_sim.AgentState()
    st.position = pt
    yaw = float(rng.uniform(0, 2 * np.pi))
    st.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
    sim.get_agent(0).set_state(st)
    obs = sim.get_sensor_observations()
    sem = obs["sem"]; dep = obs["dep"]
    objs = []
    for iid in np.unique(sem):
        c = cat.get(int(iid))
        if not c or c.lower() in SKIP: continue
        ys, xs = np.where(sem == iid)
        if len(xs) < 400: continue                    # 너무 작은 조각 제외
        cx, cy = float(np.median(xs)), float(np.median(ys))
        d = float(np.median(dep[ys, xs]))
        if not (0.3 < d < 15): continue
        objs.append(dict(label=c.lower(), ctr=[cx, cy], dist=round(d, 2)))
    if len({o["label"] for o in objs}) < 2: continue
    fn = "frames/%06d.jpg" % made
    Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(args.out, fn), quality=90)
    meta.append(dict(img=fn, objs=objs)); made += 1
    if made % 50 == 0: print("  %d/%d" % (made, args.frames), flush=True)
open(os.path.join(args.out, "meta.jsonl"), "w").write("\n".join(json.dumps(m) for m in meta))
assert made > 0, "프레임 0장 — 장면/시맨틱 확인"
print("프레임 %d → %s" % (made, args.out))
