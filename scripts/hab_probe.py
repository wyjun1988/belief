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
STRUCT = {"wall", "floor", "ceiling", "door", "window", "picture", "curtain",
          "rug", "mirror", "blinds", "wall art", "doorway", "stairs", "railing",
          "column", "beam", "vent", "switch", "outlet", "molding", "baseboard",
          "carpet", "wallpaper", "tile", "panel", "frame"}


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
if not sim.pathfinder.is_loaded:
    # ReplicaCAD 류: navmesh 를 명시 로드해야 한다 (미로드 상태의 pathfinder 호출은 무효/불안정)
    import glob as _g
    _nm = [c for c in _g.glob(os.path.join(os.path.dirname(args.dataset),
                                           "navmesh*", "*.navmesh")) if args.scene in c]
    if _nm:
        sim.pathfinder.load_nav_mesh(_nm[0])
        print("navmesh 로드:", _nm[0], flush=True)
if not sim.pathfinder.is_loaded:
    # HSSD 류: navmesh 파일이 없다 → 런타임 생성 (에이전트 반경·높이 기본값)
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius, ns.agent_height = 0.2, 1.6
    ok = sim.recompute_navmesh(sim.pathfinder, ns)
    print("navmesh 런타임 생성:", ok, flush=True)
assert sim.pathfinder.is_loaded, "navmesh 미로드(생성 실패)"

# ── 물체 GT: 시맨틱 대신 scene_instance.json 직접 파싱 + 투영·뎁스 일치 검사 ──
# (ReplicaCAD 는 시맨틱 기술자·강체 등록이 없어도 라벨·좌표가 인스턴스 JSON 에 있다.
#  뎁스 버퍼와 투영 깊이가 일치해야 채택 → 가림·미렌더가 자동으로 걸러진다)
import glob as _glob
# 장면 인스턴스 JSON 위치는 데이터셋마다 다르다 (ReplicaCAD: configs/scenes,
# HSSD: scenes-uncluttered 등) — 데이터셋 루트에서 탐색한다
_root = os.path.dirname(os.path.abspath(args.dataset))
_c = _glob.glob(os.path.join(_root, "**", args.scene + ".scene_instance.json"),
                recursive=True)
assert _c, "scene_instance.json 못 찾음: %s (%s)" % (args.scene, _root)
sc_json = _c[0]
inst = json.load(open(sc_json))
# 해시 id → 카테고리 (HSSD: metadata/fpmodels*.csv 의 main_category)
HASHCAT = {}
_meta = _glob.glob(os.path.join(_root, "metadata", "fpmodels*.csv"))
if _meta:
    import csv as _csv
    with open(_meta[0], newline="") as _f:
        for row in _csv.DictReader(_f):
            c = (row.get("main_category") or row.get("super_category") or "").strip()
            if row.get("id") and c: HASHCAT[row["id"]] = c.replace("_", " ").lower()
    print("카테고리 매핑 %d" % len(HASHCAT), flush=True)

OBJS = []
for oi in inst.get("object_instances", []):
    base = oi["template_name"].split("/")[-1]
    label = HASHCAT.get(base)
    if not label:
        if HASHCAT:
            continue          # 카테고리표가 있는데 매핑 실패 = 해시 잔재 → 버린다
        for pre in ("frl_apartment_", "apt_", "object_"):
            if base.startswith(pre): base = base[len(pre):]
        label = "".join(c for c in base if c.isalpha()).lower()
    # 구조물·벽부착물 제외: 근거리 크롭이 벽 텍스처만 담아 판별 불가하고
    # (HSSD 중간판독에서 <2m AUC 0.60 붕괴의 주범) 우리 과제 대상도 아니다.
    # 해시 잔재(매핑 실패)도 제외 — 라벨이 무의미하면 채점이 무효다.
    if not label or label in STRUCT: continue
    OBJS.append((label, np.array(oi["translation"], float)))
print("인스턴스 JSON 물체 %d" % len(OBJS), flush=True)

INDOOR_ONLY = os.environ.get("INDOOR_ONLY", "1") == "1"
INDOOR_TYPES = ("sofa", "bed", "chair", "table", "cabinet", "shelf", "desk", "counter",
                "stool", "dresser", "wardrobe", "refrigerator", "sink", "tv", "lamp",
                "couch", "nightstand", "bookcase", "oven", "microwave", "toilet")
OBJXZ = np.array([[p3[0], p3[2]] for lb, p3 in OBJS
                  if any(t in lb for t in INDOOR_TYPES)], float) if OBJS else np.zeros((0, 2))
print("실내 가구 앵커 %d" % len(OBJXZ), flush=True)
F = (args.w / 2.0) / np.tan(np.radians(45.0))
meta = []
made = 0; tries = 0
while made < args.frames and tries < args.frames * 20:
    tries += 1
    pt = sim.pathfinder.get_random_navigable_point()
    if not np.isfinite(pt).all(): continue
    # 실내 판정: 반경 3.5m 안에 실내 가구가 4개 이상 (HSSD 는 마당·외부 포함이고
    # 런타임 navmesh 가 실외 지면도 navigable 로 잡는다. 물리 미사용이라 레이캐스트
    # 대신 밀도로 — 우리 목적(물체가 보이는 프레임)에도 직결된다)
    if INDOOR_ONLY and OBJXZ.size:
        d2 = np.hypot(OBJXZ[:, 0] - pt[0], OBJXZ[:, 1] - pt[2])
        if int(np.sum(d2 < 3.5)) < 4: continue
    st = habitat_sim.AgentState()
    st.position = pt
    yaw = float(rng.uniform(0, 2 * np.pi))
    st.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
    sim.get_agent(0).set_state(st)
    obs = sim.get_sensor_observations()
    dep = obs["dep"]
    cam = np.array(pt, float) + np.array([0, 1.5, 0])
    fwd = np.array([-np.sin(yaw), 0, -np.cos(yaw)])
    rgt = np.array([np.cos(yaw), 0, -np.sin(yaw)])
    up = np.array([0, 1.0, 0])
    objs = []
    for label, p3 in OBJS:
        d = p3 - cam
        zc = float(d @ fwd)
        if not (0.4 < zc < 15): continue
        u = args.w / 2 + F * float(d @ rgt) / zc
        v = args.h / 2 - F * float(d @ up) / zc
        if not (10 <= u < args.w - 10 and 10 <= v < args.h - 10): continue
        dz = float(dep[int(v), int(u)])
        if abs(dz - zc) > 0.5: continue            # 가림/미렌더 → 제외
        objs.append(dict(label=label, ctr=[round(u, 1), round(v, 1)],
                         dist=round(float(np.linalg.norm(d)), 2)))
    if len({o["label"] for o in objs}) < 2: continue
    fn = "frames/%06d.jpg" % made
    Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(args.out, fn), quality=90)
    meta.append(dict(img=fn, objs=objs)); made += 1
    if made % 50 == 0: print("  %d/%d" % (made, args.frames), flush=True)
open(os.path.join(args.out, "meta.jsonl"), "w").write("\n".join(json.dumps(m) for m in meta))
assert made > 0, "프레임 0장 — 장면/투영 확인"
print("프레임 %d → %s" % (made, args.out))
