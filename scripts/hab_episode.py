#!/usr/bin/env python3
"""Habitat 에피소드 생성 — THOR gen2 의 미니판(연속 배회 + 물체 이동 + GT).

    ~/miniforge3/envs/hab/bin/python scripts/hab_episode.py \\
      --scene apt_0 --dataset ~/habitat-data/replica_cad/replicaCAD.scene_dataset_config.json \\
      --frames 200 --moves 3 --out data/hab_ep/house_0000

THOR 와 다른 점(포토리얼 검증의 본질):
  · **연속 보행 궤적** — 텔레포트가 아니라 navmesh 위 경로를 따라 걷는다.
    SfM/VO 포즈 사슬이 성립하는 조건(THOR 에서 못 하던 것, §119 정정).
  · 물체 이동은 rigid object 를 실제로 옮긴다 → 이동 전/후 GT 가 정확.
산출: THOR gt.json 과 같은 스키마(rooms·room_types·gt0·moves·live·scene_meta)
      + live/*.jpg. 기존 평가군(eval_online·georoom)이 그대로 읽는다.
방 구획: region 주석이 없으면 물체 클러스터로 자동 분할(k=--rooms).
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image
import habitat_sim

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--moves", type=int, default=3)
ap.add_argument("--rooms", type=int, default=4)
ap.add_argument("--move", default=None,
                help="LLM 시나리오 사전확률(hssd_move.json) — 방 체류·이동성향·목적지. "
                     "없으면 전부 균등 난수가 되어 재방문 패턴이 비현실적이고 "
                     "목적지 균등이라 belief 가 원리적으로 못 맞힌다 (§67)")
ap.add_argument("--outdoor", type=float, default=0.0,
                help="이동 중 이 비율은 **집 밖**(outdoor/balcony/porch/garage)으로 — "
                     "'가방에 넣어 나갔다' 시나리오. 답은 '밖'이 되어야 한다")
ap.add_argument("--far", type=float, default=0.5,
                help="이동 중 '가장 먼 방'으로 보내는 비율 — 경우③ 표본 확보용")
ap.add_argument("--w", type=int, default=768)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "live"), exist_ok=True)
rng = np.random.default_rng(args.seed)
W = H = args.w
F = (W / 2.0) / np.tan(np.radians(45.0))

STRUCT = {"wall", "floor", "ceiling", "door", "window", "picture", "curtain", "rug",
          "mirror", "blinds", "stairs", "railing", "beam", "frame", "tvscreen"}
MOVABLE = ("book", "cushion", "plate", "bowl", "cup", "mug", "lamp", "clock", "vase",
           "basket", "kitchenutensil", "sponge", "toy", "phone", "laptop", "can",
           "box", "picture frame", "plant", "shoe", "bottle", "handbag")

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = args.scene
cfg.scene_dataset_config_file = args.dataset
# 물리 필수: 꺼두면 물체가 stage 에 병합돼 rigid 핸들이 0개 → 이동이 렌더에
# 반영되지 않는다(GT 만 바뀌는 가짜 이동). withbullet 빌드 필요.
cfg.enable_physics = True
def sensor(uuid, stype):
    sp = habitat_sim.CameraSensorSpec(); sp.uuid = uuid; sp.sensor_type = stype
    sp.resolution = [H, W]; sp.hfov = 90; sp.position = [0.0, 1.5, 0.0]
    return sp
agc = habitat_sim.agent.AgentConfiguration()
agc.sensor_specifications = [sensor("rgb", habitat_sim.SensorType.COLOR),
                             sensor("dep", habitat_sim.SensorType.DEPTH)]
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agc]))
if not sim.pathfinder.is_loaded:
    nmf = [c for c in glob.glob(os.path.join(os.path.dirname(args.dataset),
                                             "navmesh*", "*.navmesh")) if args.scene in c]
    if nmf:
        sim.pathfinder.load_nav_mesh(nmf[0])
    else:
        ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
        ns.agent_radius, ns.agent_height = 0.2, 1.6
        sim.recompute_navmesh(sim.pathfinder, ns)
assert sim.pathfinder.is_loaded, "navmesh 미로드"

# ── 물체 목록 (인스턴스 JSON) ──
_root = os.path.dirname(os.path.abspath(args.dataset))
scj = glob.glob(os.path.join(_root, "**", args.scene + ".scene_instance.json"), recursive=True)
assert scj, "scene_instance.json 못 찾음"
inst = json.load(open(scj[0]))
HASH = {}
for mf in glob.glob(os.path.join(_root, "metadata", "fpmodels*.csv")):
    import csv
    with open(mf, newline="") as f:
        for row in csv.DictReader(f):
            c = (row.get("main_category") or "").strip()
            if row.get("id") and c: HASH[row["id"]] = c.replace("_", " ").lower()

objs = {}          # oid → dict(type, pos[3])
for k, oi in enumerate(inst.get("object_instances", [])):
    base = oi["template_name"].split("/")[-1]
    lab = HASH.get(base)
    if not lab:
        if HASH: continue
        b2 = base
        for pre in ("frl_apartment_", "apt_", "object_"):
            if b2.startswith(pre): b2 = b2[len(pre):]
        lab = "".join(c for c in b2 if c.isalpha()).lower()
    if not lab or lab in STRUCT: continue
    objs["%s|%d" % (lab, k)] = dict(type=lab, pos=[float(x) for x in oi["translation"]])
assert objs, "물체 0개"
print("물체 %d" % len(objs), flush=True)

# ── 방 구획: region 주석 우선, 없으면 물체 좌표 k-means ──
polys, obj_room = {}, {}
sem = glob.glob(os.path.join(_root, "semantics", "scenes", args.scene + "*.json"))
if sem:
    for r in json.load(open(sem[0])).get("region_annotations", []):
        polys[r["name"]] = [[p[0], p[2]] for p in r["poly_loop"]]
if not polys:
    XZ = np.array([[v["pos"][0], v["pos"][2]] for v in objs.values()])
    cen = XZ[rng.choice(len(XZ), args.rooms, replace=False)]
    for _ in range(30):
        lab = np.argmin(((XZ[:, None] - cen[None]) ** 2).sum(-1), 1)
        for c in range(args.rooms):
            if (lab == c).any(): cen[c] = XZ[lab == c].mean(0)
    RT = ["living room", "kitchen", "bedroom", "study", "hallway", "dining room"]
    for c in range(args.rooms):
        pts = XZ[lab == c]
        if not len(pts): continue
        lo, hi = pts.min(0) - 0.6, pts.max(0) + 0.6
        polys["room|%d" % c] = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    for (oid, v), c in zip(objs.items(), lab): obj_room[oid] = "room|%d" % c
def room_at(x, z):
    best = (1e9, None)
    for r, pl in polys.items():
        a = np.array(pl); c = a.mean(0)
        inside = (a[:, 0].min() <= x <= a[:, 0].max()) and (a[:, 1].min() <= z <= a[:, 1].max())
        d = 0.0 if inside else float(np.hypot(x - c[0], z - c[1]))
        if d < best[0]: best = (d, r)
    return best[1]
for oid, v in objs.items():
    obj_room.setdefault(oid, room_at(v["pos"][0], v["pos"][2]))
rt = {r: (r.split("|")[0] if "|" in r else r) for r in polys}
print("방 %d: %s" % (len(polys), list(polys)[:6]), flush=True)

# ── 이동 계획: 타입 단일 + 옮길 만한 것 ──
from collections import Counter
cnt = Counter(v["type"] for v in objs.values())
cands = [o for o, v in objs.items()
         if cnt[v["type"]] == 1 and any(m in v["type"] for m in MOVABLE)]
rng.shuffle(cands)
plan = {}
# 경우③(부재→belief) 표본: --far 비율만큼은 **현재 방에서 가장 먼 방**으로 옮긴다
# → 이동 후 재목격 확률이 낮아지고 옛 방 재방문은 유지되어 ③ 조건이 성립한다
MOVE = json.load(open(args.move)) if args.move else None
import re as _re
def _rtype(r):                      # bedroom.001 → bedroom · room|2 → room
    return _re.sub(r"\.\d+$", "", rt.get(r, r))
if MOVE:                            # 이동 물체를 **이동성향**으로 뽑는다 (균등 아님)
    mob = MOVE.get("mobility", {})
    w_ = np.array([mob.get(objs[o]["type"], 0.3) + 1e-3 for o in cands], float)
    if w_.sum() > 0 and len(cands) > args.moves:
        idx = rng.choice(len(cands), size=min(args.moves * 3, len(cands)),
                         replace=False, p=w_ / w_.sum())
        cands = [cands[i2] for i2 in idx]
cen = {r: np.array(pl).mean(0) for r, pl in polys.items()}
for i2, oid in enumerate(cands[:args.moves]):
    others = [r for r in polys if r != obj_room[oid]]
    if not others: continue
    OUT_R = [r for r in others if any(k in _rtype(r) for k in
                                      ("outdoor", "balcony", "porch", "garage", "yard"))]
    if OUT_R and i2 >= args.moves - int(args.moves * args.outdoor):
        tgt = rng.choice(OUT_R)                      # 집 밖으로 가져나감
    elif i2 < int(args.moves * args.far):
        tgt = max(others, key=lambda r: float(np.linalg.norm(cen[r] - cen[obj_room[oid]])))
    elif MOVE and MOVE.get("dest", {}).get(objs[oid]["type"]):
        # **목적지 사전확률**: 머그컵은 부엌에 보관되지만 거실에서 발견된다
        dd = MOVE["dest"][objs[oid]["type"]]
        wv = np.array([dd.get(_rtype(r), 0.02) for r in others], float)
        tgt = others[int(rng.choice(len(others), p=wv / wv.sum()))] if wv.sum() > 0 \
              else rng.choice(others)
    else:
        tgt = rng.choice(others)
    plan[int(rng.integers(args.frames // 5, args.frames * 3 // 5))] = (oid, tgt)
print("이동 계획 %d건" % len(plan), flush=True)

# ── 연속 보행 궤적 (텔레포트 아님) ──
rom = sim.get_rigid_object_manager()
# 핸들 ↔ 인스턴스 대응: 이름이 아니라 **초기 좌표 최근접**으로 찾는다
# (핸들 문자열은 템플릿 해시/접두어라 라벨과 안 맞는다)
_H = []
for h in rom.get_object_handles():
    o = rom.get_object_by_handle(h)
    if o is None: continue
    try: _H.append((h, np.array(o.translation, float)))
    except Exception: pass
print("rigid 핸들 %d" % len(_H), flush=True)
def obj_handle(oid):
    if not _H: return None
    p0 = np.array(objs[oid]["pos"], float)
    h, d = min(((h, float(np.linalg.norm(t - p0))) for h, t in _H), key=lambda x: x[1])
    return h if d < 0.35 else None

plan_oids = {o for o, _d in plan.values()}
live, moves = [], []
gt0 = {oid: dict(type=v["type"], room=obj_room[oid],
                 pos=[round(v["pos"][0], 3), round(v["pos"][1], 3), round(v["pos"][2], 3)])
       for oid, v in objs.items()}
state = {oid: dict(v) for oid, v in objs.items()}
cur = sim.pathfinder.get_random_navigable_point()
goal = sim.pathfinder.get_random_navigable_point()
path = habitat_sim.ShortestPath(); path.requested_start, path.requested_end = cur, goal
sim.pathfinder.find_path(path)
route = list(path.points) if path.points else [cur]
ri, t = 0, 0
yaw = 0.0
while t < args.frames:
    if ri + 1 >= len(route):
        if MOVE and MOVE.get("dwell"):
            # 방 체류 가중으로 다음 목적지 방을 고른다 — 거실 오래, 창고 짧게
            rs_ = list(polys); wv = np.array([MOVE["dwell"].get(_rtype(r), 0.1)
                                              for r in rs_], float)
            r_pick = rs_[int(rng.choice(len(rs_), p=wv / wv.sum()))]
            c_ = np.array(polys[r_pick]).mean(0)
            goal = sim.pathfinder.get_random_navigable_point_near(
                np.array([c_[0], float(cur[1]), c_[1]]), 3.0)
            if not np.isfinite(goal).all():
                goal = sim.pathfinder.get_random_navigable_point()
        else:
            goal = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath(); path.requested_start, path.requested_end = cur, goal
        if not sim.pathfinder.find_path(path) or len(path.points) < 2: continue
        route, ri = list(path.points), 0
    a, b = np.array(route[ri]), np.array(route[ri + 1])
    seg = np.linalg.norm(b - a)
    steps = max(1, int(seg / 0.25))
    for k in range(steps):
        if t >= args.frames: break
        p = a + (b - a) * (k / steps)
        tgt_yaw = float(np.degrees(np.arctan2(-(b[0] - a[0]), -(b[2] - a[2]))))
        yaw += ((tgt_yaw - yaw + 180) % 360 - 180) * 0.5      # 부드러운 회전
        st = habitat_sim.AgentState(); st.position = p
        ry = np.radians(yaw)
        st.rotation = np.quaternion(np.cos(ry / 2), 0, np.sin(ry / 2), 0)
        sim.get_agent(0).set_state(st)
        # 이동 사건 실행 (관측 밖 여부는 GT 로 기록만 — 채점 시 사용)
        if t in plan:
            oid, dest = plan[t]
            pl = np.array(polys[dest]); c = pl.mean(0)
            np_ = sim.pathfinder.get_random_navigable_point_near(
                np.array([c[0], p[1], c[1]]), 2.0)
            if np.isfinite(np_).all():
                h = obj_handle(oid)
                if h:
                    o = rom.get_object_by_handle(h)
                    if o is not None:
                        import magnum as mn
                        o.translation = mn.Vector3(float(np_[0]),
                                                   float(state[oid]["pos"][1]), float(np_[2]))
                else:
                    print("  ⚠ 핸들 못 찾음(렌더 미반영, GT 만 갱신): %s" % oid, flush=True)
                # ⚠️ navigable 지점이 목적지 방 폴리곤 밖일 수 있다(반경 2m 탐색).
                # **실제 좌표로 방을 재판정**해야 GT 가 정직하다 (31% 불일치 실측).
                real = room_at(float(np_[0]), float(np_[2]))
                moves.append(dict(t=t, oid=oid, frm=obj_room[oid], to=real,
                                  intended=dest))
                state[oid]["pos"] = [float(np_[0]), state[oid]["pos"][1], float(np_[2])]
                obj_room[oid] = real
        obs = sim.get_sensor_observations()
        dep = obs["dep"]
        cam = np.array(p) + np.array([0, 1.5, 0])
        fwd = np.array([-np.sin(np.radians(yaw)), 0, -np.cos(np.radians(yaw))])
        rgt = np.array([np.cos(np.radians(yaw)), 0, -np.sin(np.radians(yaw))])
        up = np.array([0, 1.0, 0])
        vis, ctr, dist, anch = [], {}, {}, {}
        for oid, v in state.items():
            d3 = np.array(v["pos"]) - cam
            zc = float(d3 @ fwd)
            if not (0.3 < zc < 12): continue
            u = W / 2 + F * float(d3 @ rgt) / zc
            vv = H / 2 - F * float(d3 @ up) / zc
            if not (5 <= u < W - 5 and 5 <= vv < H - 5): continue
            if abs(float(dep[int(vv), int(u)]) - zc) > 0.6: continue     # 가림 배제
            vis.append(oid); ctr[oid] = [round(u, 1), round(vv, 1)]
            dist[oid] = round(float(np.hypot(d3[0], d3[2])), 2)
            if oid not in plan_oids:            # 정적 물체 = 앵커 (exemplar 재국소화용)
                anch[oid] = [round(u, 1), round(vv, 1)]
        Image.fromarray(obs["rgb"][..., :3]).save(
            os.path.join(args.out, "live", "%06d.jpg" % t), quality=88)
        live.append(dict(t=t, room=room_at(float(p[0]), float(p[2])), vis=vis, ctr=ctr, anch=anch,
                         dist=dist, apos=[round(float(p[0]), 2), round(float(p[2]), 2)],
                         # ⚠️ habitat 카메라는 yaw=0 에서 -z 를 본다(OpenGL 관례).
                         # 우리 투영 규약은 yaw=0 → +z, 전방 (sin,0,cos) 이므로 180° 보정.
                         yaw=round((yaw + 180.0) % 360, 1), pitch=0.0))
        t += 1
    cur = b; ri += 1

# ── 매핑 워크: 방마다 들러 360° 스캔 (THOR gen2 --mapwalk 대응) ──
# 이게 없으면 초기 프레임이 한두 방만 돌아 **이동 물체가 exemplar 를 못 가진다**
# → 평가 타겟 목록에서 빠진다(HSSD 20채 1차 실패의 원인).
mapwalk = []
os.makedirs(os.path.join(args.out, "map"), exist_ok=True)
mi = 0
for r, pl in polys.items():
    c = np.array(pl).mean(0)
    mp_pt = sim.pathfinder.get_random_navigable_point_near(
        np.array([c[0], 0.1, c[1]]), 3.0)
    if not np.isfinite(mp_pt).all(): continue
    for yy in (0, 60, 120, 180, 240, 300):
        st = habitat_sim.AgentState(); st.position = mp_pt
        ry = np.radians(yy)
        st.rotation = np.quaternion(np.cos(ry / 2), 0, np.sin(ry / 2), 0)
        sim.get_agent(0).set_state(st)
        ob = sim.get_sensor_observations()
        dep2 = ob["dep"]
        cam2 = np.array(mp_pt) + np.array([0, 1.5, 0])
        f2 = np.array([-np.sin(ry), 0, -np.cos(ry)])
        r2 = np.array([np.cos(ry), 0, -np.sin(ry)])
        u2 = np.array([0, 1.0, 0])
        box = {}
        for oid, v in objs.items():
            d3 = np.array(v["pos"]) - cam2
            zc = float(d3 @ f2)
            if not (0.3 < zc < 12): continue
            uu = W / 2 + F * float(d3 @ r2) / zc
            vv = H / 2 - F * float(d3 @ u2) / zc
            if not (5 <= uu < W - 5 and 5 <= vv < H - 5): continue
            if abs(float(dep2[int(vv), int(uu)]) - zc) > 0.6: continue
            half = max(12.0, F * 0.25 / max(zc, 0.3))
            box[oid] = [int(max(0, uu - half)), int(max(0, vv - half)),
                        int(min(W, uu + half)), int(min(H, vv + half))]
        Image.fromarray(ob["rgb"][..., :3]).save(
            os.path.join(args.out, "map", "%04d.jpg" % mi), quality=88)
        # 투영 기반 initmap 을 위해 포즈·거리도 남긴다 (box 만으로는 방 판정 불가 —
        # "보이는 방 ≠ 있는 방" 이 초기맵 구축에서 그대로 재현된다, §121 후속)
        mapwalk.append(dict(room=r, yaw=float(yy), box=box,
                            apos=[round(float(mp_pt[0]), 2), round(float(mp_pt[2]), 2)],
                            ctr={o: [round((b[0]+b[2])/2, 1), round((b[1]+b[3])/2, 1)]
                                 for o, b in box.items()},
                            dist={o: round(float(np.hypot(objs[o]["pos"][0]-mp_pt[0],
                                                          objs[o]["pos"][2]-mp_pt[2])), 2)
                                  for o in box}))
        mi += 1
print("매핑 워크 %d장 · exemplar 물체 %d" % (mi, len({o for x in mapwalk for o in x["box"]})), flush=True)

# ⚠️ 좌표 규약 정합: habitat 은 화면 오른쪽이 방위각 **감소** 방향(왼손 회전),
# 우리 투영 규약(THOR/georoom)은 증가 방향이다. x 축을 미러링하고 yaw 부호를
# 뒤집으면 두 규약이 일치한다 (이미지·ctr 은 손대지 않는다 — GT 정합 유지).
def _mx(v):  # [x,(y),z] → x 반전
    v = list(v); v[0] = -v[0]; return v
for _m in live:
    _m["apos"] = [round(-_m["apos"][0], 2), _m["apos"][1]]
    _m["yaw"] = round((-_m["yaw"]) % 360, 1)
for _m in mapwalk:                      # 매핑워크도 같은 규약으로
    _m["apos"] = [round(-_m["apos"][0], 2), _m["apos"][1]]
    _m["yaw"] = round((-(_m["yaw"]) + 180.0) % 360, 1)
for _d in (gt0, state):
    for _v in _d.values(): _v["pos"] = _mx(_v["pos"])
for _r in polys: polys[_r] = [[-x, z] for x, z in polys[_r]]

# ── 초기 맵(매핑 워크 대용): 앞 MAPN 프레임을 map 으로도 기록 + 물체 bbox
# (exp_imgq 의 exemplar 는 map 프레임의 box 에서 크롭을 뽑는다. 우리 시스템이
#  "초기 매핑 워크로 씬그래프를 만든다"고 전제하므로 앞 구간 재사용이 정당하다)
MAPN = 0 if mapwalk else min(80, len(live))
mp = list(mapwalk)
for m in live[:MAPN]:
    box = {}
    for oid in m["vis"]:
        c = m["ctr"][oid]; d = m["dist"].get(oid, 3.0)
        rad = 0.25
        half = max(12.0, F * rad / max(d, 0.3))
        box[oid] = [int(max(0, c[0] - half)), int(max(0, c[1] - half)),
                    int(min(W, c[0] + half)), int(min(H, c[1] + half))]
    mp.append(dict(room=m["room"], yaw=m["yaw"], box=box, t=m["t"]))

static = {oid: dict(type=v["type"], room=obj_room[oid],
                    pos=[round(v["pos"][0], 3), round(v["pos"][1], 3), round(v["pos"][2], 3)])
          for oid, v in objs.items() if oid not in {m["oid"] for m in moves}}
json.dump(dict(house=0, rooms=[dict(id=r, type=rt[r]) for r in polys], room_types=rt,
               gt0=gt0, moves=moves, live=live, map=mp, fps=1.0, T=len(live),
               scene_meta=dict(polys=polys, static=static, doors=[])),
          open(os.path.join(args.out, "gt.json"), "w"))
print("프레임 %d · 이동 %d · 방 %d → %s" % (len(live), len(moves), len(polys), args.out))
