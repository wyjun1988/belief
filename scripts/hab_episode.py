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
import argparse, glob, json, os, sys
import numpy as np
from PIL import Image
import habitat_sim
import magnum as mn
from PIL import ImageDraw

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
ap.add_argument("--case3", type=float, default=0.5,
                help="이동 중 이 비율을 **경우③ 대본**으로: 이동 후 배회에서 목적지 방을 제외하고 "
                     "원래 방을 한 번 강제 재방문. 나머지는 경우② 대본(목적지 방 강제 방문). "
                     "무작위 배회로는 ③이 안 생긴다(v2: 이동 18·③ 0) — AUDIT 제안1")
ap.add_argument("--evidence", default=None,
                help="K:D — ② 역할 이동 후 궤적이 물체에서 거리 D(m) 인 지점을 K 번 들른다 "
                     "(물체를 향해 1.5m 직진 구간으로 진입 → 그 방향을 보며 걷는다). 증거량을 "
                     "시나리오 우연이 아닌 통제 변수로 (EVAL_PROTOCOL_V2)")
ap.add_argument("--remap", action="store_true",
                help="매핑워크만 다시 돌려 기존 --out 의 map/ 과 gt.json[map] 을 교체 (live 는 그대로). "
                     "종전 매핑워크는 live 루프 뒤(이동 후 장면)에서 돌았고 좌표가 인스턴스 원점이라 "
                     "이동 물체 exemplar 가 빈자리 크롭이었다 (2026-09-02)")
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

def _pip(pt, poly):
    # 점-폴리곤(ray casting). AABB 는 HSSD 처럼 L자·겹치는 방에서 틀린다.
    x, z = float(pt[0]), float(pt[1]); ins = False; n = len(poly)
    for k in range(n):
        x1, z1 = poly[k]; x2, z2 = poly[(k + 1) % n]
        if (z1 > z) != (z2 > z):
            xi = x1 + (z - z1) * (x2 - x1) / ((z2 - z1) or 1e-9)
            if x < xi: ins = not ins
    return ins

STRUCT = {"wall", "floor", "ceiling", "door", "window", "picture", "curtain", "rug",
          "mirror", "blinds", "stairs", "railing", "beam", "frame", "tvscreen"}
MOVABLE = ("book", "cushion", "plate", "bowl", "cup", "mug", "lamp", "clock", "vase",
           "basket", "kitchenutensil", "sponge", "toy", "phone", "laptop", "can",
           "box", "picture frame", "plant", "shoe", "bottle", "handbag", "drinkware",
           "toiletry", "candle", "clothing", "tray", "kettle", "remote", "bag", "hat")

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
    objs["%s|%d" % (lab, k)] = dict(type=lab, pos=[float(x) for x in oi["translation"]], tmpl=base)
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
    hits = [r for r, pl in polys.items() if _pip((x, z), pl)]
    if hits: return min(hits, key=lambda r: (lambda P: abs(sum(P[k][0]*P[(k+1)%len(P)][1]-P[(k+1)%len(P)][0]*P[k][1] for k in range(len(P)))))(polys[r]))   # 겹치면 면적 작은 방
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

rom = sim.get_rigid_object_manager()
# 핸들 ↔ 인스턴스: HSSD 는 COM 보정으로 rigid 좌표가 인스턴스 좌표와 최대 1.3m 어긋난다
# (55개 중 19개만 최근접 매칭됨 → 이동 후보 소실). 핸들은 "<템플릿해시>_:NNNN" 이므로
# **템플릿 해시로 먼저 맞추고** 같은 템플릿끼리만 xz 최근접으로 가른다.
_H = []
for h in rom.get_object_handles():
    o = rom.get_object_by_handle(h)
    if o is None: continue
    try: _H.append((h, np.array(o.translation, float)))
    except Exception: pass
_HBY = {}
for h, t in _H: _HBY.setdefault(h.split("_:")[0].split("/")[-1], []).append((h, t))
print("rigid 핸들 %d" % len(_H), flush=True)
def obj_handle(oid):
    if not _H: return None
    p0 = np.array(objs[oid]["pos"], float)
    pool = _HBY.get(objs[oid].get("tmpl"))
    if pool:
        h, d = min(((h, float(np.hypot(t[0]-p0[0], t[2]-p0[2]))) for h, t in pool), key=lambda x: x[1])
        return h if d < 1.5 else None
    h, d = min(((h, float(np.linalg.norm(t - p0))) for h, t in _H), key=lambda x: x[1])
    return h if d < 0.35 else None

def _down_hits(x, y_top, z, skip_id):
    ray = habitat_sim.geo.Ray(mn.Vector3(float(x), float(y_top), float(z)), mn.Vector3(0., -1., 0.))
    res = sim.cast_ray(ray, max_distance=6.0)
    return [h for h in res.hits if h.object_id != skip_id] if res.has_hits() else []

def support_offset(o, pos):
    # 물체 원점이 받침면 위 얼마나 떠 있나 — 자산마다 원점 규약(바닥/중심)이 달라 실측.
    # 0.6m 초과 = 벽걸이(시계·벽등): 받침면이 없으니 **이동 대상에서 제외**한다(§125 후속:
    # 원래 높이를 재현하면 허공에 뜬다).
    hs = [h for h in _down_hits(pos[0], pos[1] + 1.0, pos[2], o.object_id) if h.point[1] <= pos[1] + 0.05]
    return (pos[1] - max(h.point[1] for h in hs)) if hs else 0.0

def on_floor_originally(o, pos):
    # 원래 받침면이 바닥인가 — 플로어램프·휴지통이 카운터 위로 올라가는 것을 막는다
    hs = [h for h in _down_hits(pos[0], pos[1] + 1.0, pos[2], o.object_id) if h.point[1] <= pos[1] + 0.05]
    if not hs: return True
    y_sup = max(h.point[1] for h in hs); y_floor = min(h.point[1] for h in hs)
    return (y_sup - y_floor) < 0.15

def pick_receptacle(o, np_):
    # 보행점 주변 1m 격자에서 **테이블/카운터 상판**(바닥 위 0.25~1.2m, 수평면) 우선, 없으면 바닥.
    # 바닥에만 놓으면 접시·컵이 눈높이에서 안 보이고 비현실적이다(§125 후속 실측).
    fy = float(np_[1]); best = None
    for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for dz in (-1.0, -0.5, 0.0, 0.5, 1.0):
            x, z = float(np_[0]) + dx, float(np_[2]) + dz
            for h in _down_hits(x, fy + 2.0, z, o.object_id):
                hy = float(h.point[1])
                if 0.25 <= hy - fy <= 1.2 and float(h.normal[1]) > 0.7:
                    if best is None or hy > best[2]: best = (x, z, hy)
                    break
    return best if best else (float(np_[0]), float(np_[2]), fy)

def is_supported(o, off):
    # 배치 후 COM 에서 내려쏘아 첫 비자기 충돌이 오프셋 근처인가 — 허공에 뜬 물체를 잡는다
    p = np.array(o.translation, float)
    hs = _down_hits(p[0], p[1], p[2], o.object_id)
    return bool(hs) and abs(float(p[1] - hs[0].point[1]) - off) <= 0.15

def place_at(o, x, z, y_s, off):
    o.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    o.translation = mn.Vector3(float(x), float(y_s + off + 0.02), float(z))
    # 2단: 격자 지점의 받침면과 물체 바로 아래 받침면이 다를 수 있다 → 제자리에서 다시 재서 스냅
    hs = [h for h in _down_hits(x, float(y_s + off + 0.5), z, o.object_id)
          if float(h.normal[1]) > 0.5 and float(h.point[1]) <= y_s + off + 0.5]
    if hs:
        o.translation = mn.Vector3(float(x), float(hs[0].point[1] + off), float(z))
    return [float(v) for v in o.translation]

# ── 이동 계획: 타입 단일 + 옮길 만한 것 ──
from collections import Counter
cnt = Counter(v["type"] for v in objs.values())
MOUNTED = ("ceiling", "wall lamp", "wall clock", "curtain", "chandelier", "sconce")
cands = [o for o, v in objs.items()
         if cnt[v["type"]] == 1 and any(m in v["type"] for m in MOVABLE)
         and not any(m in v["type"] for m in MOUNTED)]
_n0 = len(cands)
cands = [o_ for o_ in cands if obj_handle(o_)]
_n1 = len(cands)
def _hpos(o_):   # rigid 노드(COM 보정) 좌표 — 인스턴스 좌표(자산 원점)와 최대 1.3m 다르다
    return [float(v) for v in rom.get_object_by_handle(obj_handle(o_)).translation]
cands = [o_ for o_ in cands
         if support_offset(rom.get_object_by_handle(obj_handle(o_)), _hpos(o_)) <= 0.6]
print("이동 후보 %d → 핸들 있음 %d → 벽걸이 제외 %d" % (_n0, _n1, len(cands)), flush=True)
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
    # 역할은 **실제 후보 수** 에 대한 비율로 번갈아(c2 먼저) — 종전 `i2 < moves*case3` 는 채당 후보가
    # 1~3개라 전부 ③ 이 됐다 (v3/v3b: c3 34 · c2 1 → ② 5건). N=1→c2, N=2→c2,c3, N=3→c2,c3,c2
    _N = min(len(cands), args.moves)
    role = "c3" if int(round((i2 + 1) * args.case3)) > int(round(i2 * args.case3)) else "c2"
    if role == "c3":
        _dw = (MOVE or {}).get("dwell", {})
        _in = [r for r in others if not any(k in _rtype(r) for k in
                                            ("outdoor", "balcony", "porch", "garage", "yard"))] or others
        _low = [r for r in _in if _dw.get(_rtype(r), 0.1) <= 0.35] or _in      # 실외는 ④ 몫 — ③ 목적지에서 제외
        tgt = max(_low, key=lambda r: float(np.linalg.norm(cen[r] - cen[obj_room[oid]])))
    plan[int(rng.integers(args.frames // 5, args.frames * 3 // 5))] = (oid, tgt, role)
print("이동 계획 %d건 (③대본 %d)" % (len(plan), sum(1 for v in plan.values() if v[2] == "c3")), flush=True)
excluded_rooms = set()      # ③ 대본: 이동 후 배회에서 제외할 목적지 방
hidden_oids = []            # ③ 대본: 배회 경로에서 시선이 닿으면 안 되는 물체
forced_goals = []           # 대본이 요구하는 다음 목적지 방 (③: 원래 방 / ②: 목적지 방)

# ── 연속 보행 궤적 (텔레포트 아님) ──

plan_oids = {v[0] for v in plan.values()}
live, moves = [], []
gt0 = {oid: dict(type=v["type"], room=obj_room[oid],
                 pos=[round(v["pos"][0], 3), round(v["pos"][1], 3), round(v["pos"][2], 3)])
       for oid, v in objs.items()}
state = {oid: dict(v) for oid, v in objs.items()}

# ── 배치·가시성 정직성 (AUDIT 2026-09-02 후속) ──
# 종전: 이동 시 x·z 만 바닥 보행점으로 옮기고 **높이는 원래 값 유지** → 테이블 위 램프가
# 새 자리에서 소파 몸체에 박혀 렌더에 없었다(이동 후 OWL 검출 0/417). GT 는 "보인다" 고
# 했으므로 ② 는 처음부터 풀 수 없는 문제였다. 고침: 받침면 레이캐스트 배치 + GT 는 실제
# 좌표 + 가시성은 카메라→물체 시선 검사 + 이동마다 증인 렌더 self-test.
OBJID = {}
for _oid in objs:
    _h = obj_handle(_oid)
    if _h:
        _o = rom.get_object_by_handle(_h)
        if _o is not None: OBJID[_oid] = _o.object_id
print("물체 %d 중 rigid 핸들 대응 %d" % (len(objs), len(OBJID)), flush=True)
for _oid, _id in OBJID.items():
    try:
        _p = [float(v) for v in rom.get_object_by_id(_id).translation]
        state[_oid]["pos"] = _p; gt0[_oid]["pos"] = [round(v, 3) for v in _p]
    except Exception: pass

def obj_center(oid):
    if oid in OBJID:
        try:
            o = rom.get_object_by_id(OBJID[oid]); bb = o.root_scene_node.cumulative_bb
            return np.array(o.transformation.transform_point(bb.center()), float)
        except Exception: pass
    q = state[oid]["pos"]; return np.array([q[0], q[1] + 0.05, q[2]], float)

def line_of_sight(cam, oid):
    # 카메라→물체 광선의 첫 충돌이 그 물체인가. None = 판정 불가(핸들 없음) → depth 후퇴
    if oid not in OBJID: return None
    tg = obj_center(oid); d = tg - np.asarray(cam, float); L = float(np.linalg.norm(d))
    if L < 1e-3: return True
    ray = habitat_sim.geo.Ray(mn.Vector3(*[float(v) for v in cam]), mn.Vector3(*[float(v) for v in d / L]))
    res = sim.cast_ray(ray, max_distance=L + 0.3)
    if not res.has_hits(): return True
    h = res.hits[0]
    return h.object_id == OBJID[oid] or abs(h.ray_distance - L) < 0.15

def evidence_goals(oid, pos, K, D):
    # 거리 D 에서 시선이 닿는 보행점 K 개 → 각각 [멀리(D+1.5) → 가까이(D)] 두 목적지로 심는다.
    # 두 번째 구간을 걸을 때 진행 방향 = 물체 방향이므로 카메라가 물체를 본다.
    tg = obj_center(oid); out = []; start = float(rng.uniform(0, 360))
    for ang in np.linspace(start, start + 360, 16, endpoint=False):
        a = np.radians(ang); dirv = np.array([np.sin(a), 0, np.cos(a)])
        near = sim.pathfinder.snap_point(np.array([pos[0], pos[1], pos[2]]) + D * dirv)
        far = sim.pathfinder.snap_point(np.array([pos[0], pos[1], pos[2]]) + (D + 1.5) * dirv)
        if not (np.isfinite(near).all() and np.isfinite(far).all()): continue
        dn = float(np.hypot(near[0] - pos[0], near[2] - pos[2]))
        if not (0.6 * D <= dn <= 1.4 * D): continue
        if not line_of_sight(np.array(near, float) + np.array([0, 1.5, 0]), oid): continue
        out.append((("pt", np.array(far, float)), ("pt", np.array(near, float))))
        if len(out) >= K: break
    return out

def witness(oid, pos, t, k):
    # 새 자리 주위 8방향 중 보행 가능한 2m 지점에서 물체를 향해 렌더 — 시선이 닿으면 저장
    os.makedirs(os.path.join(args.out, "witness"), exist_ok=True)
    tg = obj_center(oid)
    for ang in range(0, 360, 45):
        a = np.radians(ang)
        q = sim.pathfinder.snap_point(np.array([pos[0] + 2.0 * np.sin(a), pos[1], pos[2] + 2.0 * np.cos(a)]))
        if not np.isfinite(q).all() or np.hypot(q[0] - pos[0], q[2] - pos[2]) < 1.0: continue
        cam = np.array(q, float) + np.array([0, 1.5, 0]); d = tg - cam
        ry = float(np.arctan2(-d[0], -d[2]))
        st = habitat_sim.AgentState(); st.position = q
        st.rotation = np.quaternion(np.cos(ry / 2), 0, np.sin(ry / 2), 0)
        sim.get_agent(0).set_state(st)
        if not line_of_sight(cam, oid): continue
        ob = sim.get_sensor_observations()
        fwd = np.array([-np.sin(ry), 0, -np.cos(ry)]); rgt = np.array([np.cos(ry), 0, -np.sin(ry)])
        zc = float(d @ fwd); u = W / 2 + F * float(d @ rgt) / zc; vv = H / 2 - F * float(d[1]) / zc
        im = Image.fromarray(ob["rgb"][..., :3])
        ImageDraw.Draw(im).ellipse([u - 12, vv - 12, u + 12, vv + 12], outline="red", width=3)
        im.save(os.path.join(args.out, "witness", "%02d_t%d_%s.jpg" % (k, t, objs[oid]["type"].replace(" ", "_"))), quality=85)
        return [round(float(u), 1), round(float(vv), 1)]
    return None
skipped_moves = 0
cur = sim.pathfinder.get_random_navigable_point()
goal = sim.pathfinder.get_random_navigable_point()
path = habitat_sim.ShortestPath(); path.requested_start, path.requested_end = cur, goal
sim.pathfinder.find_path(path)
route = list(path.points) if path.points else [cur]
# ── 매핑 워크: 방마다 들러 360° 스캔 — **이동 전**(live 루프 앞)에 돈다 ──
# 좌표는 rigid COM(obj_center), 가림은 시선 레이캐스트, 박스는 bbox 8꼭짓점 투영.
mapwalk = []
_mapdir = os.path.join(args.out, "map")
if args.remap:
    for _f in glob.glob(os.path.join(_mapdir, "*.jpg")): os.remove(_f)
os.makedirs(_mapdir, exist_ok=True)
mi = 0
for r, pl in polys.items():
    c = np.array(pl).mean(0)
    mp_pt = sim.pathfinder.get_random_navigable_point_near(np.array([c[0], 0.1, c[1]]), 3.0)
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
        box = {}; dist_ = {}
        for oid, v in objs.items():
            pos3 = obj_center(oid) if oid in OBJID else np.array(v["pos"], float)
            d3 = pos3 - cam2
            zc = float(d3 @ f2)
            if not (0.3 < zc < 12): continue
            uu = W / 2 + F * float(d3 @ r2) / zc
            vv = H / 2 - F * float(d3 @ u2) / zc
            if not (5 <= uu < W - 5 and 5 <= vv < H - 5): continue
            if oid in OBJID:
                if not line_of_sight(cam2, oid): continue
            elif abs(float(dep2[int(vv), int(uu)]) - zc) > 0.6: continue
            half = max(12.0, F * 0.25 / max(zc, 0.3))
            b_ = [uu - half, vv - half, uu + half, vv + half]
            if oid in OBJID:
                try:
                    o_ = rom.get_object_by_id(OBJID[oid]); bb = o_.root_scene_node.cumulative_bb; T_ = o_.transformation
                    us, vs = [], []
                    for cx_ in (bb.min[0], bb.max[0]):
                        for cy_ in (bb.min[1], bb.max[1]):
                            for cz_ in (bb.min[2], bb.max[2]):
                                pw_ = np.array(T_.transform_point(mn.Vector3(cx_, cy_, cz_)), float) - cam2
                                z_ = float(pw_ @ f2)
                                if z_ <= 0.1: continue
                                us.append(W / 2 + F * float(pw_ @ r2) / z_); vs.append(H / 2 - F * float(pw_ @ u2) / z_)
                    if len(us) >= 4: b_ = [min(us), min(vs), max(us), max(vs)]
                except Exception: pass
            bx_ = [int(max(0, b_[0])), int(max(0, b_[1])), int(min(W, b_[2])), int(min(H, b_[3]))]
            if bx_[2] - bx_[0] < 8 or bx_[3] - bx_[1] < 8: continue
            box[oid] = bx_; dist_[oid] = round(float(np.hypot(pos3[0] - mp_pt[0], pos3[2] - mp_pt[2])), 2)
        Image.fromarray(ob["rgb"][..., :3]).save(os.path.join(_mapdir, "%04d.jpg" % mi), quality=88)
        mapwalk.append(dict(room=r, yaw=float(yy), box=box,
                            apos=[round(float(mp_pt[0]), 2), round(float(mp_pt[2]), 2)],
                            ctr={o: [round((b[0]+b[2])/2, 1), round((b[1]+b[3])/2, 1)] for o, b in box.items()},
                            dist=dist_))
        mi += 1
print("매핑 워크 %d장 · exemplar 물체 %d" % (mi, len({o for x in mapwalk for o in x["box"]})), flush=True)
if args.remap:
    for _m in mapwalk:                  # 좌표 규약(x 미러·yaw 부호) — 본 경로의 마지막 블록과 동일
        _m["apos"] = [round(-_m["apos"][0], 2), _m["apos"][1]]
        _m["yaw"] = round((-(_m["yaw"]) + 180.0) % 360, 1)
    _gp = os.path.join(args.out, "gt.json"); _g = json.load(open(_gp))
    _g["map"] = mapwalk; _g["_remap"] = True
    json.dump(_g, open(_gp, "w"))
    print("remap 완료 → %s (map %d장)" % (_gp, len(mapwalk)), flush=True); sys.exit(0)

ri, t = 0, 0
yaw = 0.0
_retry = 0
while t < args.frames:
    if ri + 1 >= len(route):
        if forced_goals or (MOVE and MOVE.get("dwell")):
            # 대본 목적지가 있으면 그 방, 아니면 방 체류 가중(③ 제외 방은 뺀다)
            r_pick = None; goal = None
            if forced_goals:
                fg = forced_goals.pop(0)
                if isinstance(fg, tuple) and fg[0] == "pt": goal = np.array(fg[1], float)
                else: r_pick = fg
            else:
                rs_ = [r for r in polys if r not in excluded_rooms] or list(polys)
                wv = np.array([MOVE["dwell"].get(_rtype(r), 0.1) for r in rs_], float)
                r_pick = rs_[int(rng.choice(len(rs_), p=wv / wv.sum()))]
            if goal is None:
                c_ = np.array(polys[r_pick]).mean(0)
                goal = sim.pathfinder.get_random_navigable_point_near(
                    np.array([c_[0], float(cur[1]), c_[1]]), 3.0)
            if not np.isfinite(goal).all():
                goal = sim.pathfinder.get_random_navigable_point()
        else:
            goal = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath(); path.requested_start, path.requested_end = cur, goal
        if not sim.pathfinder.find_path(path) or len(path.points) < 2: continue
        if excluded_rooms and _retry < 30 and any(
                room_at(float(q[0]), float(q[2])) in excluded_rooms for q in path.points):
            _retry += 1; continue          # ③ 대본: 제외 방을 **경유**하는 경로도 버린다
        if hidden_oids and _retry < 30:
            # ③ 대본: 경로 위 어느 지점(눈높이)에서든 ③ 물체가 12m 안에서 **시선에 들어오면** 버린다
            # (옷장 속 시계가 침실 문 너머로 보이던 실측 — 방 제외만으로는 못 막는다)
            _pts = [np.array(q, float) for q in path.points]
            _dense = []
            for _a, _b in zip(_pts[:-1], _pts[1:]):
                _n = max(1, int(np.linalg.norm(_b - _a) / 0.75))
                _dense += [_a + (_b - _a) * (k_ / _n) for k_ in range(_n)]
            _leak = False
            for q in _dense + [_pts[-1]]:
                camq = q + np.array([0, 1.5, 0])
                for ho in hidden_oids:
                    tg = obj_center(ho)
                    if np.linalg.norm(tg - camq) < 12.0 and line_of_sight(camq, ho):
                        _leak = True; break
                if _leak: break
            if _leak: _retry += 1; continue
        _retry = 0
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
            oid, dest, role = plan[t]
            pl = np.array(polys[dest]); c = pl.mean(0)
            np_ = sim.pathfinder.get_random_navigable_point_near(np.array([c[0], p[1], c[1]]), 2.0)
            h = obj_handle(oid)
            o = rom.get_object_by_handle(h) if h else None
            if not np.isfinite(np_).all() or o is None:
                # 렌더에 반영 못 하는 이동은 **기록하지 않는다** (종전: GT 만 갱신 → 거짓 ②)
                skipped_moves += 1
                print("  ⚠ 이동 건너뜀(핸들/보행점 없음): %s" % oid, flush=True)
            else:
                off = support_offset(o, state[oid]["pos"])
                if on_floor_originally(o, state[oid]["pos"]):
                    # 바닥 높이는 navmesh 점이 아니라 **실제 바닥 메시**(내려쏘기)로 — navmesh 는
                    # 발높이 근사라 10~20cm 어긋나 받침 검사에 걸렸다
                    _fh = [h for h in _down_hits(np_[0], float(np_[1]) + 1.0, np_[2], o.object_id)
                           if float(h.normal[1]) > 0.7 and abs(float(h.point[1]) - float(np_[1])) < 0.5]
                    x1, z1 = float(np_[0]), float(np_[2])
                    y1 = float(_fh[0].point[1]) if _fh else float(np_[1])
                else:
                    x1, z1, y1 = pick_receptacle(o, np_)
                _orig_tr = o.translation
                newp = place_at(o, x1, z1, y1, off)
                sup = is_supported(o, off)
                _st = sim.get_agent(0).get_state()
                wit = witness(oid, newp, t, len(moves)) if sup else None
                sim.get_agent(0).set_state(_st)
                real = room_at(newp[0], newp[2])          # 실제 좌표로 방 재판정
                if real == obj_room[oid]:
                    sup = False                           # 같은 방 = 이동이 아니다 → 취소
                if sup and wit is not None:
                    state[oid]["pos"] = newp              # GT = 실제 배치 좌표
                    wit_file = "witness/%02d_t%d_%s.jpg" % (len(moves), t, objs[oid]["type"].replace(" ", "_"))
                    moves.append(dict(t=t, oid=oid, frm=obj_room[oid], to=real, intended=dest,
                                      pos=[round(v, 3) for v in newp], witness=True, supported=True,
                                      witness_file=wit_file, witness_ctr=wit))
                    moves[-1]["role"] = role
                    if role == "c3":
                        excluded_rooms.add(real); forced_goals.append(moves[-1]["frm"])
                        hidden_oids.append(oid)
                    else:
                        if args.evidence:
                            _K, _D = args.evidence.split(":"); _K, _D = int(_K), float(_D)
                            _evs = evidence_goals(oid, newp, _K, _D)
                            _low = sorted(polys, key=lambda r: (MOVE or {}).get("dwell", {}).get(_rtype(r), 0.1))
                            for _j, (_gf, _gn) in enumerate(_evs):
                                forced_goals += [_gf, _gn]
                                if _j < len(_evs) - 1 and _low:   # 방문 사이에 다른 방을 끼워 별개 방문으로
                                    forced_goals.append(_low[0] if _low[0] != real else (_low[1] if len(_low) > 1 else _low[0]))
                            moves[-1]["evidence_visits"] = len(_evs)
                            if not _evs: forced_goals.append(real)   # 시선 지점 없으면 방 방문으로 후퇴
                        else:
                            forced_goals.append(real)
                    obj_room[oid] = real
                else:
                    # 기하 게이트 실패(허공/박힘/시선 없음) → **원위치로 되돌리고 기록하지 않는다**
                    o.translation = _orig_tr
                    skipped_moves += 1
                    print("  ⚠ 이동 취소(받침 %s·시선 %s): %s" % (sup, wit is not None, oid), flush=True)
        obs = sim.get_sensor_observations()
        dep = obs["dep"]
        cam = np.array(p) + np.array([0, 1.5, 0])
        fwd = np.array([-np.sin(np.radians(yaw)), 0, -np.cos(np.radians(yaw))])
        rgt = np.array([np.cos(np.radians(yaw)), 0, -np.sin(np.radians(yaw))])
        up = np.array([0, 1.0, 0])
        vis, ctr, dist, anch = [], {}, {}, {}
        for oid, v in state.items():
            d3 = (obj_center(oid) if oid in OBJID else np.array(v["pos"])) - cam
            zc = float(d3 @ fwd)
            if not (0.3 < zc < 12): continue
            u = W / 2 + F * float(d3 @ rgt) / zc
            vv = H / 2 - F * float(d3 @ up) / zc
            if not (5 <= u < W - 5 and 5 <= vv < H - 5): continue
            _los = line_of_sight(cam, oid)
            if _los is None:
                if abs(float(dep[int(vv), int(u)]) - zc) > 0.6: continue     # depth 후퇴
            elif not _los: continue                                          # 시선 차단(가구 속 등)
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
for _m in moves:                         # 이동 후 좌표도 같은 규약으로 (2026-09-02: 빠져 있었다)
    if _m.get("pos"): _m["pos"] = _mx(_m["pos"])

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

print("이동 기록 %d · 렌더 미반영 건너뜀 %d · 증인 렌더 OK %d/%d"
      % (len(moves), skipped_moves, sum(1 for m in moves if m.get("witness")), len(moves)), flush=True)
