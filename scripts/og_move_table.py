#!/usr/bin/env python3
"""OmniGibson 어휘용 시나리오 사전확률 표 (2026-09-10) — hssd_move.json(LLM 생성)을 OG 타입·방 이름으로 사상하고, 없는 타입은 휴리스틱으로 채운다.
27B mlx 생성은 이 머신에서 안 돼(py3.9 mlx_lm 은 qwen3_5 미지원, py3.11 판은 출력 깨짐) 파생표로 대신한다 — 이동성 게이트(ABS_MOB_ALL)와 belief 사전확률이 OG 이름을 읽게 하는 것이 목적.
  python scripts/og_move_table.py <rows_D.jsonl 또는 vocab.json> data/og_move.json"""
import json, re, sys, collections
src, out = sys.argv[1], sys.argv[2]; H = json.load(open("data/hssd_move.json"))
if src.endswith(".jsonl"):
    rs = [json.loads(l) for l in open(src)]; objs = sorted(set(r["type"] for r in rs)); rooms = sorted({re.sub(r"_\d+$", "", v) for r in rs for k in ("tgt", "record", "ans") for v in [r.get(k)] if v})
else:
    V = json.load(open(src)); objs, rooms = V["objects"], sorted({re.sub(r"[._]\d+$", "", r) for r in V["rooms"]})
FIXED = ("table", "counter", "countertop", "bed", "stove", "washer", "dryer", "oven", "sink", "fridge", "refrigerator", "shelf", "shelves", "cabinet", "sofa", "couch", "desk", "dresser", "wardrobe", "bathtub", "toilet", "lamp", "tv", "television", "mirror", "carpet", "rug", "piano", "bookcase", "bench", "nightstand", "microwave", "dishwasher", "range", "fireplace", "door", "window", "curtain", "armchair", "chair", "stool", "ottoman", "crib", "loudspeaker", "radiator", "shower", "hammock", "burner", "clock")
SYN = {"coffee_cup": "mug", "pot_plant": "potted plant", "standing_tv": "tv", "swivel_chair": "chair", "clothes_dryer": "washer dryer", "drop_in_sink": "sink", "furniture_sink": "sink", "multi_station_furniture_sink": "sink", "gas_fireplace": "fireplace", "openable_window": "window", "breakfast_table": "table", "grandfather_clock": "clock", "shower_stall": "shower", "board_game": "chess set"}
ROOM_MAP = {"childs_room": "bedroom", "pantry_room": "kitchen", "corridor": "hallway", "garden": "outdoor", "outside": "outdoor", "staircase": "hallway", "storage_room": "utilityroom", "utility_room": "utilityroom", "entryway": "entryway", "living_room": "living room", "dining_room": "dining room"}
def hkey(t):
    k = t.replace("_", " ")
    if k in H["mobility"]: return k
    s = SYN.get(t)
    if s and s in H["mobility"]: return s
    return None
mob = {}; src_cnt = collections.Counter()
for t in objs:
    k = hkey(t)
    if k: mob[t] = H["mobility"][k]; src_cnt["hssd"] += 1
    elif any(w in t for w in FIXED): mob[t] = 0.0; src_cnt["heur-fixed"] += 1
    else: mob[t] = 0.5; src_cnt["heur-movable"] += 1
dwell = {}
for r in rooms:
    hr = ROOM_MAP.get(r, r); dwell[r] = H["dwell"].get(hr, H["dwell"].get(r, 0.1))
dest = {}
for t in objs:
    k = hkey(t); hd = H["dest"].get(k, {}) if k else {}
    d = {}
    for r in rooms:
        hr = ROOM_MAP.get(r, r); d[r] = hd.get(hr, hd.get(r, 0.0))
    if sum(d.values()) <= 0: d = {r: dwell[r] for r in rooms}           # 미등재 타입: 체류 분포로
    z = sum(d.values()); dest[t] = {r: round(v / z, 4) for r, v in d.items()}
json.dump(dict(dwell=dwell, mobility=mob, dest=dest, _source="derived from hssd_move.json (LLM) by name mapping + heuristics, 2026-09-10; objects %d rooms %d; sources %s" % (len(objs), len(rooms), dict(src_cnt))), open(out, "w"), indent=1, ensure_ascii=False)
print("OG_MOVE_DONE %s · objects %d (%s) · rooms %d" % (out, len(objs), dict(src_cnt), len(rooms)))
