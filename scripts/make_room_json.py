#!/usr/bin/env python3
"""reloc_hloc --room-json 재료: 지도 프레임 방(gt.json map[k].room, 스캔 라벨 = 사용자 입력)과 라이브 프레임 방(ROOM_JSONL 임베딩 판정, 무GT).
  THOR_ROOT=... ROOM_JSONL=... OUT_DIR=... python scripts/make_room_json.py  → <OUT_DIR>/room_<house>.json"""
import json, glob, os, collections
ROOT = os.environ["THOR_ROOT"]; RJ = os.environ["ROOM_JSONL"]; OUT = os.environ.get("OUT_DIR", "/tmp"); os.makedirs(OUT, exist_ok=True)
rooms = collections.defaultdict(dict)
for l in open(RJ): r = json.loads(l); rooms[r["house"]][int(r["t"])] = r["room"]
n = 0
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hn = os.path.basename(hd); g = json.load(open(os.path.join(os.path.realpath(hd), "gt.json")))
    gf = os.path.join(os.path.realpath(hd), "room_groups.json"); gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}; grp = lambda x: gm.get(x, x) if x else x
    scan = [grp(m.get("room")) for m in g["map"]]; live = {str(t): grp(r) for t, r in rooms.get(hn, {}).items()}
    json.dump(dict(scan=scan, live=live), open(os.path.join(OUT, "room_%s.json" % hn), "w")); n += 1
print("room-json %d채 → %s" % (n, OUT))
