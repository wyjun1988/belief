#!/usr/bin/env python3
"""GT 초기맵(⚠️ 진단 전용 사다리 재료): gt0 의 스캔 시점 위치·방을 그대로 초기맵 형식으로 적는다.
`INITMAP_FILE=initmap_gt.json` 로 벤치하면 "기록이 완벽하면 얼마인가"를 잰다(§166-32: HSSD 0.777→0.939).
  THOR_ROOT=<데이터 루트> python scripts/make_initmap_gt.py"""
import json, glob, os
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); OUT = os.environ.get("INITMAP_GT_OUT", "initmap_gt.json")
n_h = n_i = 0
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hdr = os.path.realpath(hd); gp = os.path.join(hdr, "gt.json")
    if not os.path.exists(gp): continue
    g = json.load(open(gp))
    out = [dict(type=v["type"], room=v["room"], w=1.0, pos=[round(v["pos"][0], 2), round(v["pos"][2], 2)], n=1, oid=o)
           for o, v in g["gt0"].items() if v.get("room") and v.get("pos")]
    json.dump(out, open(os.path.join(hdr, OUT), "w"), ensure_ascii=False); n_h += 1; n_i += len(out)
print("GT 초기맵 %d채 · 인스턴스 %d → %s" % (n_h, n_i, OUT))
