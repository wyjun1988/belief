# 검증: Qwen 움직임 사전확률이 실제로 데이터를 바꿨나. 균등이면 아래가 전부 균등이어야 한다.
import os
ROOT = os.environ.get("THOR_ROOT", "data/thor3")
import json, glob, os, numpy as np
from collections import Counter
MV = json.load(open("data/thor_move.json"))
dw = Counter(); dest = Counter(); mob = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    g = json.load(open(hd + "/gt.json"))
    rt = g["room_types"]
    for m in g["live"]:
        dw[rt.get(m["room"], "?")] += 1
    for mv in g["moves"]:
        dest[rt.get(mv["to"], "?")] += 1
        t = g["gt0"].get(mv["oid"], {}).get("type")
        if t: mob.append(MV["mobility"].get(t, .5))
n = sum(dw.values())
print("=== 방 체류 (프레임 %d장, 주택 %d채) ===" % (n, len(glob.glob(ROOT + "/house_*"))))
for r in ("Kitchen", "LivingRoom", "Bedroom", "Bathroom"):
    print("  %-11s 실측 **%.2f**  ← Qwen %.2f" % (r, dw[r]/n, MV["dwell"].get(r, 0)))
m = sum(dest.values())
print("=== 이동 목적지 (%d건) ===" % m)
for r in ("Kitchen", "LivingRoom", "Bedroom", "Bathroom"):
    print("  %-11s 실측 **%.2f**" % (r, dest[r]/m))
print("=== 옮겨진 물체의 이동성향 ===")
allm = list(MV["mobility"].values())
print("  옮겨진 것 중앙 **%.2f** · 전체 물체 중앙 %.2f (균등이면 같아야 한다)"
      % (np.median(mob), np.median(allm)))
