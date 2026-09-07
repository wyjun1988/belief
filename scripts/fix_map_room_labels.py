#!/usr/bin/env python3
"""매핑워크 프레임 방 라벨을 **카메라 위치**(평면도 폴리곤) 기준으로 고친다 — hab_episode `--map-travel` 버그 보정.
    python scripts/fix_map_room_labels.py data/hssd40_c3 [data/hssd20_c3 ...]
버그: 지점 i→i+1 이동 프레임에 **출발 지점의 방**이 붙었다(hab_episode.py `_poses.append((_r, ...))`). 이동 프레임 지도가 있는
데이터셋에서 19~41% 가 틀린 라벨이었고(2026-09-05 밤 발견), 이 라벨이 sfm_reloc --align sites(지점 라벨 정렬)·build_initmap(프레임 방)·
room_embed(노드 라벨)에 전부 들어갔다. 배포 그림(사용자가 걸으며 방 이름을 말한다)에서도 라벨은 사람의 **현재 위치**를 따르므로
위치 기준이 맞다. 원본 라벨은 `room_walk` 로 보존, 폴리곤 밖이면 최근접 폴리곤."""
import json, os, sys, glob
def pip(pt, poly):
    x, z = pt; ins = False; n = len(poly)
    for i in range(n):
        x1, z1 = poly[i][0], poly[i][-1]; x2, z2 = poly[(i + 1) % n][0], poly[(i + 1) % n][-1]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
    return ins
tot = chg = 0
for root in sys.argv[1:]:
    for hd in sorted(glob.glob(os.path.join(root, "house_*"))):
        gp = os.path.join(os.path.realpath(hd), "gt.json"); g = json.load(open(gp)); polys = (g.get("scene_meta") or {}).get("polys") or {}
        if not polys or not g.get("map"): continue
        def room_of(pt):
            for r, pl in polys.items():
                if pip(pt, pl): return r
            return min(polys, key=lambda r: min((pt[0] - v[0]) ** 2 + (pt[1] - v[-1]) ** 2 for v in polys[r]))
        c = 0
        for m in g["map"]:
            if "room_walk" not in m: m["room_walk"] = m["room"]
            r = room_of(m["apos"])
            if r != m["room"]: c += 1
            m["room"] = r
        g["_map_room_by_pos"] = True
        json.dump(g, open(gp, "w"))
        tot += len(g["map"]); chg += c
        print("  %s/%s map %d · 고친 라벨 %d (%.0f%%)" % (os.path.basename(root), os.path.basename(hd), len(g["map"]), c, 100 * c / len(g["map"])), flush=True)
print("합계 map %d · 고침 %d (%.0f%%)" % (tot, chg, 100 * chg / max(tot, 1)))
