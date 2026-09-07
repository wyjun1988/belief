#!/usr/bin/env python3
"""초기맵 변형 비교 — 벤치를 돌리지 않고 초기맵 파일만으로 세 지표를 잰다(그룹 단위, GT 는 채점에만).
    python scripts/diag_initmap_eval.py data/hssd40_c3 initmap_owl.json initmap_owl_v2.json
  (a) 이동 물체(③·② 후보)의 **이동 전 방** 기록 일치 — 틀리면 ③확인기회X(기록방오류) 로 빠진다
  (b) 타입 단위 방배정(어느 인스턴스든 GT 방과 겹치면 정답)
  (c) 안 움직인 물체 인스턴스 단위: 초기맵 최대 w 방 = 그 인스턴스의 방 (①의 '초기맵만' 근사)"""
import json, glob, os, sys, collections
root = sys.argv[1]; files = sys.argv[2:]
def load(hd, fn):
    p = os.path.join(hd, fn); return json.load(open(p)) if os.path.exists(p) else None
res = {f: collections.Counter() for f in files}
for hd in sorted(glob.glob(os.path.join(root, "house_*"))):
    g = json.load(open(hd + "/gt.json")); gf = hd + "/room_groups.json"; grp = json.load(open(gf))["groups"] if os.path.exists(gf) else {}; G = lambda r: grp.get(r, r)
    gt_rooms = collections.defaultdict(set)
    for v in g["gt0"].values(): gt_rooms[v["type"]].add(G(v["room"]))
    moved = {(m.get("oid") or m.get("id")) for m in g["moves"]}
    for fn in files:
        im = load(hd, fn)
        if im is None: continue
        byt = collections.defaultdict(list)
        for e in im: byt[e["type"]].append(e)
        top = {t: G(max(es, key=lambda e: e["w"])["room"]) for t, es in byt.items()}
        allr = {t: {G(e["room"]) for e in es} for t, es in byt.items()}
        for mv in g["moves"]:
            oid = mv.get("oid") or mv.get("id"); v = g["gt0"].get(oid, {}); t = v.get("type"); frm = G(mv.get("from") or v.get("room"))
            res[fn]["a_n"] += 1; res[fn]["a_ok"] += (top.get(t) == frm); res[fn]["a_missing"] += (t not in top)
        for t, rs in gt_rooms.items():
            if t in allr: res[fn]["b_n"] += 1; res[fn]["b_ok"] += bool(allr[t] & rs)
            else: res[fn]["b_missing"] += 1
        for oid, v in g["gt0"].items():
            if oid in moved: continue
            t = v["type"]; res[fn]["c_n"] += 1
            if t in top: res[fn]["c_ok"] += (top[t] == G(v["room"]))
            else: res[fn]["c_missing"] += 1
for fn in files:
    r = res[fn]
    if not r["a_n"]: print("%s: 없음" % fn); continue
    print("%-22s (a) 이동물체 이동전 방 일치 %2d/%2d (%.2f, 타입없음 %d) · (b) 타입 방배정 %.3f (n=%d, 타입없음 %d) · (c) 정지 인스턴스 최대w 일치 %.3f (n=%d, 타입없음 %d)" % (
        fn, r["a_ok"], r["a_n"], r["a_ok"] / r["a_n"], r["a_missing"], r["b_ok"] / max(r["b_n"], 1), r["b_n"], r["b_missing"], r["c_ok"] / max(r["c_n"], 1), r["c_n"], r["c_missing"]))
