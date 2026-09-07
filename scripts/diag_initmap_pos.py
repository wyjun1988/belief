#!/usr/bin/env python3
"""초기맵 인스턴스 **위치** 정확도(GT 는 채점에만): 타입 단일·안 움직인 타겟마다 최고 w 인스턴스 / 가장 가까운 인스턴스의 평면 오차와 방 정답.
    python scripts/diag_initmap_pos.py <THOR_ROOT> initmap_owl.json [initmap_owl_pts.json ...]
③ 경로의 첫 병목이 기록 자리 오차(2026-09-07: ep2 중앙 2.1 m, ≤1 m 25%)라 이 지표로 초기맵 변형을 비교한다."""
import json, glob, os, sys, math, collections, numpy as np
root = sys.argv[1]; files = sys.argv[2:]
res = {f: [] for f in files}
for hd in sorted(glob.glob(os.path.join(root, "house_*"))):
    hdr = os.path.realpath(hd); g = json.load(open(hdr + "/gt.json"))
    gf = hdr + "/room_groups.json"; gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}; grp = lambda r: gm.get(r, r) if r else r
    cnt = collections.Counter(v["type"] for v in g["gt0"].values()); mv = {m["oid"] for m in g["moves"]}
    for f in files:
        p = os.path.join(hdr, f)
        if not os.path.exists(p): continue
        inst = collections.defaultdict(list)
        for it in json.load(open(p)):
            if it.get("pos"): inst[it["type"]].append(it)
        for oid, v0 in g["gt0"].items():
            if oid in mv or cnt[v0["type"]] > 1 or not v0["room"]: continue
            gs = (v0["pos"][0], v0["pos"][2]); its = inst.get(v0["type"])
            if not its: res[f].append(dict(miss=True)); continue
            best = max(its, key=lambda it: it["w"]); near = min(its, key=lambda it: math.hypot(it["pos"][0]-gs[0], it["pos"][1]-gs[1]))
            res[f].append(dict(miss=False, eb=math.hypot(best["pos"][0]-gs[0], best["pos"][1]-gs[1]), en=math.hypot(near["pos"][0]-gs[0], near["pos"][1]-gs[1]), room=grp(best["room"]) == grp(v0["room"]), n_inst=len(its)))
print("%-26s %5s %5s | 최고w 오차 중앙 · ≤1m · ≤0.5m | 근접 오차 중앙 · ≤1m | 방 정답 | 인스턴스/타입" % ("파일", "타겟", "누락"))
for f in files:
    R = [r for r in res[f] if not r["miss"]]; eb = np.array([r["eb"] for r in R]); en = np.array([r["en"] for r in R])
    if not R: print(f, "결과 없음"); continue
    print("%-26s %5d %5d | %.2f m · %.2f · %.2f | %.2f m · %.2f | %.2f | %.1f" % (f, len(res[f]), sum(r["miss"] for r in res[f]), np.median(eb), np.mean(eb <= 1), np.mean(eb <= 0.5), np.median(en), np.mean(en <= 1), np.mean([r["room"] for r in R]), np.mean([r["n_inst"] for r in R])))
