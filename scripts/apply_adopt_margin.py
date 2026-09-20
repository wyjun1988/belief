#!/usr/bin/env python3
"""채택 판정기 마진으로 검증기 점수 파일을 거른다 — 챔피언 채택 필터 생성 (§166-88 절차의 2단계).

  python scripts/apply_adopt_margin.py <margin.jsonl> <문턱> <out.jsonl> [--verify <t1_all.jsonl>] [--cache-prefix <hs2_a_>]

원리: 검증기(exp_t1_verify_mlx)를 통과한 프레임(s_ab ≥ VTH, s_ac ≥ VTH2) 중 채택 판정기 마진이
문턱 미만인 것을 버린다. 마진은 (house, oid, t) 로 키를 맞춘다 — scored 항목의 첫 값은 **앵커 색인 i** 이고
t = ts[i] (캐시의 ts). 이 스크립트는 원래 스크래치패드에 있다가 사라져 벤치 4판이 빈손이 됐다(2026-09-20)
— 저장소로 옮겼다.
"""
import argparse, collections, json, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("margin"); ap.add_argument("th", type=float); ap.add_argument("out")
ap.add_argument("--verify", default=os.path.expanduser("~/khcache/bench-v2full/scores/t1_all.jsonl"))
ap.add_argument("--cache-prefix", default=os.path.expanduser("~/khcache/bench-v2full/cache/hs2_a_"))
ap.add_argument("--vth", type=float, default=2.069); ap.add_argument("--vth2", type=float, default=0.887)
a = ap.parse_args()
mg = {}
for ln in open(os.path.expanduser(a.margin)):
    d = json.loads(ln); mg[(d["house"], d["oid"], int(d["t"]))] = float(d.get("margin", 0))
Z = {}; st = collections.Counter()
with open(os.path.expanduser(a.out), "w") as fo:
    for ln in open(os.path.expanduser(a.verify)):
        d = json.loads(ln); h, oid = d["house"], d["oid"]
        if h not in Z:
            try: Z[h] = np.load(a.cache_prefix + h + ".npz", allow_pickle=True)
            except Exception: Z[h] = None
        if Z[h] is None: fo.write(ln); st["캐시없음(그대로)"] += 1; continue
        ts = Z[h]["ts"]; keep = []
        for e in d.get("scored") or []:
            if not (e[1] >= a.vth and (len(e) < 3 or e[2] >= a.vth2)): keep.append(e); continue
            i = int(e[0])
            if i >= len(ts): keep.append(e); continue
            m = mg.get((h, oid, int(ts[i])))
            if m is None: keep.append(e); st["마진없음"] += 1; continue
            if m >= a.th: keep.append(e); st["통과"] += 1
            else: st["버림"] += 1
        d["scored"] = keep; fo.write(json.dumps(d) + "\n")
print("APPLY_DONE 문턱 %+.2f · %s → %s" % (a.th, dict(st), a.out), flush=True)
