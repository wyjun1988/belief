#!/usr/bin/env python3
"""모든 구성을 **완전히 같은 자**로 잰다 — 보고 표의 근거 스크립트.

    $P scripts/eval_fair.py gtdepth t23 da3lc_aligned full_aligned da3zone_aligned

⚠️ 구성마다 지표가 다르면 비교가 성립하지 않는다. 실제로 한 번 어긋났었다 —
GT-seg 구성은 카테고리 기반, FastSAM 구성은 외형 재식별 기반으로 나란히 놓는 바람에
"세그를 모델로 바꿨더니 올랐다"처럼 보였다. 같은 자로 재면 맞춘 개수가 66개로 동일했다.

그래서 세 가지를 고정한다:

  · 후보 선택: 전부 카테고리 기반
  · 질의 시점: **모든 구성이 답할 수 있는 질의만** 공통으로 추림 (분모가 같아진다)
  · 정답: GT 좌표를 기준 구역지도(GT 기하·GT 시드)에 넣은 값 — 전 구성 동일
"""
import json, os, sys

import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.environ.get("KX_SEQ", os.path.join(
    ROOT, "data", "seq", "Apartment_release_decoration_seq137_M1292"))

from kx.eval.node_belief import candidates, answer
from kx.eval.room_belief import load_regions
from kx.graph.regions import assign

gt = json.load(open(D + "/gt/objects.json"))["instances"]
meta = json.load(open(D + "/graph_gtdepth.json"))["regions"]
ref = load_regions(np.load(D + "/regions_gtdepth.npz"), meta["zone_names"], meta["up"])
zf = lambda p: assign(ref, p)[1]
TICK = 50

tags = sys.argv[1:]
graphs = {t: json.load(open(f"{D}/graph_{t}.json")) for t in tags}
N = min(g["n_frames"] for g in graphs.values())

# 공통 질의 목록: (GT 인스턴스, 틱). 모든 구성이 답을 낼 수 있는 것만.
Q = []
for iid, rec in gt.items():
    if rec["motion_type"] != "dynamic" or not rec["moves"]:
        continue
    P = np.array(rec["positions"])
    fm = min(m["start_idx"] for m in rec["moves"])
    for t in range(0, min(len(P), N), TICK):
        if zf(P[t]) is None:
            continue
        ok = True
        for g in graphs.values():
            c = candidates(g, by="category", key=rec.get("category"))
            if not c or answer(g, c, t, zf) is None:
                ok = False; break
        if ok:
            Q.append((iid, t, zf(P[t]), t >= fm))

print("공통 질의 %d개 (동적 물체 %d개 · 틱 %d프레임 간격)"
      % (len(Q), len({q[0] for q in Q}), TICK))
print("%-18s %-12s %-9s %-12s %-9s %s" % ("구성", "맞음/질의", "정확도", "이동후", "후보수", "위치오차"))
for t_ in tags:
    g = graphs[t_]
    ok = okm = nm = 0; err = []; nc = []
    for iid, t, gz, mv in Q:
        c = candidates(g, by="category", key=gt[iid].get("category"))
        a = answer(g, c, t, zf)
        hit = a["zone"] == gz
        ok += hit; nc.append(a["n_candidates"])
        err.append(float(np.linalg.norm(np.array(a["position"]) - np.array(gt[iid]["positions"])[t])))
        if mv: nm += 1; okm += hit
    print("%-18s %-12s %-9.3f %-12s %-9.0f %.3f"
          % (t_, "%d / %d" % (ok, len(Q)), ok / len(Q),
             "%d / %d = %.3f" % (okm, nm, okm / max(nm, 1)), np.median(nc), np.median(err)))
