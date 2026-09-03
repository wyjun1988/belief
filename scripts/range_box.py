#!/usr/bin/env python3
"""박스 크기 레인지 — 카테고리 크기 사전확률 × OWL 박스 높이로 거리를 내고 DA(아핀)와 융합. geo_depth 형식 출력.

    THOR_ROOT=... A3_PREFIX=... SCORES=t1.jsonl DA_JSONL=geo_depth_v3c.jsonl OUT_JSONL=range_fused.jsonl python scripts/range_box.py

d_box = f · H_cat / h_px  (H_cat: 카테고리 대표 높이 m — 손 사전확률, GT 아님). 근거리일수록 박스가 커서 정확하고
DA 는 근거리에서 편향(§134 <2m 0.38)이므로: d = d_box (d_box < X_NEAR) / DA (그 외) / 둘 다 있으면 역분산 가중.
평가: GT dist 대조 거리 구간별 상대오차 — 박스만 · DA만 · 융합.
"""
import glob, json, os
import numpy as np
ROOT = os.environ.get("THOR_ROOT"); A3P = os.path.expanduser(os.environ.get("A3_PREFIX"))
SC = os.environ.get("SCORES"); DAJ = os.environ.get("DA_JSONL"); OUT = os.environ.get("OUT_JSONL", "/tmp/range_fused.jsonl")
W = float(os.environ.get("FRAME_W", "768")); F = W / 2; X_NEAR = float(os.environ.get("X_NEAR", "2.5"))
# 카테고리 대표 높이(m) — 손 사전확률. 없으면 0.35
H_CAT = {"table lamp": 0.5, "floor lamp": 1.5, "trashcan": 0.5, "laptop": 0.25, "book": 0.25, "plate": 0.03, "tray": 0.05,
         "alarm clock": 0.12, "mantel clock": 0.25, "wall clock": 0.3, "potted plant": 0.6, "picture frame": 0.3, "bottle": 0.25,
         "drinkware": 0.12, "vase": 0.3, "candle": 0.15, "cushion": 0.4, "plush toy": 0.3, "shoes": 0.1, "clothing": 0.5,
         "toiletry": 0.15, "bowl": 0.1, "mobile phone": 0.15, "phone": 0.15, "kettle": 0.25, "toaster": 0.2, "stand": 0.8,
         "bench": 0.5, "chest of drawers": 0.9, "coffee maker": 0.35, "ceiling lamp": 0.4, "wall lamp": 0.3}
DA = {}
if DAJ and os.path.exists(DAJ):
    for l in open(DAJ):
        d = json.loads(l); DA[(d["house"], d["t"], d["oid"])] = d["d"]
recs = [json.loads(l) for l in open(SC)]
errs = {"box": [], "da": [], "fused": []}; out = open(OUT, "w"); n_out = 0
for rc in recs:
    hn, oid = rc["house"], rc["oid"]
    hd = [d for d in glob.glob(ROOT + "/house_*") if os.path.basename(os.path.realpath(d)) == hn]
    if not hd: continue
    g = json.load(open(hd[0] + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    za = np.load(A3P + hn + ".npz", allow_pickle=True); ts, vocab = za["ts"], list(za["vocab"])
    if "bx" not in za.files: continue
    bx = za["bx"]; ty = g["gt0"][oid]["type"]; ti = vocab.index(ty); Hc = H_CAT.get(ty, 0.35)
    for i, s_ab, s_ac in rc["scored"]:
        t = int(ts[i]); h_px = float(bx[i, ti][3]) * W
        d_box = F * Hc / max(h_px, 4.0)
        d_da = DA.get((hn, t, oid))
        if d_da is None: d = d_box; src = "box"
        elif d_box < X_NEAR: d = d_box; src = "box"                       # 근거리: 박스
        else: d = float(np.exp(0.5 * (np.log(d_box) + np.log(d_da)))); src = "geo"   # 원거리: 기하평균
        gt = (live.get(t, {}).get("dist") or {}).get(oid)
        out.write(json.dumps(dict(house=hn, t=t, oid=oid, d=round(float(d), 2), d_box=round(float(d_box), 2), d_da=d_da, src=src, gt=gt)) + "\n"); n_out += 1
        if gt:
            errs["box"].append((gt, abs(d_box - gt) / gt))
            if d_da is not None: errs["da"].append((gt, abs(d_da - gt) / gt))
            errs["fused"].append((gt, abs(d - gt) / gt))
out.close()
def rep(name):
    e = np.array(errs[name]) if errs[name] else np.zeros((0, 2))
    if not len(e): return "%-6s (없음)" % name
    b = lambda m: "%.2f(n=%d)" % (np.median(e[m, 1]), m.sum()) if m.any() else "—"
    return "%-6s 전체 %.2f · <2m %s · 2-5m %s · 5m+ %s" % (name, np.median(e[:, 1]), b(e[:, 0] < 2), b((e[:, 0] >= 2) & (e[:, 0] < 5)), b(e[:, 0] >= 5))
print("\n".join(rep(k) for k in ("box", "da", "fused"))); print("→ %s (%d)" % (OUT, n_out))
