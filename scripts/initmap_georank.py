#!/usr/bin/env python3
"""P0-C 기하 재순위: 후보 군집마다 raw 투영점(카메라 위치 + 방위)의 **광선 교차 잔차**를 잰다.
실물 하나를 여러 자리에서 봤다면 광선들이 한 점 근처에서 만나고(잔차 작음), 오검출 군집은 카메라마다 다른 것을 봐서 안 만난다.
점수 s = 1/(1+잔차) × 카메라 위치 다양성(≥2 시점) → w' = w × (0.2 + s_norm)^ALPHA.
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json RAW=initmap_raw.json python scripts/initmap_georank.py → <house>/<initmap>_rr_geo.json"""
import os, json, glob, math, collections, numpy as np
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); RAW = os.environ.get("RAW", "initmap_raw.json"); ALPHA = float(os.environ.get("ALPHA", "1.0")); R = float(os.environ.get("R", "2.0"))
def ray_residual(pts):
    """pts: [(x,z,score,ax,az)] → 최소제곱 교차점과 그 점까지 광선 거리의 가중 RMS (m), 시점 수"""
    A = np.zeros((2, 2)); b = np.zeros(2); cams = set(); rows = []
    for x, z, s, ax, az in pts:
        d = np.array([x - ax, z - az]); n = np.linalg.norm(d)
        if n < 0.3: continue
        d /= n; P = np.eye(2) - np.outer(d, d); c = np.array([ax, az]); A += s * P; b += s * (P @ c); rows.append((c, d, s)); cams.add((round(ax, 1), round(az, 1)))
    if len(rows) < 2 or np.linalg.cond(A) > 1e6: return None, len(cams)
    q = np.linalg.solve(A, b); res = math.sqrt(sum(s * float(np.linalg.norm((np.eye(2) - np.outer(d, d)) @ (q - c))) ** 2 for c, d, s in rows) / sum(s for _, _, s in rows))
    return res, len(cams)
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); imp = os.path.join(hdr, IMF); rp = os.path.join(hdr, RAW)
    if not (os.path.exists(imp) and os.path.exists(rp)): print(hn, "없음"); continue
    im = json.load(open(imp)); raw = json.load(open(rp)); byt = collections.defaultdict(list)
    for it in im: byt[it["type"]].append(it)
    out = []
    for t, its in byt.items():
        feats = []
        for it in its:
            pts = [(p[0], p[1], p[2], p[3], p[4]) for p in raw.get(t, []) if it.get("pos") and math.hypot(p[0] - it["pos"][0], p[1] - it["pos"][1]) <= R]
            res, nc = ray_residual(pts) if pts else (None, 0)
            feats.append((res, nc))
        vals = [(1.0 / (1.0 + r)) * min(1.0, nc / 3.0) if r is not None else None for r, nc in feats]; have = [v for v in vals if v is not None]; lo, hi = (min(have), max(have)) if have else (0, 1)
        for it, v, (r, nc) in zip(its, vals, feats):
            s_ = 0.5 if v is None else ((v - lo) / (hi - lo) if hi > lo else 0.5)
            o = dict(it); o["w0"] = it["w"]; o["ray_res"] = None if r is None else round(r, 3); o["ncam"] = nc; o["w"] = it["w"] * (0.2 + s_) ** ALPHA; out.append(o)
    json.dump(out, open(imp.replace(".json", "_rr_geo.json"), "w"), ensure_ascii=False)
    print("%s: 인스턴스 %d · 잔차 계산 %d" % (hn, len(out), sum(1 for o in out if o["ray_res"] is not None)), flush=True)
print("GEO_DONE")
