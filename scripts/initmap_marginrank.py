#!/usr/bin/env python3
"""P0-D 타입 마진 재순위: raw 투영점의 (패치 최상위 타입, 마진 = 해당 타입 점수 − 최상위 경쟁 타입 점수) 를 군집별로 모아
'이 후보는 그 타입이 아니라 다른 타입일 가능성' 을 잰다. s = 군집 점들의 마진 가중 평균(경쟁 타입이 이기면 음수) → w' = w × (0.2 + s_norm)^ALPHA.
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json RAW=initmap_raw_m.json python scripts/initmap_marginrank.py → <initmap>_rr_margin.json"""
import os, json, glob, math, collections
ROOT = os.environ.get("THOR_ROOT"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); RAW = os.environ.get("RAW", "initmap_raw_m.json"); ALPHA = float(os.environ.get("ALPHA", "1.0")); R = float(os.environ.get("R", "2.0"))
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); imp = os.path.join(hdr, IMF); rp = os.path.join(hdr, RAW)
    if not (os.path.exists(imp) and os.path.exists(rp)): print(hn, "없음"); continue
    im = json.load(open(imp)); raw = json.load(open(rp)); byt = collections.defaultdict(list)
    for it in im: byt[it["type"]].append(it)
    out = []; n_ok = 0
    for t, its in byt.items():
        vals = []
        for it in its:
            pts = [p for p in raw.get(t, []) if it.get("pos") and len(p) >= 8 and math.hypot(p[0] - it["pos"][0], p[1] - it["pos"][1]) <= R]
            if pts: vals.append(sum(p[2] * p[7] for p in pts) / sum(p[2] for p in pts))
            else: vals.append(None)
        have = [v for v in vals if v is not None]; lo, hi = (min(have), max(have)) if have else (0, 1)
        for it, v in zip(its, vals):
            s_ = 0.5 if v is None else ((v - lo) / (hi - lo) if hi > lo else 0.5)
            o = dict(it); o["w0"] = it["w"]; o["margin"] = None if v is None else round(v, 4); o["w"] = it["w"] * (0.2 + s_) ** ALPHA; out.append(o); n_ok += v is not None
    json.dump(out, open(imp.replace(".json", "_rr_margin.json"), "w"), ensure_ascii=False); print("%s: 인스턴스 %d · 마진 있음 %d" % (hn, len(out), n_ok), flush=True)
print("MARGIN_DONE")
