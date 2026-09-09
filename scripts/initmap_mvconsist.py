#!/usr/bin/env python3
"""P0-B(1) 다시점 일관성 재순위: anchor_registry.py 가 저장한 인스턴스별 스캔 뷰 크롭 CLIP 임베딩(≤3뷰)의 **뷰 간 평균 코사인**을 외형 점수로 쓴다.
같은 물체를 여러 각도에서 본 군집은 뷰끼리 닮고, 오검출이 모인 군집은 안 닮는다는 가정. w' = w × (0.2 + s)^ALPHA, s = 타입 내 0~1 정규화.
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json python scripts/initmap_mvconsist.py → <house>/<initmap>_rr_mv.json"""
import os, json, glob, collections, numpy as np
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); ALPHA = float(os.environ.get("ALPHA", "1.0"))
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); rp = os.path.join(hdr, "anchor_registry.json"); imp = os.path.join(hdr, IMF)
    if not (os.path.exists(rp) and os.path.exists(imp)): print(hn, "등록부/초기맵 없음"); continue
    reg = json.load(open(rp)); rz = np.load(os.path.join(hdr, "anchor_registry.npz")); E, IDS, VO = rz["emb"], list(rz["ids"]), rz["view_of"]
    im = json.load(open(imp)); key = lambda it: (it["type"], tuple(round(v, 2) for v in (it.get("pos") or [])))
    cons = {}; nview = {}
    for k, r in enumerate(reg):
        rows = [i for i, v in enumerate(VO) if int(v) == k]
        if len(rows) >= 2:
            M = E[rows] @ E[rows].T; n = len(rows); cons[key(r)] = float((M.sum() - n) / (n * (n - 1)))
        else: cons[key(r)] = None
        nview[key(r)] = len(rows)
    byt = collections.defaultdict(list)
    for it in im: byt[it["type"]].append(it)
    out = []
    for t, its in byt.items():
        vals = [cons.get(key(it)) for it in its]; have = [v for v in vals if v is not None]; lo, hi = (min(have), max(have)) if have else (0, 1)
        for it, v in zip(its, vals):
            s_ = 0.5 if v is None else ((v - lo) / (hi - lo) if hi > lo else 0.5)
            o = dict(it); o["w0"] = it["w"]; o["mv"] = None if v is None else round(v, 3); o["nview"] = nview.get(key(it), 0); o["w"] = it["w"] * (0.2 + s_) ** ALPHA; out.append(o)
    json.dump(out, open(imp.replace(".json", "_rr_mv.json"), "w"), ensure_ascii=False)
    print("%s: 인스턴스 %d · 뷰 ≥2 %d" % (hn, len(out), sum(1 for o in out if o["mv"] is not None)), flush=True)
print("MV_DONE")
