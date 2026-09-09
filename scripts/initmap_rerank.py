#!/usr/bin/env python3
"""P0-A 첫 실험: 초기맵 인스턴스 재순위 — 검출 통계(w) 에 **외형 점수**를 곱한다 (2026-09-10).
외형 점수 = 스캔 크롭(anchor_crops.py) 의 CLIP 텍스트-이미지 유사도("a photo of a <type>") 또는 VLM yes/no 로짓(anchors_vlm 형식 q1).
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json SCORE=clip|vlm python scripts/initmap_rerank.py → <house>/initmap_owl_rr_<score>.json
채점은 diag_initmap_pos.py(방 정답) 로. 9-11 의 '개별 yes/no 로 가지치기' 와 달리 후보를 버리지 않고 **순위만** 바꾼다."""
import os, json, glob, math, collections, numpy as np
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); SCORE = os.environ.get("SCORE", "clip")
ALPHA = float(os.environ.get("ALPHA", "1.0")); HOUSES = os.environ.get("HOUSES", "").split()
if SCORE == "clip":
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval(); cpp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    def clip_score(img_paths, types):
        with torch.no_grad():
            ie = cm.get_image_features(**cpp(images=[Image.open(p).convert("RGB") for p in img_paths], return_tensors="pt").to(DEV)); ie = ie / ie.norm(dim=-1, keepdim=True)
            te = cm.get_text_features(**cpp(text=["a photo of a %s" % t for t in types], return_tensors="pt", padding=True).to(DEV)); te = te / te.norm(dim=-1, keepdim=True)
        return (ie * te).sum(-1).float().cpu().numpy()
hds = [os.path.join(ROOT, h) for h in HOUSES] if HOUSES else sorted(glob.glob(os.path.join(ROOT, "house_*")))
for hd in hds:
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); imp = os.path.join(hdr, IMF); cp = os.path.join(hdr, "anchor_cands.json")
    if not (os.path.exists(imp) and os.path.exists(cp)): print(hn, "초기맵/크롭 없음"); continue
    im = json.load(open(imp)); cands = json.load(open(cp)); key = lambda it: (it["type"], tuple(round(v, 2) for v in (it.get("pos") or [])))
    sc = {}
    if SCORE == "clip":
        paths = [os.path.join(hdr, "anchor_crops", "%03d.jpg" % c["i"]) for c in cands]; types = [c["type"] for c in cands]
        vals = []
        for i in range(0, len(paths), 32): vals.extend(clip_score(paths[i:i+32], types[i:i+32]))
        for c, v in zip(cands, vals): sc[key(c)] = float(v)
    else:
        vf = os.path.join(hdr, "anchors_vlm.json")
        for a in (json.load(open(vf)) if os.path.exists(vf) else []): sc[key(a)] = a.get("q1") if a.get("q1") is not None else -5.0
    out = []
    # 타입별로 외형 점수를 0~1 로 정규화(같은 타입 후보 사이의 상대 순위만 의미) → w' = w × (eps + s)^ALPHA
    byt = collections.defaultdict(list)
    for it in im: byt[it["type"]].append(it)
    for t, its in byt.items():
        vals = [sc.get(key(it)) for it in its]; have = [v for v in vals if v is not None]
        lo, hi = (min(have), max(have)) if have else (0, 1)
        for it, v in zip(its, vals):
            s_ = 0.5 if v is None else ((v - lo) / (hi - lo) if hi > lo else 0.5)
            o = dict(it); o["w0"] = it["w"]; o["appear"] = round(s_, 3); o["w"] = it["w"] * (0.2 + s_) ** ALPHA; out.append(o)
    json.dump(out, open(os.path.join(hdr, IMF.replace(".json", "_rr_%s.json" % SCORE)), "w"), ensure_ascii=False)
    print("%s: 인스턴스 %d · 외형 점수 있음 %d" % (hn, len(out), sum(1 for o in out if o["appear"] != 0.5)), flush=True)
print("RERANK_DONE")
