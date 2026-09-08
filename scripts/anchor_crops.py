#!/usr/bin/env python3
"""고정 앵커 명부 1단계(kx-venv): 초기맵 인스턴스마다 raw 투영점 중 최고 점수 프레임에서 OWL 박스를 다시 구해 크롭을 저장한다.
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json HOUSES="house_0014" python scripts/anchor_crops.py
산출: <house>/anchor_crops/NNN.jpg + <house>/anchor_cands.json = [{i, type, pos, room, w, k, box, owl}]. 2단계 anchor_vlm_mlx.py (mlx-venv)."""
import os, json, glob, math, time, numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json")
HOUSES = os.environ.get("HOUSES", "").split(); W_MIN = float(os.environ.get("W_MIN", "1.0")); S_MIN = float(os.environ.get("S_MIN", "0.2")); MAXI = int(os.environ.get("MAX_INST", "150")); REDO = os.environ.get("REDO", "0") == "1"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
def owl_box(im, typ):
    inp = op(text=[["a photo of a %s" % typ]], images=[im], return_tensors="pt").to(DEV)
    with torch.no_grad(): out = on(**inp)
    W, H = im.size; res = op.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[H, W]]))[0]
    if len(res["scores"]) == 0: return None, 0.0
    j = int(res["scores"].argmax()); return [float(v) for v in res["boxes"][j]], float(res["scores"][j])
def crop(im, b, m=0.25):
    x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0; W, H = im.size
    x0, y0, x1, y1 = max(0, x0 - m * w), max(0, y0 - m * h), min(W, x1 + m * w), min(H, y1 + m * h)
    c = im.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8)))); c.thumbnail((448, 448)); return c
hds = [os.path.join(ROOT, h) for h in HOUSES] if HOUSES else sorted(glob.glob(os.path.join(ROOT, "house_*")))
for hd in hds:
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); outp = os.path.join(hdr, "anchor_cands.json"); cdir = os.path.join(hdr, "anchor_crops")
    if os.path.exists(outp) and not REDO: print("%s 있음 — 건너뜀" % hn, flush=True); continue
    imp, rawp = os.path.join(hdr, IMF), os.path.join(hdr, "initmap_raw.json")
    if not (os.path.exists(imp) and os.path.exists(rawp)): print("%s 초기맵/raw 없음" % hn, flush=True); continue
    im_ = json.load(open(imp)); raw = json.load(open(rawp)); mfs = sorted(glob.glob(os.path.join(hdr, "map", "*.jpg"))); os.makedirs(cdir, exist_ok=True)
    cands = []
    for it in im_:
        if not it.get("pos") or it["w"] < W_MIN: continue
        pts = [p for p in raw.get(it["type"], []) if math.hypot(p[0] - it["pos"][0], p[1] - it["pos"][1]) <= 1.0]
        if not pts: continue
        best = max(pts, key=lambda p: p[2])
        if best[2] >= S_MIN: cands.append((it, best))
    cands.sort(key=lambda c: -c[0]["w"]); cands = cands[:MAXI]; T0 = time.time(); out = []
    for it, best in cands:
        k = int(best[5])
        if k >= len(mfs): continue
        im = Image.open(mfs[k]).convert("RGB"); b, sc = owl_box(im, it["type"])
        if b is None: continue
        i = len(out); crop(im, b).save(os.path.join(cdir, "%03d.jpg" % i), quality=90)
        out.append(dict(i=i, type=it["type"], pos=it["pos"], room=it["room"], w=it["w"], k=k, box=[round(v, 1) for v in b], owl=round(sc, 3)))
    json.dump(out, open(outp, "w"), ensure_ascii=False); print("%s: 후보 %d 크롭 (%.0fs)" % (hn, len(out), time.time() - T0), flush=True)
print("ANCHOR_CROPS_DONE")
