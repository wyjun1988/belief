#!/usr/bin/env python3
"""고정 앵커 **인스턴스 등록부**(스캔 단계, 무GT; 사용자 결정 2026-09-08 17:45 — "같은 타입이어도 인스턴스로 구분되게 등록").
초기맵의 가구 인스턴스마다 raw 투영점의 상위 프레임 K 장에서 OWL 박스를 다시 구해 크롭 → CLIP(ViT-B/16) 임베딩을 저장한다.
라이브에서 같은 타입 검출 크롭을 이 임베딩과 대조해 **어느 인스턴스인지** 고른다(anchor_ctx.py).
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json [HOUSES="house_0014"] python scripts/anchor_registry.py
산출: <house>/anchor_registry.json = [{id, type, pos, room, w, views:[{k, box, owl}]}] + anchor_registry.npz (emb: N×512, ids, view_of)"""
import os, json, glob, math, time, numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, CLIPModel, CLIPProcessor
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); HOUSES = os.environ.get("HOUSES", "").split()
W_MIN = float(os.environ.get("W_MIN", "1.0")); S_MIN = float(os.environ.get("S_MIN", "0.15")); KV = int(os.environ.get("K_VIEWS", "3")); REDO = os.environ.get("REDO", "0") == "1"
ANCH = [t.strip() for t in os.environ.get("ANCH_TYPES", "bed,couch,table,cabinet,shelves,chest of drawers,wardrobe,counter,tv,toilet,fridge,shower,bathtub,washer dryer,bench,floor lamp,stand,sink,fireplace,dishwasher").split(",")]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval(); cpp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
def owl_box(im, typ):
    inp = op(text=[["a photo of a %s" % typ]], images=[im], return_tensors="pt").to(DEV)
    with torch.no_grad(): out = on(**inp)
    W, H = im.size; res = op.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[H, W]]))[0]
    if len(res["scores"]) == 0: return None, 0.0
    j = int(res["scores"].argmax()); return [float(v) for v in res["boxes"][j]], float(res["scores"][j])
def crop(im, b, m=0.15):
    x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0; W, H = im.size
    return im.crop((int(max(0, x0 - m * w)), int(max(0, y0 - m * h)), int(min(W, max(x1 + m * w, x0 + 8))), int(min(H, max(y1 + m * h, y0 + 8)))))
def clip_emb(ims):
    with torch.no_grad(): e = cm.get_image_features(**cpp(images=ims, return_tensors="pt").to(DEV))
    e = e / e.norm(dim=-1, keepdim=True); return e.float().cpu().numpy()
hds = [os.path.join(ROOT, h) for h in HOUSES] if HOUSES else sorted(glob.glob(os.path.join(ROOT, "house_*")))
for hd in hds:
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); outp = os.path.join(hdr, "anchor_registry.json")
    if os.path.exists(outp) and not REDO: print("%s 있음 — 건너뜀" % hn, flush=True); continue
    imp, rawp = os.path.join(hdr, IMF), os.path.join(hdr, "initmap_raw.json")
    if not (os.path.exists(imp) and os.path.exists(rawp)): print("%s 초기맵/raw 없음" % hn, flush=True); continue
    im_ = json.load(open(imp)); raw = json.load(open(rawp)); mfs = sorted(glob.glob(os.path.join(hdr, "map", "*.jpg"))); cdir = os.path.join(hdr, "anchor_reg_crops"); os.makedirs(cdir, exist_ok=True)
    T0 = time.time(); reg = []; embs = []; ids = []; view_of = []; ntype = {}
    for it in sorted(im_, key=lambda x: -x["w"]):
        if it["type"] not in ANCH or not it.get("pos") or it["w"] < W_MIN: continue
        pts = sorted([p for p in raw.get(it["type"], []) if math.hypot(p[0] - it["pos"][0], p[1] - it["pos"][1]) <= 1.0], key=lambda p: -p[2])
        ks = []
        for p in pts:
            if p[2] < S_MIN: break
            if int(p[5]) not in ks and int(p[5]) < len(mfs): ks.append(int(p[5]))
            if len(ks) >= KV: break
        if not ks: continue
        aid = "%s#%d" % (it["type"], ntype.get(it["type"], 0)); ntype[it["type"]] = ntype.get(it["type"], 0) + 1
        views = []; crops = []
        for k in ks:
            im = Image.open(mfs[k]).convert("RGB"); b, sc = owl_box(im, it["type"])
            if b is None or sc < 0.08: continue
            c = crop(im, b); c.save(os.path.join(cdir, "%s_%d.jpg" % (aid.replace(" ", "_").replace("#", "-"), k)), quality=88); crops.append(c)
            views.append(dict(k=k, box=[round(v, 1) for v in b], owl=round(sc, 3)))
        if not crops: continue
        e = clip_emb(crops)
        for j in range(len(crops)): embs.append(e[j]); ids.append(aid); view_of.append(len(reg))
        reg.append(dict(id=aid, type=it["type"], pos=it["pos"], room=it.get("room"), w=it["w"], views=views))
    json.dump(reg, open(outp, "w"), ensure_ascii=False)
    np.savez(os.path.join(hdr, "anchor_registry.npz"), emb=np.array(embs, np.float32) if embs else np.zeros((0, 512), np.float32), ids=np.array(ids), view_of=np.array(view_of))
    print("%s: 앵커 인스턴스 %d · 뷰 %d (%.0fs)" % (hn, len(reg), len(embs), time.time() - T0), flush=True)
print("ANCHOR_REGISTRY_DONE")
