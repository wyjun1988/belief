#!/usr/bin/env python3
"""기록 자리 판정 추론 — 초기맵 후보 인스턴스마다 "빨간 박스 안의 것이 정말 <타입>인가" 를 묻고 마진을 남긴다 (2026-09-15).

학습(lora_place_data.py · --task place)과 같은 프롬프트·같은 크롭 규칙.
  THOR_ROOT=data/hssd_v2 INITMAP_FILE=initmap_k8.json ADAPTER=~/khcache/place_ad/lora_place_4b \
  OUT_JSONL=~/khcache/place_margin.jsonl [HOUSES="house_0001 ..."] [VIEWS=1] python scripts/lora_place_infer.py
산출: {house, iid, type, room, margin} — 인스턴스당 뷰 최대 VIEWS 장의 **최대 마진**.
"""
import os, json, glob, time, collections
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2")
IMF = os.environ.get("INITMAP_FILE", "initmap_k8.json")
ADAPTER = os.path.expanduser(os.environ.get("ADAPTER", ""))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-4B"); DEV = os.environ.get("DEVICE", "mps")
OUT = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/place_margin.jsonl"))
VIEWS = int(os.environ.get("VIEWS", "1")); PAD = float(os.environ.get("PAD", "0.25"))
IMG_W = int(os.environ.get("IMG_W", "448"))
HOUSES = [h for h in os.environ.get("HOUSES", "").split() if h]
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
tk = getattr(proc, "tokenizer", proc)
YES, NO = tk.encode("yes", add_special_tokens=False)[0], tk.encode("no", add_special_tokens=False)[0]
def art(t): return ("an " if t[:1] in "aeiou" else "a ") + t
PF = '{"is_type": "'
def crop(src, box):
    im = Image.open(src).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    cx0 = max(0, x0 - bw * PAD); cy0 = max(0, y0 - bh * PAD)
    cx1 = min(W, x1 + bw * PAD); cy1 = min(H, y1 + bh * PAD)
    if cx1 - cx0 < 24 or cy1 - cy0 < 24: return None
    ImageDraw.Draw(im).rectangle([max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)],
                                 outline=(255, 0, 0), width=max(3, W // 150))
    im = im.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    w2, h2 = im.size; s = IMG_W / float(w2)
    return im.resize((IMG_W, max(8, int(h2 * s))))
def ask(im, typ):
    q = ("The image shows part of a room. Inside the red box, an object detector claims to have found %s. "
         "Ignore everything outside the box. Is the object inside the red box really %s? "
         'Answer with JSON only: {"is_type": "yes"|"no", "confidence": 0-100}' % (art(typ), art(typ)))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    try: t = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: t = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if "<think>" not in t[-60:]: t = t + ("" if t.endswith("\n") else "\n") + "<think>\n\n</think>\n\n"
    inp = proc(text=[t + PF], images=[im], return_tensors="pt").to(model.device)
    with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1); return float(lp[YES] - lp[NO])
hds = sorted(glob.glob(os.path.join(ROOT, "house_*")))
if HOUSES: hds = [h for h in hds if os.path.basename(h) in HOUSES]
n = 0; t0 = time.time(); st = collections.Counter()
with open(OUT, "w") as fo:
    for hd in hds:
        hn = os.path.basename(hd); rp = os.path.join(hd, "anchor_registry.json")
        if not os.path.exists(rp): st["등록부 없음"] += 1; continue
        for it in json.load(open(rp)):
            ms = []
            for v in (it.get("views") or [])[:VIEWS]:
                k, b = v.get("k"), v.get("box")
                if k is None or not b: continue
                im = crop(os.path.join(hd, "map", "%04d.jpg" % int(k)), b)
                if im is None: continue
                ms.append(ask(im, it["type"].replace("_", " ").lower()))
            if not ms: st["크롭 실패"] += 1; continue
            fo.write(json.dumps(dict(house=hn, iid=it["id"], type=it["type"], room=it.get("room"),
                                     pos=it.get("pos"), w=it.get("w"), margin=round(max(ms), 3), n_view=len(ms))) + "\n")
            n += 1
            if n % 200 == 0: print("  %d · %.0fs · %s" % (n, time.time() - t0, dict(st)), flush=True); fo.flush()
print("PLACE_INFER_DONE %d 인스턴스 · %.0fs · %s → %s" % (n, time.time() - t0, dict(st), OUT))
