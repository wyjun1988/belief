#!/usr/bin/env python3
"""② 채택 판정 추론 — 검증기 통과 프레임마다 "이 빨간 박스가 기록 물체와 같은 것인가" 를 묻고, 통과분만 남긴 점수 파일을 쓴다 (2026-09-14).

학습(lora_adopt_data.py · lora_presence_train.py --task adopt)과 **같은 프롬프트·같은 박스 규칙**을 쓴다.
  BACKEND=hf  MODEL=Qwen/Qwen3.5-4B ADAPTER=~/khcache/adopt4b/lora_adopt_qwen3_5-4b \\
  VERIFY_JSONL=~/khcache/bench-v2full/scores/t1_all.jsonl A3_PREFIX=~/khcache/bench-v2full/cache/hs2_a_ \\
  THOR_ROOT=data/hssd_v2 OUT_JSONL=~/khcache/bench-v2full/scores/t1_adopt_real.jsonl python scripts/lora_adopt_infer.py
옵션: DEVICE=mps|cuda · LIMIT=0(전체) · VERDICT_JSONL=<판정 원장> · KEEP_UNJUDGED=1(판정 실패 프레임은 남긴다)
"""
import os, json, re, time, glob, collections, numpy as np
from PIL import Image, ImageDraw
ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2")
VJ = os.path.expanduser(os.environ["VERIFY_JSONL"]); A3P = os.path.expanduser(os.environ["A3_PREFIX"])
OUT = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/t1_adopt_real.jsonl"))
VERD = os.path.expanduser(os.environ.get("VERDICT_JSONL", OUT.replace(".jsonl", "_verdict.jsonl")))
VTH, VTH2 = float(os.environ.get("VERIFY_TH", "2.069")), float(os.environ.get("VERIFY_TH2", "0.887"))
IMG_W = int(os.environ.get("IMG_W", "448")); LIMIT = int(os.environ.get("LIMIT", "0"))
KEEP_UNJ = os.environ.get("KEEP_UNJUDGED", "1") == "1"
BACKEND = os.environ.get("BACKEND", "hf"); MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-4B")
ADAPTER = os.path.expanduser(os.environ.get("ADAPTER", "")); DEV = os.environ.get("DEVICE", "mps")
def article(t): return ("an " if t[:1] in "aeiou" else "a ") + t
def prompt_of(typ):
    return ("Image 1 shows %s at its recorded place in a house (inside the red box). Image 2 is a later view from elsewhere in the same house, "
            "where an object detector proposed %s (inside the red box). The object may have been moved, so a different room is possible. "
            "Judge the object itself, not the room: is the thing in the red box in image 2 the same %s as in image 1? "
            "Answer with JSON only: {\"same_object\": \"yes\"|\"no\", \"confidence\": 0-100}" % (article(typ), article(typ), typ))
PREFILL = '{"same_object": "'
if BACKEND == "hf":
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    if ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER); model.eval()
    def ask(ims, typ):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt_of(typ)}]}]
        try: t = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError: t = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if "<think>" not in t[-60:]: t = t + ("" if t.endswith("\n") else "\n") + "<think>\n\n</think>\n\n"
        inp = proc(text=[t + PREFILL], images=ims, return_tensors="pt").to(model.device)
        with torch.no_grad(): o = model.generate(**inp, max_new_tokens=24, do_sample=False)
        return PREFILL + proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
else:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template as act
    m, proc = load(MODEL); cfg = m.config
    def ask(ims, typ):
        p = act(proc, cfg, prompt_of(typ), num_images=2)
        return PREFILL + generate(m, proc, p + PREFILL, ims, max_tokens=24, verbose=False)
def parse(txt):
    m_ = re.search(r"\{.*?\}", txt, re.S)
    if m_:
        try: return str(json.loads(m_.group(0)).get("same_object", "")).lower()
        except Exception: pass
    m_ = re.search(r'"same_object"\s*:\s*"(yes|no)"', txt)
    if m_: return m_.group(1)
    m_ = re.search(r"\b(yes|no)\b", txt.lower())
    return m_.group(1) if m_ else ""
def boxed(path, box, w_out=IMG_W):
    im = Image.open(path).convert("RGB"); W, H = im.size
    if box:
        x0, y0, x1, y1 = [max(0, min(W - 1 if i % 2 == 0 else H - 1, int(v))) for i, v in enumerate(box)]
        if x1 > x0 and y1 > y0: ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=max(3, W // 150))
    s = w_out / float(W); return im.resize((w_out, max(8, int(H * s))))
G = {}; Z = {}; REF = {}
st = collections.Counter(); t0 = time.time(); n = 0
fo = open(OUT, "w"); fv = open(VERD, "w")
for ln in open(VJ):
    d = json.loads(ln); h, oid = d["house"], d["oid"]; rows = d.get("scored") or []
    if h not in G:
        try:
            G[h] = json.load(open(f"{ROOT}/{h}/gt.json")); Z[h] = np.load(A3P + h + ".npz", allow_pickle=True)
        except Exception: G[h] = None
    if G[h] is None: fo.write(ln); continue
    g = G[h]; z = Z[h]; ts = z["ts"]; vocab = [str(x) for x in z["vocab"]]; typ = oid.split("|")[0]
    if typ not in vocab: fo.write(ln); continue
    ti = vocab.index(typ)
    if (h, oid) not in REF:
        c0 = [(m["dist"][oid], k) for k, m in enumerate(g["map"]) if oid in (m.get("ctr") or {}) and oid in (m.get("dist") or {})]
        if not c0: REF[(h, oid)] = None
        else:
            _, k0 = min(c0); mk = g["map"][k0]; bx = (mk.get("box") or {}).get(oid); ctr = mk["ctr"][oid]
            rb = bx if bx else [ctr[0] - 60, ctr[1] - 60, ctr[0] + 60, ctr[1] + 60]
            p = f"{ROOT}/{h}/map/%04d.jpg" % k0
            REF[(h, oid)] = boxed(p, rb) if os.path.exists(p) else None
    ref = REF[(h, oid)]
    if ref is None: fo.write(ln); continue
    keep = []
    for e in rows:
        if not (e[1] >= VTH and (len(e) < 3 or e[2] >= VTH2)): keep.append(e); continue
        i = int(e[0])
        if i >= len(ts): keep.append(e); continue
        t = int(ts[i]); p = f"{ROOT}/{h}/live/%06d.jpg" % t
        if not os.path.exists(p): keep.append(e); continue
        W = float(Image.open(p).size[0]); bcx, bcy, bw, bh = [float(x) * W for x in z["bx"][i, ti]]
        box = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2] if (bw > 1 and bh > 1) else None
        try: raw = ask([ref, boxed(p, box)], typ.replace("_", " ").lower())
        except Exception as ex:
            st["오류"] += 1; fv.write(json.dumps(dict(house=h, oid=oid, t=t, ans="", err=str(ex)[:120])) + "\n")
            if KEEP_UNJ: keep.append(e)
            continue
        a = parse(raw); st[a or "무응답"] += 1; n += 1
        fv.write(json.dumps(dict(house=h, oid=oid, t=t, i=i, ans=a, raw=raw[:120])) + "\n")
        if a == "yes" or (a == "" and KEEP_UNJ): keep.append(e)
        if n % 200 == 0:
            print("  %d장 · %.0fs · %s" % (n, time.time() - t0, dict(st)), flush=True); fv.flush()
        if LIMIT and n >= LIMIT: break
    d["scored"] = keep; fo.write(json.dumps(d) + "\n")
    if LIMIT and n >= LIMIT: break
fo.close(); fv.close()
print("ADOPT_INFER_DONE %d장 · %.0fs · %s → %s" % (n, time.time() - t0, dict(st), OUT))
