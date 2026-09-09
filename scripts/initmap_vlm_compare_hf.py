#!/usr/bin/env python3
"""P0-B(2) 후보 비교 질의 — HF(CUDA) 판. initmap_vlm_compare_mlx.py 와 같은 프롬프트·출력, 모델만 HF Qwen (RTX 9-15).
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json MODEL=Qwen/Qwen3.5-9B python scripts/initmap_vlm_compare_hf.py → <house>/<initmap>_rr_vlm.json
전제: anchor_crops.py 로 <house>/anchor_cands.json + anchor_crops/ 가 있어야 한다(MAX_INST=400 W_MIN=0.3 S_MIN=0.12)."""
import os, json, glob, collections, math, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
ROOT = os.environ.get("THOR_ROOT", "data/og12"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); MAXC = int(os.environ.get("MAX_CANDS", "4")); MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
pr = AutoProcessor.from_pretrained(MODEL); md = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval(); tok = pr.tokenizer
NUM = {n: tok.encode(str(n), add_special_tokens=False)[0] for n in range(1, MAXC + 1)}
def pick(imgs, typ):
    q = "These %d photos were taken in the same house. Exactly one shows the %s. Which photo number is the %s? Answer only the number." % (len(imgs), typ, typ)
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": q}]}]
    try: text = pr.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError: text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=imgs, text=text, return_tensors="pt").to(md.device)
    with torch.no_grad(): lg = md(**inp).logits[0, -1]
    v = [float(lg[NUM[n + 1]]) for n in range(len(imgs))]; m = max(v); e = [math.exp(x - m) for x in v]; s = sum(e); return [x / s for x in e]
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); imp = os.path.join(hdr, IMF); cp = os.path.join(hdr, "anchor_cands.json")
    if not (os.path.exists(imp) and os.path.exists(cp)): print(hn, "초기맵/크롭 없음"); continue
    im = json.load(open(imp)); cands = json.load(open(cp)); key = lambda it: (it["type"], tuple(round(v, 2) for v in (it.get("pos") or [])))
    crop_of = {key(c): os.path.join(hdr, "anchor_crops", "%03d.jpg" % c["i"]) for c in cands}
    byt = collections.defaultdict(list)
    for it in im: byt[it["type"]].append(it)
    out = []; n_q = 0
    for t, its in byt.items():
        its = sorted(its, key=lambda x: -x["w"]); have = [it for it in its if key(it) in crop_of][:MAXC]; p = {}
        if len(have) >= 2:
            try:
                probs = pick([Image.open(crop_of[key(it)]).convert("RGB") for it in have], t); n_q += 1
                for it, pr_ in zip(have, probs): p[key(it)] = pr_
            except Exception as e: print("  ", hn, t, "실패:", str(e)[:80])
        for it in its:
            o = dict(it); o["w0"] = it["w"]; o["vlm_p"] = round(p[key(it)], 3) if key(it) in p else None; o["w"] = it["w"] * (0.2 + (p[key(it)] if key(it) in p else 0.5)); out.append(o)
    json.dump(out, open(imp.replace(".json", "_rr_vlm.json"), "w"), ensure_ascii=False); print("%s: 인스턴스 %d · 비교 질의 %d타입" % (hn, len(out), n_q), flush=True)
print("VLM_COMPARE_DONE")
