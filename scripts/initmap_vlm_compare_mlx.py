#!/usr/bin/env python3
"""P0-B(2) 후보 비교 질의(mlx Qwen): 타입마다 후보 크롭(anchor_crops.py)을 한 프롬프트에 함께 주고 "이 집의 <type> 은 몇 번 사진인가"를 묻는다.
개별 yes/no(9-11)가 아니라 **택1 비교**. 번호 토큰 로짓의 softmax 를 외형 점수로 → w' = w × (0.2 + p).
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc.json ~/mlx-venv/bin/python scripts/initmap_vlm_compare_mlx.py → <house>/<initmap>_rr_vlm.json"""
import os, json, glob, collections, math
from PIL import Image
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); MAXC = int(os.environ.get("MAX_CANDS", "4"))
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
model, processor = load(MODEL); cfg = model.config; tok = processor.tokenizer
NUM = {n: tok.encode(str(n), add_special_tokens=False)[0] for n in range(1, MAXC + 1)}
def pick(imgs, typ):
    q = "These %d photos were taken in the same house. Exactly one shows the %s. Which photo number is the %s? Answer only the number." % (len(imgs), typ, typ)
    prompt = apply_chat_template(processor, cfg, q, num_images=len(imgs))
    inp = prepare_inputs(processor, images=imgs, prompts=[prompt], image_token_index=getattr(cfg, "image_token_index", None))
    out = model(inp["input_ids"], inp["pixel_values"], mask=inp.get("attention_mask"), **{k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask")})
    lg = out.logits[0, -1]; mx.eval(lg); v = [float(lg[NUM[n + 1]]) for n in range(len(imgs))]; m = max(v); e = [math.exp(x - m) for x in v]; s = sum(e)
    return [x / s for x in e]
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); imp = os.path.join(hdr, IMF); cp = os.path.join(hdr, "anchor_cands.json")
    if not (os.path.exists(imp) and os.path.exists(cp)): print(hn, "초기맵/크롭 없음"); continue
    im = json.load(open(imp)); cands = json.load(open(cp)); key = lambda it: (it["type"], tuple(round(v, 2) for v in (it.get("pos") or [])))
    crop_of = {key(c): os.path.join(hdr, "anchor_crops", "%03d.jpg" % c["i"]) for c in cands}
    byt = collections.defaultdict(list)
    for it in im: byt[it["type"]].append(it)
    out = []; n_q = 0
    for t, its in byt.items():
        its = sorted(its, key=lambda x: -x["w"]); have = [it for it in its if key(it) in crop_of][:MAXC]
        p = {}
        if len(have) >= 2:
            try:
                probs = pick([Image.open(crop_of[key(it)]).convert("RGB") for it in have], t); n_q += 1
                for it, pr in zip(have, probs): p[key(it)] = pr
            except Exception as e: print("  ", hn, t, "실패:", str(e)[:80])
        for it in its:
            o = dict(it); o["w0"] = it["w"]; o["vlm_p"] = round(p[key(it)], 3) if key(it) in p else None
            o["w"] = it["w"] * (0.2 + (p[key(it)] if key(it) in p else 0.5)); out.append(o)
    json.dump(out, open(imp.replace(".json", "_rr_vlm.json"), "w"), ensure_ascii=False)
    print("%s: 인스턴스 %d · 비교 질의 %d타입" % (hn, len(out), n_q), flush=True)
print("VLM_COMPARE_DONE")
