#!/usr/bin/env python3
"""exp_vlm_verify3 의 mlx 네이티브판 (Apple Silicon) — 같은 입출력, 4B mxfp4.

    PAIRS=/tmp/hab_pairs OUT_JSONL=hab_scores.jsonl \\
      ~/mlx-venv/bin/python scripts/exp_vlm_verify3_mlx.py

로짓 모드 4B=9B 동급(§94) 전제. meta.jsonl(cand/label/alt/truth[/dist]) →
s_yn/s_ab/s_ac + dist 통과. rtx7_sweep 이 그대로 읽는다.
"""
import json, os
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
PAIRS = os.environ.get("PAIRS", "/tmp/pairs")
OUTJ = os.environ.get("OUT_JSONL", "vlm3_mlx_scores.jsonl")

model, processor = load(MODEL)
cfg = model.config
tok = processor.tokenizer
IDS = {t: tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B", "C", "Yes", "No")}


def last_logits(img_path, question):
    prompt = apply_chat_template(processor, cfg, question, num_images=1)
    inp = prepare_inputs(processor, images=[img_path], prompts=[prompt],
                         image_token_index=getattr(cfg, "image_token_index", None))
    out = model(inp["input_ids"], inp["pixel_values"],
                mask=inp.get("attention_mask"),
                **{k: v for k, v in inp.items()
                   if k not in ("input_ids", "pixel_values", "attention_mask")})
    lg = out.logits[0, -1]
    mx.eval(lg)
    return lg


rec = [json.loads(l) for l in open(os.path.join(PAIRS, "meta.jsonl"))]
out = open(OUTJ, "w")
for n, m in enumerate(rec):
    a, b = m["label"], m["alt"]
    lg = last_logits(m["cand"], "Is there a %s in this image? Answer only Yes or No." % a)
    s_yn = float(lg[IDS["Yes"]] - lg[IDS["No"]])
    lg = last_logits(m["cand"], "Which object is in this image: (A) %s or (B) %s? "
                     "Answer only A or B." % (a, b))
    s_ab = float(lg[IDS["A"]] - lg[IDS["B"]])
    lg = last_logits(m["cand"], "Which is in this image: (A) %s, (B) %s, or (C) neither? "
                     "Answer only A, B, or C." % (a, b))
    s_ac = float(lg[IDS["A"]] - max(float(lg[IDS["B"]]), float(lg[IDS["C"]])))
    out.write(json.dumps(dict(cand=os.path.basename(m["cand"]), truth=m["truth"],
                              s_yn=round(s_yn, 3), s_ab=round(s_ab, 3), s_ac=round(s_ac, 3),
                              **({"dist": m["dist"]} if "dist" in m else {}))) + "\n")
    if n % 50 == 0: print(n, flush=True)
out.close()
print("완료 →", OUTJ)
