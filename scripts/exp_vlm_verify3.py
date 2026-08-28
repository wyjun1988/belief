#!/usr/bin/env python3
"""VLM 검증 4차 — **토큰 로그확률 점수화**. 운용점을 우리가 고른다. (H100)

    MODEL=Qwen/Qwen3.5-9B PAIRS=/tmp/vlmpair_hh OUT_JSONL=vlm4_scores.jsonl \\
      python scripts/exp_vlm_verify3.py

배경: 생성 텍스트 yes/no 는 운용점 고정이라 기각-우선 세팅이 불가. 실측
(0.67/0.885)을 T1 에 모의하면 0.455 로 통계(0.493)보다 못함. 로그확률 차
(logit_yes−logit_no · logit_A−logit_B)를 **연속 점수**로 받아 문턱을 로컬에서
스윕한다 — 기각 0.95+ 에서 수용이 얼마 남는지가 판정.
"""
import json, os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
PAIRS = os.environ.get("PAIRS", "/tmp/vlmpair_hh")
OUTJ = os.environ.get("OUT_JSONL", "vlm4_scores.jsonl")
pr = AutoProcessor.from_pretrained(MODEL)
md = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto").eval()
tok = pr.tokenizer

def first_id(s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0]

IDS = {k: first_id(k) for k in ("yes", "no", "Yes", "No", "A", "B", "C", " A", " B")}
print("모델:", MODEL, flush=True)


def logits_for(imgs, q):
    msgs = [{"role": "user", "content": [{"type": "image"}] * len(imgs)
             + [{"type": "text", "text": q}]}]
    try:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=imgs, text=text, return_tensors="pt").to(md.device)
    with torch.no_grad():
        lg = md(**inp).logits[0, -1]
    return lg


NORM = {"gpu rtx": "computer graphics card", "gpu 1080ti": "computer graphics card",
        "muscle gun": "massage gun", "ab roller": "exercise wheel",
        "spring clamp": "clamp", "push up board": "exercise board"}
items = [json.loads(l) for l in open(PAIRS + "/meta.jsonl")]
base = os.path.dirname(PAIRS + "/")
L = lambda p: Image.open(os.path.join(base, os.path.basename(p))).convert("RGB")
out = open(OUTJ, "w")
for n, m in enumerate(items):
    cand = L(m["cand"])
    a = NORM.get(m["label"], m["label"]).replace("_", " ")
    b = NORM.get(m["alt"], m["alt"]).replace("_", " ")
    lg = logits_for([cand], "Is there a %s in this image? Answer only yes or no." % a)
    s_yn = float(max(lg[IDS["yes"]], lg[IDS["Yes"]]) - max(lg[IDS["no"]], lg[IDS["No"]]))
    lg2 = logits_for([cand], "Which object is in this image: (A) %s or (B) %s? "
                     "Answer only A or B." % (a, b))
    s_ab = float(max(lg2[IDS["A"]], lg2[IDS[" A"]]) - max(lg2[IDS["B"]], lg2[IDS[" B"]]))
    lg3 = logits_for([cand], "Which is in this image: (A) %s, (B) %s, or (C) neither? "
                     "Answer only A, B, or C." % (a, b))
    s_ac = float(lg3[IDS["A"]] - max(lg3[IDS["B"]], lg3[IDS["C"]]))
    out.write(json.dumps(dict(cand=os.path.basename(m["cand"]), truth=m["truth"],
                              s_yn=round(s_yn, 3), s_ab=round(s_ab, 3),
                              s_ac=round(s_ac, 3),
                              **({"dist": m["dist"]} if "dist" in m else {}))) + "\n")
    if n % 50 == 0: print(n, flush=True)
out.close()
print("완료 →", OUTJ)
