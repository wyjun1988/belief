#!/usr/bin/env python3
"""VLM 검증 2차 — 라벨 의존을 줄인 세 형식 비교. (H100)

    MODEL=... PAIRS=/tmp/vlmpair python scripts/exp_vlm_verify2.py

  pair   등록 크롭 + 후보 크롭 두 장: "같은 물건인가?"  ← 라벨 불필요 (본명 실험)
  2afc   "이건 A 인가 B 인가?" (혼동 라벨과 강제 선택)   ← yes 편향 제거
  norm   yes/no 인데 라벨을 일상어로 정규화

1차 결과: yes/no 단문형은 4B 0.61/0.72 · 9B 0.68/0.70 — 크기가 답이 아니었다.
⚠️ thinking 모델이면 enable_thinking=False (1차에서 물린 함정 — 켜져 있으면
추론 문장이 먼저 나와 파서가 전부 no 처리한다).
"""
import json, os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
PAIRS = os.environ.get("PAIRS", "/tmp/vlmpair")
pr = AutoProcessor.from_pretrained(MODEL)
md = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto").eval()
print("모델:", MODEL, flush=True)

NORM = {"gpu rtx": "computer graphics card", "gpu 1080ti": "computer graphics card",
        "muscle gun": "massage gun", "ab roller": "exercise wheel",
        "spring clamp": "clamp", "push up board": "exercise board"}


def gen(imgs, q):
    msgs = [{"role": "user", "content": [{"type": "image"}] * len(imgs)
             + [{"type": "text", "text": q}]}]
    kw = {}
    try:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=imgs, text=text, return_tensors="pt").to(md.device)
    with torch.no_grad():
        out = md.generate(**inp, max_new_tokens=8, do_sample=False)
    return pr.decode(out[0][inp["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip().lower()


items = [json.loads(l) for l in open(PAIRS + "/meta.jsonl")]
base = os.path.dirname(PAIRS + "/")
L = lambda p: Image.open(os.path.join(base, os.path.basename(p))).convert("RGB")

OUTJ = os.environ.get("OUT_JSONL", "")
logf = open(OUTJ, "w") if OUTJ else None
for mode in ("2afc", "norm", "3afc"):
    tp = np_ = fp = nn = 0
    for m in items:
        cand = L(m["cand"])
        if mode == "pair":
            q = ("The first image shows a specific object. Does the second image "
                 "show the SAME object? Answer only yes or no.")
            y = gen([L(m["enroll"]), cand], q).startswith("yes")
        elif mode == "2afc":
            a = NORM.get(m["label"], m["label"]).replace("_", " ")
            b = NORM.get(m["alt"], m["alt"]).replace("_", " ")
            ans = gen([cand], "Which object is in this image: (A) %s or (B) %s? "
                      "Answer only A or B." % (a, b))
            y = ans.startswith("a") or a.split()[0] in ans
        elif mode == "3afc":
            # (C) 둘 다 아님 — 2AFC 의 기각력을 유지하며 오검출(제3의 물체·배경)을
            # C 로 흘려보낸다. 수용 = A 선택.
            a = NORM.get(m["label"], m["label"]).replace("_", " ")
            b = NORM.get(m["alt"], m["alt"]).replace("_", " ")
            ans = gen([cand], "Which is in this image: (A) %s, (B) %s, or "
                      "(C) neither? Answer only A, B, or C." % (a, b))
            y = ans.startswith("a")
        else:
            a = NORM.get(m["label"], m["label"]).replace("_", " ")
            y = gen([cand], "Is there a %s in this image? Answer only yes or no."
                    % a).startswith("yes")
        if logf:
            logf.write(json.dumps(dict(mode=mode, cand=os.path.basename(m["cand"]),
                                       truth=m["truth"], y=int(y))) + "\n")
        if m["truth"]: np_ += 1; tp += y
        else: nn += 1; fp += y
    print("[%s] 진짜 수용 **%.3f** (%d/%d) · 가짜 기각 **%.3f** (%d/%d)"
          % (mode, tp/max(np_,1), tp, np_, 1-fp/max(nn,1), nn-fp, nn), flush=True)
if logf: logf.close()
