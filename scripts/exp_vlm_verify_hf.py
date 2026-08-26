#!/usr/bin/env python3
"""큰 VLM 검증 — H100 용 (transformers). 두 크롭 세트를 같은 스펙으로 채점한다.

    MODEL=Qwen/Qwen3.5-VL-9B-Instruct CROPS=/path/vlmreal python scripts/exp_vlm_verify_hf.py

  세트 A  CROPS 디렉터리 (meta.jsonl — IT3DEgo 실사 160장, 손으로 옮겨온 것)
  세트 B  thor4 프레임에서 즉석 추출 (GT ctr 로 양성 / 다른 물체 크롭 음성)

목표 스펙: 진짜 수용 ≥0.85 / 가짜 기각 ≥0.85 (§83). 4B(mlx)는 실사·시뮬 모두
0.60/0.72 로 미달 — 모델 크기가 원인인지 이 실행이 가른다.
게이트 저장소면 HF_TOKEN 환경변수 필요.
"""
import glob, json, os, sys
import numpy as np
from PIL import Image
import torch

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-VL-9B-Instruct")
CROPS = os.environ.get("CROPS", "")
ROOT = os.environ.get("THOR_ROOT", "data/thor4")
N = int(os.environ.get("N", "100"))
from transformers import AutoProcessor, AutoModelForImageTextToText
pr = AutoProcessor.from_pretrained(MODEL)
md = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
print("모델:", MODEL, flush=True)


def ask(img, label):
    q = "Is there a %s in this image? Answer only yes or no." % label.replace("_", " ")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=[img], text=text, return_tensors="pt").to(md.device)
    with torch.no_grad():
        out = md.generate(**inp, max_new_tokens=6, do_sample=False)
    ans = pr.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return ans.strip().lower().startswith("yes")


def score(items, tag):
    tp = np_ = fp = nn = 0
    for img, label, truth in items:
        y = ask(img, label)
        if truth: np_ += 1; tp += y
        else: nn += 1; fp += y
    print("[%s] 진짜 yes **%.3f** (%d/%d) · 가짜 no **%.3f** (%d/%d)"
          % (tag, tp/max(np_,1), tp, np_, 1-fp/max(nn,1), nn-fp, nn), flush=True)


if CROPS and os.path.exists(CROPS + "/meta.jsonl"):
    items = []
    for line in open(CROPS + "/meta.jsonl"):
        m = json.loads(line)
        f = os.path.join(CROPS, os.path.relpath(m["f"], "/tmp/vlmreal"))
        if os.path.exists(f):
            items.append((Image.open(f).convert("RGB"), m["label"], m["truth"]))
    score(items, "실사 IT3DEgo %d장" % len(items))

# 세트 B — thor4 시뮬 크롭
def words(t): return "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
rng = np.random.default_rng(0); items = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    if len(items) >= 2 * N: break
    g = json.load(open(os.path.join(hd, "gt.json")))
    lv = {int(os.path.basename(p)[:-4]): p
          for p in glob.glob(os.path.join(hd, "live", "*.jpg"))}
    if not lv: continue
    typ = {o: v["type"] for o, v in g["gt0"].items()}
    ms = [m for m in g["live"] if m["t"] in lv and m.get("ctr")]
    rng.shuffle(ms)
    for m in ms[:30]:
        oids = [o for o, c in m["ctr"].items() if c and o in typ]
        if not oids or len(items) >= 2 * N: continue
        o = oids[int(rng.integers(len(oids)))]
        cx, cy = m["ctr"][o]
        im = Image.open(lv[m["t"]]).convert("RGB")
        c = im.crop((max(0, cx-64), max(0, cy-64), min(384, cx+64), min(384, cy+64)))
        c = c.resize((336, 336), Image.LANCZOS)
        items.append((c, words(typ[o]), 1))
        other = rng.choice([t for t in set(typ.values()) if t != typ[o]])
        items.append((c, words(other), 0))
score(items, "시뮬 thor4 %d장" % len(items))
