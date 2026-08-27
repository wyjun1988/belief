#!/usr/bin/env python3
"""EgoLife where-질의 v2 — 사건 군집 + '직전 사건' + 광각 프레임. (M1 Max · 4B mlx)

    python scripts/egolife_qa_v2.py

v1(0.30) 진단 반영:
  ① "Where was X before?" 는 최신 목격이 아니라 **직전 안정 구간**을 묻는다
     → 문턱 통과 목격을 시간 군집(사건)으로 묶고, 마지막 사건이 '사용 중'이면
       그 **직전 사건**의 프레임을 증거로 쓴다
  ② 타이트 크롭이 방·가구 맥락을 제거 → **광각 원본 프레임**을 그대로 준다
     (선택지가 "거실 테이블 위" 같은 장소 표현이므로)
"""
import glob, json, os, re
import numpy as np
from PIL import Image

QAF = os.path.expanduser("~/khronos/EgoLifeQA_A1_JAKE.json")
FRAMES = "/tmp/egl_frames"
CACHE = "/tmp/egl_owl.npz"
OUT = "/tmp/egl_answers_v2.jsonl"

z = np.load(CACHE, allow_pickle=True)
S, P, ph, pw = z["s"], z["p"], int(z["ph"]), int(z["pw"])
secs, vocab = z["secs"], list(z["vocab"])
files = {int(s): os.path.join(FRAMES, "%05d.jpg" % s) for s in secs}

qa = json.load(open(QAF))
qs = []
for q in qa:
    if q["type"] != "EntityLog" or not q["question"].lower().startswith("where"):
        continue
    obj = re.sub(r"^where (was|were|is|are) (the )?", "", q["question"].lower())
    obj = obj.split(" before")[0].split(" placed")[0].split(" in ")[0].split("?")[0].strip()
    tm = q["query_time"]["time"]
    qs.append(dict(id=q["ID"], obj=obj, q=q["question"],
                   tq=int(tm[:2])*3600+int(tm[2:4])*60+int(tm[4:6]),
                   choices=[q["choice_%s" % c] for c in "abcd"], answer=q["answer"]))

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
model, processor = load("RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
cfg = model.config

def ask_choice(imgs, question, choices):
    qtext = ("These images show moments related to the question. %s "
             "(A) %s (B) %s (C) %s (D) %s. Answer with only one letter A, B, C, or D."
             % (question, *choices))
    prompt = apply_chat_template(processor, cfg, qtext, num_images=len(imgs))
    out = generate(model, processor, prompt, imgs, max_tokens=4, verbose=False)
    t = (out.text if hasattr(out, "text") else str(out)).strip().upper()
    for c in "ABCD":
        if c in t[:3]: return c.lower()
    return "a"

res = []
for q in qs:
    j = vocab.index(q["obj"]) if q["obj"] in vocab else None
    if j is None: continue
    mask = secs < q["tq"]
    idx = np.where(mask)[0]
    sc = S[idx, j]
    th = np.quantile(sc, 0.985)
    hits = sorted([idx[k] for k in np.where(sc >= th)[0]], key=lambda i: secs[i])
    # 사건 군집 (간격 120초)
    evs = []
    for i in hits:
        if evs and secs[i] - secs[evs[-1][-1]] <= 120: evs[-1].append(i)
        else: evs.append([i])
    evs = [e for e in evs if len(e) >= 2]
    if not evs:
        res.append(dict(id=q["id"], pred="a", answer=q["answer"], note="사건없음")); continue
    # "before" → 직전 사건 (마지막 사건이 질의 직전 5분 안이면 '사용 중' 으로 보고 이전 것)
    use = evs[-1]
    note = "최후사건"
    if "before" in q["q"].lower() and len(evs) >= 2 and q["tq"] - secs[evs[-1][-1]] < 300:
        use = evs[-2]; note = "직전사건"
    pick = sorted(use, key=lambda i: -S[i, j])[:2]
    imgs = [Image.open(files[int(secs[i])]).convert("RGB").resize((560, 420))
            for i in pick]
    pred = ask_choice(imgs, q["q"], q["choices"])
    res.append(dict(id=q["id"], pred=pred, answer=q["answer"], note=note,
                    ev_t=[int(secs[i]) for i in pick]))
    print(q["id"], q["obj"][:20], note, pred, "정답", q["answer"], flush=True)
ok = sum(r["pred"] == r["answer"].lower() for r in res)
print("v2: %d/%d = %.2f  (v1 0.30 · CLIP 0.20 · 우연 0.25)" % (ok, len(res), ok/len(res)))
open(OUT, "w").write("\n".join(json.dumps(r) for r in res))
