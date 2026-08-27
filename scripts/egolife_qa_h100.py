#!/usr/bin/env python3
"""EgoLife where-질의 종단 — OWL 검색 캐시 + Qwen 로짓 선택지 대조. (H100)

    FRAMES=/path/egl_frames QA=/path/EgoLifeQA_A1_JAKE.json MODEL=Qwen/Qwen3.5-9B \\
      OUT=egl_answers.jsonl python scripts/egolife_qa_h100.py

단계:
  ① 질문에서 물체 구문 추출 → OWL 로 전 프레임 점수+argmax 패치 (캐시, 재개 가능)
  ② 질의 시각 이전 상위 프레임 → 패치 크롭
  ③ 크롭 + 4지선다를 Qwen 로짓으로 대조 (A/B/C/D 첫 토큰 로짓 — §85 방식)
채점은 로컬(정답 키 포함 jsonl 반환). 어안 원형 밖 픽셀·타임스탬프 오버레이는
크롭 단계에서 자연 회피(중앙부 argmax 위주).
"""
import json, os, re
import numpy as np
from PIL import Image
import torch

FRAMES = os.environ.get("FRAMES", "/tmp/egl_frames")
QAF = os.environ.get("QA", "EgoLifeQA_A1_JAKE.json")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
OUT = os.environ.get("OUT", "egl_answers.jsonl")
CACHE = os.environ.get("CACHE", "/tmp/egl_owl.npz")
TOPK = int(os.environ.get("TOPK", "5"))

qa = json.load(open(QAF))
qs = []
for q in qa:
    if q["type"] != "EntityLog": continue
    ql = q["question"].lower()
    if not ql.startswith("where"): continue
    obj = re.sub(r"^where (was|were|is|are) (the )?", "", ql)
    obj = obj.split(" before")[0].split(" placed")[0].split(" in ")[0].split("?")[0].strip()
    tm = q["query_time"]["time"]
    qs.append(dict(id=q["ID"], obj=obj, tq=int(tm[:2])*3600+int(tm[2:4])*60+int(tm[4:6]),
                  choices=[q["choice_%s" % c] for c in "abcd"], answer=q["answer"]))
vocab = sorted({q["obj"] for q in qs})
print("where 질문 %d · 물체 구문 %d" % (len(qs), len(vocab)), flush=True)

import glob as _g
files = sorted(_g.glob(os.path.join(FRAMES, "*.jpg")))
secs = np.array([int(os.path.basename(f)[:-4]) for f in files])

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    S, P, ph, pw = z["s"], z["p"], int(z["ph"]), int(z["pw"])
else:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    opr = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble", dtype=torch.float16).to(dev).eval()
    ti = opr(text=[["a photo of a " + v for v in vocab]],
             images=[Image.new("RGB", (256, 256), (128,)*3)], return_tensors="pt").to(dev)
    with torch.no_grad():
        o = om.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                     pixel_values=ti["pixel_values"].half(), return_dict=True)
    TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
    S_, P_ = [], []
    B = 16
    for i in range(0, len(files), B):
        ims = [Image.open(f).convert("RGB") for f in files[i:i+B]]
        pv = opr(images=ims, return_tensors="pt")["pixel_values"].half().to(dev)
        with torch.no_grad():
            fm = om.image_embedder(pixel_values=pv)[0]
            b, ph, pw, hd = fm.shape
            lg, _ = om.class_predictor(fm.reshape(b, ph*pw, hd),
                                       TX.unsqueeze(0).expand(b, -1, -1),
                                       MK.unsqueeze(0).expand(b, -1))
        pr_ = torch.sigmoid(lg)
        S_.append(pr_.amax(1).float().cpu().numpy())
        P_.append(pr_.argmax(1).int().cpu().numpy())
        if i % 800 == 0: print("  OWL %d/%d" % (i, len(files)), flush=True)
    S, P = np.concatenate(S_), np.concatenate(P_)
    np.savez_compressed(CACHE, s=S, p=P, ph=ph, pw=pw, secs=secs,
                        vocab=np.array(vocab, object))
    print("캐시 저장 →", CACHE, flush=True)

from transformers import AutoProcessor, AutoModelForImageTextToText
vpr = AutoProcessor.from_pretrained(MODEL)
vm = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto").eval()
tok = vpr.tokenizer
IDS = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]

def choice_logits(imgs, obj, choices):
    q = ("Look at the image(s) showing '%s'. Where was it? "
         "(A) %s (B) %s (C) %s (D) %s. Answer only A, B, C, or D."
         % (obj, *choices))
    msgs = [{"role": "user", "content": [{"type": "image"}]*len(imgs)
             + [{"type": "text", "text": q}]}]
    try:
        text = vpr.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = vpr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = vpr(images=imgs, text=text, return_tensors="pt").to(vm.device)
    with torch.no_grad():
        lg = vm(**inp).logits[0, -1]
    return [float(lg[i]) for i in IDS]

W = 760
out = open(OUT, "w")
for q in qs:
    j = vocab.index(q["obj"])
    mask = secs < q["tq"]
    if mask.sum() < 10: continue
    idx = np.where(mask)[0]
    top = idx[np.argsort(-S[idx, j])[:TOPK]]
    imgs = []
    for i in top[:3]:
        im = Image.open(files[i]).convert("RGB")
        cx = (P[i, j] % pw + .5) / pw * im.width
        cy = (P[i, j] // pw + .5) / ph * im.height
        s = 160
        imgs.append(im.crop((int(cx-s), int(cy-s), int(cx+s), int(cy+s))).resize((336, 336)))
    lg = choice_logits(imgs, q["obj"], q["choices"])
    out.write(json.dumps(dict(id=q["id"], obj=q["obj"], pred="abcd"[int(np.argmax(lg))],
                              answer=q["answer"], logits=[round(x, 3) for x in lg],
                              frames=[int(secs[i]) for i in top[:3]])) + "\n")
    out.flush()
out.close()
print("→", OUT)
