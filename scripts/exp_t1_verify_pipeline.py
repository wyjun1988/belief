#!/usr/bin/env python3
"""T1 파이프라인 내 실검증 — 모의가 아니라 진짜 크롭으로. (H100 · thor4)

    MODEL=Qwen/Qwen3.5-9B THOR_ROOT=data/thor4 TH=0.875 \\
      OUT_JSONL=t1_verified.jsonl python scripts/exp_t1_verify_pipeline.py

각 타입단일 타겟: 후보(점수 q0.80+) 를 최신부터 걸으며 프레임 크롭을 2AFC 로짓으로
검증(문턱 TH — hh 실측 기각0.99 지점), 3장 확인되면 정지. 확인 프레임 목록을
JSONL 로 — 국소화·T1 채점은 로컬에서 한다.

⚠️ 모의와 다른 점: 실제 오검출은 같은 혼동물의 반복이라 FA 가 상관될 수 있다.
이 실행이 그 위험을 판정한다. 대조 라벨은 그 패치에서 두 번째로 강한 타겟 타입.
"""
import glob, json, os
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.environ.get("A3_PREFIX", "/tmp/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/qc_")
TH = float(os.environ.get("TH", "0.875"))
OUTJ = os.environ.get("OUT_JSONL", "t1_verified.jsonl")
MAXWALK = int(os.environ.get("MAXWALK", "25"))
pr = AutoProcessor.from_pretrained(MODEL)
md = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto").eval()
tok = pr.tokenizer
IDA = tok.encode("A", add_special_tokens=False)[0]
IDB = tok.encode("B", add_special_tokens=False)[0]
def words(t): return "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
print("모델 %s · 문턱 %.3f" % (MODEL, TH), flush=True)


def sab(img, a, b):
    q = "Which object is in this image: (A) %s or (B) %s? Answer only A or B." % (a, b)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    try:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=[img], text=text, return_tensors="pt").to(md.device)
    with torch.no_grad():
        lg = md(**inp).logits[0, -1]
    return float(lg[IDA] - lg[IDB])


from collections import Counter
out = open(OUTJ, "w")
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    lv = {int(os.path.basename(p)[:-4]): p
          for p in glob.glob(os.path.join(hd, "live", "*.jpg"))}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    moves = {m["oid"]: m for m in g["moves"]}
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        if oid not in moves: continue                    # T1 후보(이동)만 — 비용 절약
        ti = vocab.index(v0["type"])
        TS = QS[:, j] + STx[:, j]
        th80 = np.quantile(TS, 0.80)
        cands = sorted(np.where(TS >= th80)[0], key=lambda i: -ts[i])[:MAXWALK]
        ver = []; walked = 0
        for i in cands:
            t = int(ts[i])
            if t not in lv: continue
            walked += 1
            cx = (P[i, ti] % pw + .5) / pw * 384
            cy = (P[i, ti] // pw + .5) / ph * 384
            im = Image.open(lv[t]).convert("RGB")
            c = im.crop((max(0, int(cx)-64), max(0, int(cy)-64),
                         min(384, int(cx)+64), min(384, int(cy)+64))).resize((336, 336))
            alt_c = int(np.argsort(-S[i, :nT])[1] if np.argsort(-S[i, :nT])[0] == ti
                        else np.argsort(-S[i, :nT])[0])
            s = sab(c, words(v0["type"]), words(vocab[alt_c]))
            if s >= TH:
                ver.append(int(i))
                if len(ver) == 3: break
        out.write(json.dumps(dict(house=hn, oid=oid, verified=ver, walked=walked)) + "\n")
        out.flush()
    print(hn, "완료", flush=True)
out.close()
print("→", OUTJ)
