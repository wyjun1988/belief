#!/usr/bin/env python3
"""T1 실검증 — mlx 네이티브판 (Apple Silicon). exp_t1_verify_pipeline 과 같은 산출.

    THOR_ROOT=data/hssd20 A3_PREFIX=/tmp/hs_a_ QC_PREFIX=/tmp/hs_q_ \\
      OUT_JSONL=/tmp/t1_scores_hs.jsonl MAXWALK=40 \\
      ~/mlx-venv/bin/python scripts/exp_t1_verify_mlx.py

타입단일 이동 타겟의 후보(FLOOR 분위 문턱→최신순)를 크롭해 s_ab/s_ac 로짓 기록.
박스 크롭(bx, BOXES=1 캐시) 우선 — §117. 문턱 판정은 로컬 스윕(§89).
"""
import glob, json, os
import numpy as np
from collections import Counter
from PIL import Image
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
ROOT = os.environ.get("THOR_ROOT", "data/hssd20")
A3P = os.environ.get("A3_PREFIX", "/tmp/hs_a_")
QCP = os.environ.get("QC_PREFIX", "/tmp/hs_q_")
OUTJ = os.environ.get("OUT_JSONL", "/tmp/t1_scores_hs.jsonl")
MAXWALK = int(os.environ.get("MAXWALK", "40"))
FLOOR = float(os.environ.get("FLOOR", "0.80"))

model, processor = load(MODEL)
cfg = model.config
tok = processor.tokenizer
IDS = {t: tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B", "C")}


def logits(img_path_or_im, q):
    prompt = apply_chat_template(processor, cfg, q, num_images=1)
    inp = prepare_inputs(processor, images=[img_path_or_im], prompts=[prompt],
                         image_token_index=getattr(cfg, "image_token_index", None))
    out = model(inp["input_ids"], inp["pixel_values"], mask=inp.get("attention_mask"),
                **{k: v for k, v in inp.items()
                   if k not in ("input_ids", "pixel_values", "attention_mask")})
    lg = out.logits[0, -1]
    mx.eval(lg)
    return lg


def words(t): return t.replace("_", " ").lower()

out = open(OUTJ, "w")
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    BXa = za["bx"] if "bx" in za.files else None
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    lv = {int(os.path.basename(p)[:-4]): p
          for p in glob.glob(os.path.join(hd, "live", "*.jpg"))}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    moves = {m["oid"]: m for m in g["moves"]}
    for j, oid in enumerate(QT):
        v0 = g["gt0"].get(oid)
        if not v0 or not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        if oid not in moves: continue
        ti = vocab.index(v0["type"])
        TS = QS[:, j] + STx[:, j]
        th = np.quantile(TS, FLOOR)
        cands = sorted(np.where(TS >= th)[0], key=lambda i: -int(ts[i]))[:MAXWALK]
        ver = []
        for i in cands:
            t = int(ts[i])
            if t not in lv: continue
            im = Image.open(lv[t]).convert("RGB")
            W, H = im.size
            if BXa is not None:
                bcx, bcy, bw, bh = [float(x) * max(W, H) for x in BXa[i, ti]]
                h2 = max(48, int(max(bw, bh) * 0.65)); cx, cy = bcx, bcy
            else:
                cx = (P[i, ti] % pw + .5) / pw * W
                cy = (P[i, ti] // pw + .5) / ph * H
                h2 = max(64, W // 6)
            crop = im.crop((max(0, int(cx)-h2), max(0, int(cy)-h2),
                            min(W, int(cx)+h2), min(H, int(cy)+h2))).resize((336, 336))
            cp = "/tmp/_t1c.jpg"; crop.save(cp, quality=92)
            a = words(v0["type"])
            alt_c = int(np.argsort(-S[i, :nT])[1] if np.argsort(-S[i, :nT])[0] == ti
                        else np.argsort(-S[i, :nT])[0])
            b = words(vocab[alt_c])
            lg = logits(cp, "Which object is in this image: (A) %s or (B) %s? "
                        "Answer only A or B." % (a, b))
            s_ab = float(lg[IDS["A"]] - lg[IDS["B"]])
            lg = logits(cp, "Which is in this image: (A) %s, (B) %s, or (C) neither? "
                        "Answer only A, B, or C." % (a, b))
            s_ac = float(lg[IDS["A"]] - max(float(lg[IDS["B"]]), float(lg[IDS["C"]])))
            ver.append([int(i), round(s_ab, 3), round(s_ac, 3)])
        out.write(json.dumps(dict(house=hn, oid=oid, scored=ver, walked=len(ver))) + "\n")
        out.flush()
    print(hn, "완료", flush=True)
out.close()
print("→", OUTJ)
