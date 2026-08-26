#!/usr/bin/env python3
"""VLM 검증 스모크 테스트 — 진짜 목격 크롭 vs 오검출 크롭을 가르는가. (thor5)

    THOR_ROOT=data/thor5 python scripts/exp_vlm_verify.py

GT(vis+ctr)로 양성/음성 크롭을 뽑아 "Is there a {type}?" 에 yes/no 로 답하게 한다.
OWL 이 못 가른 것(문턱 통과 오검출)만 음성으로 쓴다 — 실전에서 VLM 이 받을 그 분포다.
"""
import glob, json, os, sys
import numpy as np
from PIL import Image
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor5")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/a5_5_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/q5_5_"))
N = int(os.environ.get("N", "60"))
CROP = int(os.environ.get("CROP", "128"))

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
model, processor = load("RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
cfg = model.config


def ask(img, t):
    words = "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
    mode = os.environ.get("QMODE", "strict")
    if mode == "soft":
        q = "Is there a %s in this image? Answer only yes or no." % words
    elif mode == "open":
        q = "Name the main small object at the center of this image in one or two words."
    else:
        q = "Look at the image. Is there a %s clearly visible? Answer only yes or no." % words
    prompt = apply_chat_template(processor, cfg, q, num_images=1)
    out = generate(model, processor, prompt, [img], max_tokens=8, verbose=False)
    txt = (out.text if hasattr(out, "text") else str(out)).strip().lower()
    if os.environ.get("QMODE") == "open":
        return any(w in txt for w in words.split())
    return txt.startswith("yes")


pos = []; neg = []
rng = np.random.default_rng(0)
for hd in sorted(glob.glob(ROOT + "/house_*")):
    if len(pos) >= N and len(neg) >= N: break
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    live = {m["t"]: m for m in g["live"]}
    lv = {int(os.path.basename(p)[:-4]): p
          for p in glob.glob(os.path.join(os.path.realpath(hd), "live", "*.jpg"))}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        TS = QS[:, j] + STx[:, j]
        th = np.quantile(TS, 0.98)
        hits = np.where(TS >= th)[0]
        rng.shuffle(hits)
        for i in hits[:4]:
            t = int(ts[i])
            if t not in lv: continue
            m = live[t]
            true_vis = oid in m.get("vis", [])
            if true_vis and len(pos) >= N: continue
            if not true_vis and len(neg) >= N: continue
            if true_vis and m.get("ctr", {}).get(oid):
                cx, cy = m["ctr"][oid]
            else:
                cx = (P[i, ti] % pw + .5) / pw * 384
                cy = (P[i, ti] // pw + .5) / ph * 384
            im = Image.open(lv[t]).convert("RGB")
            h = CROP // 2
            box = (max(0, int(cx)-h), max(0, int(cy)-h),
                   min(384, int(cx)+h), min(384, int(cy)+h))
            crop = im.crop(box).resize((336, 336), Image.LANCZOS)
            (pos if true_vis else neg).append((crop, v0["type"]))

print("크롭: 양성 %d · 음성 %d" % (len(pos), len(neg)), flush=True)
tp = sum(ask(c, t) for c, t in pos)
tn = sum(not ask(c, t) for c, t in neg)
print("=== VLM 검증 스모크 (%s · 크롭 %dpx→336) ===" % (ROOT, CROP))
print("  진짜 목격을 yes  **%.3f** (%d/%d)" % (tp/max(len(pos),1), tp, len(pos)))
print("  오검출을  no   **%.3f** (%d/%d)" % (tn/max(len(neg),1), tn, len(neg)))
