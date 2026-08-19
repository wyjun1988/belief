#!/usr/bin/env python3
"""EgoLife 파일럿 — v2 운영층(1fps 잠재 + 검색)이 주week급 실데이터에서 서는지.

    $P scripts/egolife_pilot.py --stage embed     # 30초 클립 → 1fps CLIP 인덱스
    $P scripts/egolife_pilot.py --stage eval      # EgoLifeQA 근거시점 검색 채점

시험 내용: EgoLifeQA 의 질문마다 근거 시점(target_time)이 주석돼 있다.
질문 키워드의 CLIP 텍스트 임베딩으로 1fps 프레임 인덱스를 검색해, top-k 안에
근거 시점 ±허용창 프레임이 들어오는지(hit@k) 를 잰다. ADT 와 달리 포즈·GT 가
없으므로 이 파일럿은 **검색층만** 검증한다 — 기하 부트스트랩은 AEA(MPS 포함) 몫.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "egolife")


def hms_to_sec(t):
    """'11152408' = 11:15:24.08 → 초"""
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + int(t[6:8]) / 100.0


def stage_embed(args):
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    dev = torch.device(args.device)
    name = "openai/clip-vit-base-patch16"
    proc = CLIPImageProcessor.from_pretrained(name)
    vis = CLIPVisionModelWithProjection.from_pretrained(name, use_safetensors=True).eval().to(dev)

    clips = sorted(f for f in os.listdir(os.path.join(D, args.clips)) if f.endswith(".mp4"))
    embs, tss = [], []
    batch, bts = [], []

    def flush():
        nonlocal batch, bts
        if not batch:
            return
        with torch.no_grad():
            px = proc(images=batch, return_tensors="pt")["pixel_values"].to(dev)
            e = vis(pixel_values=px).image_embeds
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        embs.append(e)
        tss.extend(bts)
        batch, bts = [], []

    for ci, f in enumerate(clips):
        t0 = hms_to_sec(f.split("_")[-1].split(".")[0])
        cap = cv2.VideoCapture(os.path.join(D, args.clips, f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for sec in range(int(n / fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            batch.append(Image.fromarray(fr))
            bts.append(t0 + sec)
            if len(batch) >= args.batch:
                flush()
        cap.release()
        if ci % 30 == 0:
            print("클립 %d/%d · 프레임 %d" % (ci, len(clips), sum(len(e) for e in embs) + len(batch)))
    flush()
    E = np.concatenate(embs)
    np.savez(os.path.join(D, "index_a1_day1.npz"), emb=E.astype(np.float16),
             ts=np.array(tss, np.float64))
    print("인덱스 저장: 프레임 %d · %.1f분 분량" % (len(E), (max(tss) - min(tss)) / 60))


def stage_eval(args):
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    z = np.load(os.path.join(D, "index_a1_day1.npz"))
    E, ts = z["emb"].astype(np.float32), z["ts"]
    lo, hi = ts.min(), ts.max()
    q = json.load(open(os.path.join(D, "EgoLifeQA", "EgoLifeQA_A1_JAKE.json")))
    Q = [x for x in q
         if (x.get("target_time") or {}).get("date") == "DAY1"
         and (x.get("target_time") or {}).get("time")
         and lo <= hms_to_sec(x["target_time"]["time"]) <= hi]
    print("평가 질문 %d개 (인덱스 창 %.1f분)" % (len(Q), (hi - lo) / 60))

    dev = torch.device(args.device)
    name = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(name)
    txt = CLIPTextModelWithProjection.from_pretrained(name, use_safetensors=True).eval().to(dev)

    texts = [(x.get("keywords") or x["question"]) for x in Q]
    with torch.no_grad():
        tt = tok(["a photo of " + t for t in texts], padding=True, return_tensors="pt").to(dev)
        te = torch.nn.functional.normalize(txt(**tt).text_embeds, dim=-1).cpu().numpy()

    sims = te @ E.T                                    # (Q, N)
    tol = args.tol
    rng = np.random.default_rng(0)
    for k in (1, 5, 20):
        hits, rand = 0, 0.0
        for i, x in enumerate(Q):
            tgt = hms_to_sec(x["target_time"]["time"])
            top = np.argsort(-sims[i])[:k]
            hits += any(abs(ts[j] - tgt) <= tol for j in top)
            rj = rng.choice(len(ts), k, replace=False)
            rand += any(abs(ts[j] - tgt) <= tol for j in rj)
        print("hit@%-2d (±%ds): **%.2f** (무작위 %.2f)"
              % (k, tol, hits / len(Q), rand / len(Q)))

    # 타입별 hit@5
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    for i, x in enumerate(Q):
        tgt = hms_to_sec(x["target_time"]["time"])
        top = np.argsort(-sims[i])[:5]
        by[x["type"]][0] += any(abs(ts[j] - tgt) <= tol for j in top)
        by[x["type"]][1] += 1
    for t, (h, n) in sorted(by.items()):
        print("  %-12s hit@5 %.2f (%d)" % (t, h / n, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["embed", "eval"])
    ap.add_argument("--clips", default="A1_DAY1")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tol", type=float, default=60.0, help="근거 허용창(초)")
    args = ap.parse_args()
    if args.stage == "embed":
        stage_embed(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()
