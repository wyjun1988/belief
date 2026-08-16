#!/usr/bin/env python3
"""SuperMemory-VQA 파일럿 — 물체·위치 기억 QA 의 근거 검색 채점.

    $P scripts/supermem_pilot.py --stage embed
    $P scripts/supermem_pilot.py --stage eval

EgoLife 파일럿과 같은 잣대(1fps CLIP + 허브니스 중심화 + 시간 평활 + NMS)를
쓰되, 여기는 **답 근거 구간(answer_evidence.time_spans)** 이 초 단위로 주석돼
있어 채점이 정확하다. 질문은 물체 중심("작은 흰 병 어디 뒀지")이라 행위 중심이던
EgoLife 보다 CLIP 에 유리하다는 가설을 시험한다.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
SESS = {
    "Person_1_session_8_03102026_glasses_1264": "s8",
    "Person_1_session_14_03152026_glasses_1266": "s14",
}


def stage_embed(args):
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    dev = torch.device(args.device)
    name = "openai/clip-vit-base-patch16"
    proc = CLIPImageProcessor.from_pretrained(name)
    vis = CLIPVisionModelWithProjection.from_pretrained(name, use_safetensors=True).eval().to(dev)
    for vid, sd in SESS.items():
        cap = cv2.VideoCapture(os.path.join(D, sd, "video.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        embs, tss, batch, bts = [], [], [], []

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

        for sec in range(int(n / fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (352, 352))
            batch.append(Image.fromarray(fr))
            bts.append(float(sec))
            if len(batch) >= args.batch:
                flush()
        flush()
        cap.release()
        E = np.concatenate(embs)
        np.savez(os.path.join(D, sd, "index.npz"), emb=E.astype(np.float16),
                 ts=np.array(tss))
        print("%s: 프레임 %d (%.1f분)" % (sd, len(E), len(E) / 60))


def stage_eval(args):
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    E, ts, sid = [], [], []
    for vid, sd in SESS.items():
        z = np.load(os.path.join(D, sd, "index.npz"))
        E.append(z["emb"].astype(np.float32))
        ts.extend(z["ts"])
        sid.extend([vid] * len(z["ts"]))
    E = np.concatenate(E)
    ts = np.array(ts)
    sid = np.array(sid)

    q = json.load(open(os.path.join(D, "qa_person_1.json")))
    Q = []
    for x in q:
        ev = [(e["video_id"], e["time_span"]["start_time"], e["time_span"]["end_time"])
              for e in ((x.get("answer_evidence") or {}).get("evidence_list") or [])
              if e.get("video_id") in SESS and e.get("time_span")]
        if ev:
            Q.append((x, ev))
    print("채점 문항 %d개 (인덱스 %d프레임·두 세션)" % (len(Q), len(E)))

    dev = torch.device(args.device)
    name = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(name)
    txt = CLIPTextModelWithProjection.from_pretrained(name, use_safetensors=True).eval().to(dev)
    with torch.no_grad():
        tt = tok(["a photo of " + x["question"] for x, _ in Q],
                 padding=True, truncation=True, return_tensors="pt").to(dev)
        te = torch.nn.functional.normalize(txt(**tt).text_embeds, dim=-1).cpu().numpy()
    S = te @ E.T
    S = S - S.mean(0, keepdims=True)                    # 허브니스 중심화
    k15 = np.ones(15) / 15
    S = np.apply_along_axis(lambda r: np.convolve(r, k15, mode="same"), 1, S)

    rng = np.random.default_rng(0)
    tol = args.tol
    from collections import defaultdict
    for k in (1, 5, 20):
        by = defaultdict(lambda: [0, 0])
        hits = rand = 0
        for i, (x, ev) in enumerate(Q):
            order = np.argsort(-S[i])
            picked = []
            for j in order:
                if all(abs(ts[j] - ts[p]) > 30 or sid[j] != sid[p] for p in picked):
                    picked.append(j)
                if len(picked) >= k:
                    break
            def hit_at(js):
                return any(sid[j] == v and a - tol <= ts[j] <= b + tol
                           for j in js for v, a, b in ev)
            h = hit_at(picked)
            hits += h
            rand += hit_at(rng.choice(len(ts), k, replace=False))
            sk = x["metadata"]["skill"]
            by[sk][0] += h
            by[sk][1] += 1
        print("hit@%-2d (±%ds): **%.2f** (무작위 %.2f)" % (k, tol, hits / len(Q), rand / len(Q)))
        if k == 5:
            for skl, (h, n) in sorted(by.items()):
                print("   %-24s %.2f (%d)" % (skl, h / n, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["embed", "eval"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tol", type=float, default=30.0)
    args = ap.parse_args()
    (stage_embed if args.stage == "embed" else stage_eval)(args)


if __name__ == "__main__":
    main()
