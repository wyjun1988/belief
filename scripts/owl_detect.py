#!/usr/bin/env python3
"""오픈보캡 검출기(OWLv2)로 지각층 교체 — CLIP 전체프레임 z 의 대안.

    $P scripts/owl_detect.py --seq <adt-seq> --stage bench   # GT 로 CLIP vs OWLv2 채점
    $P scripts/owl_detect.py --seq <adt-seq> --stage run     # 프레임별 검출 저장

왜 바꾸는가(실측 근거): 실영상에서 CLIP **전체 프레임** 점수는 "이 그림이 그 물체와
비슷한가" 만 말한다. 그래서 방 분류 0.49, 가구 belief 0.16, 부재 블라인드 정밀도
0.09 에 머물렀다. 오픈보캡 검출기는 **박스 단위로 '있다/없다'** 를 답하므로,
존재판정·가구 접지·부재 게이트가 모두 같은 병목에서 풀린다.

채점: ADT 는 GT 세그멘테이션이 있어 프레임별 물체 존재의 정답을 안다.
같은 프레임·같은 어휘로 CLIP z 문턱과 OWLv2 박스 검출을 나란히 잰다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL = "google/owlv2-base-patch16-ensemble"


def load_owl(device="cpu"):
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).eval().to(device)
    return proc, model


def detect(proc, model, images, queries, device="cpu", thr=0.15):
    """[PIL] × 질의어 → 프레임별 {질의 인덱스: 최고 점수}"""
    import torch
    out = []
    for im in images:
        inputs = proc(text=[queries], images=im, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inputs)
        r = proc.post_process_grounded_object_detection(
            o, threshold=thr, target_sizes=torch.tensor([im.size[::-1]]))[0]
        best = {}
        for s, l in zip(r["scores"].tolist(), r["labels"].tolist()):
            best[l] = max(best.get(l, 0.0), s)
        out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Apartment_release_decoration_seq137_M1292")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--stage", default="bench", choices=["bench", "run"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--n-vocab", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
    ids = json.load(open(os.path.join(sd, "gt", "seg_ids.json")))
    # GT 세그에서 프레임별 '실제로 보인 카테고리' 를 만든다 (정답)
    import collections
    from PIL import Image as PILImage
    seg_dir = os.path.join(sd, "gt", "seg")
    files = sorted(f for f in os.listdir(seg_dir) if f.endswith(".png"))
    step = max(1, len(files) // args.n_frames)
    picks = files[::step][:args.n_frames]
    cat_of = {}
    for local, m in ids.items():
        rec = gt.get(str(m.get("gt_instance") or m.get("instance_id")))
        if rec and rec.get("category"):
            cat_of[int(local)] = rec["category"].strip().lower()
    truth, frames = [], []
    for f in picks:
        s = np.array(PILImage.open(os.path.join(seg_dir, f)))
        u, c = np.unique(s, return_counts=True)
        vis = {cat_of[int(a)] for a, n in zip(u, c)
               if int(a) in cat_of and n >= 600}      # 충분히 크게 보인 것만
        truth.append(vis)
        frames.append(PILImage.open(os.path.join(sd, "rgb", f.replace(".png", ".jpg"))).convert("RGB"))
    # 어휘: 자주 보이는 카테고리 상위 N
    freq = collections.Counter(c for t in truth for c in t)
    vocab = [c for c, _ in freq.most_common(args.n_vocab)]
    print("프레임 %d · 어휘 %d개 · 프레임당 정답 물체 중앙 %.1f개"
          % (len(frames), len(vocab), np.median([len(t & set(vocab)) for t in truth])))

    # ① CLIP 전체프레임 z (현행 방식)
    from scripts.absence_evidence import clip_text
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    E, fidx = z["emb"].astype(np.float32), z["idx"]
    V = clip_text(["a photo of a " + w for w in vocab], "mps")
    S = V @ E.T
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    fmap = {int(f): i for i, f in enumerate(fidx)}
    cols = [fmap[int(p.split(".")[0])] for p in picks if int(p.split(".")[0]) in fmap]

    # ② OWLv2 박스 검출
    proc, model = load_owl(args.device)
    print("OWLv2 검출 중… (프레임 %d)" % len(frames))
    det = detect(proc, model, frames, ["a photo of a " + w for w in vocab],
                 args.device, args.thr)

    # 채점: 프레임×어휘 이진 판정
    def score(pred_bin):
        tp = fp = fn = 0
        for i, t in enumerate(truth):
            for j, w in enumerate(vocab):
                p = pred_bin[i][j]
                g = w in t
                tp += p and g
                fp += p and not g
                fn += (not p) and g
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)

    print("\n%-22s %-8s %-8s %s" % ("방식", "정밀도", "재현율", "F1"))
    for zt in (1.0, 1.5, 2.0):
        pb = [[Z[j, cols[i]] >= zt for j in range(len(vocab))] for i in range(len(cols))]
        print("CLIP 전체프레임 z≥%.1f   %.2f     %.2f     %.2f" % ((zt,) + score(pb)))
    for ot in (0.10, 0.15, 0.20):
        pb = [[det[i].get(j, 0.0) >= ot for j in range(len(vocab))] for i in range(len(frames))]
        print("**OWLv2 박스 ≥%.2f**      %.2f     %.2f     %.2f" % ((ot,) + score(pb)))

    if args.out:
        json.dump(dict(vocab=vocab, frames=[p for p in picks],
                       det=[{str(k): v for k, v in d.items()} for d in det]),
                  open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
