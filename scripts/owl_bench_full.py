#!/usr/bin/env python3
"""지각층 전수 벤치 — CLIP 전체프레임 z vs OWLv2, ADT GT 세그 전 프레임.

    $P scripts/owl_bench_full.py --owl data/adt_owl/owl_adt_decoration.json

기존 owl_detect.py 는 40프레임·16어휘 표본이었다. 여기선 918프레임·122어휘를
전부 채점한다 — 지각층 교체가 이 파이프라인의 근거이므로 표본이 커야 한다.

정답: GT 인스턴스 세그에서 **충분히 크게 보인**(≥600px) 카테고리.
CLIP 은 프레임 전체 임베딩이라 "그 물건이 있다"가 아니라 "그 장면이 그 단어와
비슷하다"만 말한다. OWLv2 는 물체를 직접 찾는다. 이 차이를 정밀도·재현율로 가른다.
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Apartment_release_decoration_seq137_M1292")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--owl", required=True)
    ap.add_argument("--min-px", type=int, default=600)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    sd = os.path.join(args.root, args.seq)
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
    ids = json.load(open(os.path.join(sd, "gt", "seg_ids.json")))
    cat_of = {}
    for local, m in ids.items():
        rec = gt.get(str(m.get("gt_instance") or m.get("instance_id")))
        if rec and rec.get("category"):
            cat_of[int(local)] = rec["category"].strip().lower()

    det = json.load(open(args.owl))
    seg_dir = os.path.join(sd, "gt", "seg")
    files = sorted(f for f in os.listdir(seg_dir) if f.endswith(".png"))
    truth, keys = [], []
    for f in files:
        k = f.replace(".png", ".jpg")
        if k not in det:
            continue
        s = np.array(Image.open(os.path.join(seg_dir, f)))
        u, c = np.unique(s, return_counts=True)
        truth.append({cat_of[int(a)] for a, n in zip(u, c)
                      if int(a) in cat_of and n >= args.min_px})
        keys.append(k)
    vocab = sorted({w for d in det.values() for w in d} |
                   {c for t in truth for c in t})
    print("프레임 %d · 어휘 %d · 프레임당 정답 물체 중앙 %.1f개"
          % (len(keys), len(vocab), np.median([len(t) for t in truth])))

    # CLIP 전체프레임 z (현행)
    from scripts.absence_evidence import clip_text
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    E, fidx = z["emb"].astype(np.float32), z["idx"]
    fmap = {int(f): i for i, f in enumerate(fidx)}
    cols = [fmap[int(k.split(".")[0])] for k in keys]
    V = clip_text(["a photo of a " + w for w in vocab], args.device)
    S = V @ E.T
    Z = ((S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9))[:, cols]

    vi = {w: i for i, w in enumerate(vocab)}
    G = np.zeros((len(vocab), len(keys)), bool)
    for j, t in enumerate(truth):
        for w in t:
            G[vi[w], j] = True
    O = np.zeros((len(vocab), len(keys)), np.float32)
    for j, k in enumerate(keys):
        for w, s in det[k].items():
            O[vi[w], j] = s

    def score(P):
        tp = int((P & G).sum()); fp = int((P & ~G).sum()); fn = int((~P & G).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)

    print("\n%-24s %-8s %-8s %s" % ("방식", "정밀도", "재현율", "F1"))
    best = {}
    for zt in (0.5, 1.0, 1.5, 2.0):
        r = score(Z >= zt)
        print("CLIP 전체프레임 z≥%.1f    %.2f     %.2f     %.2f" % ((zt,) + r))
        if r[2] > best.get("clip", (0, 0, 0))[2]:
            best["clip"] = r
    # CLIP 에 더 유리한 판정규칙 — 프레임당 상위 k개. 문턱 방식은 단어마다
    # z 분포가 달라 불리할 수 있으니, 프레임 안에서 순위만 쓰는 규칙도 같이 잰다.
    for k in (5, 11, 20):
        P = np.zeros_like(G)
        for j in range(Z.shape[1]):
            P[np.argsort(-Z[:, j])[:k], j] = True
        r = score(P)
        print("CLIP 프레임당 상위%-2d      %.2f     %.2f     %.2f" % ((k,) + r))
        if r[2] > best.get("clip", (0, 0, 0))[2]:
            best["clip"] = r
    for ot in (0.10, 0.15, 0.20, 0.30):
        r = score(O >= ot)
        print("**OWLv2 ≥%.2f**           %.2f     %.2f     %.2f" % ((ot,) + r))
        if r[2] > best.get("owl", (0, 0, 0))[2]:
            best["owl"] = r
    print("\n최고 F1 — CLIP %.2f → **OWLv2 %.2f** (%.1f배)"
          % (best["clip"][2], best["owl"][2], best["owl"][2] / max(best["clip"][2], 1e-9)))


if __name__ == "__main__":
    main()
