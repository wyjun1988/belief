#!/usr/bin/env python3
"""CLIP 제로샷 카테고리 — 구역 시드의 마지막 GT 의존을 끊는다.

    $P scripts/clip_zero_shot.py --seq <name> --src sam_daaam

GT 마스크가 공짜로 주던 세 가지 중 ①마스크는 FastSAM, ②연관은 BotSort+ReID 가 받았고
남은 ③카테고리를 여기서 받는다. 구역 분할이 "냉장고가 있으면 kitchen" 식 의미 시드에
의존하므로, 이게 없으면 벽 없는 거실·부엌·다이닝을 가를 근거가 사라진다.

입력은 `run_daaam_seg.py --clip` 이 남긴 트랙별 이미지 임베딩(open_clip ViT-B-16/openai,
512d, L2 정규화)이고, 여기서는 **같은 체크포인트의 텍스트 타워**(HF openai/clip-vit-base-patch16)
로 라벨을 임베딩해 코사인 비교한다. 두 타워가 같은 체크포인트여야 비교가 성립한다.

⚠️ 시드로 쓸 24개 카테고리만 후보로 두면 벽·바닥·커튼까지 억지로 가구가 된다.
**방해 라벨(distractor)** 을 같이 넣고, 상위 답이 시드 카테고리일 때만 채택한다.

출력: <src>/clip_categories.json + <src>/seg_ids_clip.json (category 만 CLIP 으로 교체)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.graph.regions import SEED_CATEGORIES        # noqa: E402

HJ = os.path.expanduser("~/work/home-jepa")
if HJ not in sys.path:
    sys.path.insert(0, HJ)
from homejepa.adt import CLS_MAP, FURN_CATS          # noqa: E402

# 뒷단이 실제로 읽는 어휘는 셋이다: 구역 시드(SEED_CATEGORIES), home-jepa 수용체(FURN_CATS),
# home-jepa 물체(CLS_MAP). 시드 어휘만 후보로 두면 액자·컵 같은 **동적 물체가 통째로
# 무라벨**이 되어 belief 질의가 0개가 된다(처음에 그렇게 나왔다).
ACCEPT = list(dict.fromkeys(list(SEED_CATEGORIES) + list(FURN_CATS) + list(CLS_MAP)))

# 시드가 아닌 것들 — 이게 없으면 벽·바닥이 전부 'couch' 가 된다
DISTRACTORS = [
    "a wall", "a floor", "a ceiling", "a door", "a window", "a curtain",
    "a person", "a rug", "a lamp", "a potted plant", "a picture frame on a wall",
    "a bookshelf", "a cardboard box", "a book", "a cup", "a plate", "a bottle",
    "a pillow", "a blanket", "a cabinet", "a countertop", "a staircase",
    "a mirror", "a clock", "a blurry background", "a close-up of a surface",
]
TEMPLATES = ["a photo of {}", "a photo of {} in a room",
             "a cropped photo of {}", "{}"]


def _article(label):
    return label if label.startswith(("a ", "an ")) else \
        ("an " + label if label[0] in "aeiou" else "a " + label)


def text_bank(labels, device="cpu"):
    import torch
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    mdl = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch16", use_safetensors=True).eval().to(device)
    out = []
    with torch.no_grad():
        for lb in labels:
            prompts = [t.format(_article(lb)) for t in TEMPLATES]
            e = mdl(**tok(prompts, padding=True, return_tensors="pt").to(device)).text_embeds
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)                      # 프롬프트 앙상블
            out.append((e / e.norm()).cpu().numpy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--src", default="sam_daaam")
    ap.add_argument("--min-area", type=int, default=400, help="이보다 작은 마스크는 집계 제외")
    ap.add_argument("--min-score", type=float, default=0.0, help="시드-방해 라벨 마진 하한")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    src = os.path.join(seq_dir, args.src)

    # --- 트랙별 임베딩 집계 (마스크 면적 가중 평균) -----------------------------
    acc, cnt = {}, {}
    frames = sorted(f for f in os.listdir(os.path.join(src, "meta")) if f.endswith(".json"))
    for f in frames:
        meta = json.load(open(os.path.join(src, "meta", f)))
        ep = os.path.join(src, "clip", f.replace(".json", ".npy"))
        if not meta or not os.path.exists(ep):
            continue
        E = np.load(ep).astype(np.float32)
        for k, md in enumerate(meta):
            if k >= len(E) or md["area"] < args.min_area:
                continue
            t = int(md["track_id"])
            w = float(np.sqrt(md["area"]))
            acc[t] = acc.get(t, 0) + w * E[k]
            cnt[t] = cnt.get(t, 0) + w
    tracks = sorted(acc)
    V = np.stack([acc[t] / cnt[t] for t in tracks])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.savez(os.path.join(src, "track_emb.npz"), ids=np.array(tracks), emb=V)
    print("트랙 %d개 집계 (프레임 %d) → track_emb.npz" % (len(tracks), len(frames)))

    # --- 텍스트 뱅크 --------------------------------------------------------------
    seed_labels = ACCEPT
    labels = seed_labels + DISTRACTORS
    T = text_bank(labels)
    Sim = V @ T.T
    n_seed = len(seed_labels)

    best_seed = Sim[:, :n_seed].argmax(1)
    best_dist = Sim[:, n_seed:].max(1)
    margin = Sim[np.arange(len(V)), best_seed] - best_dist

    ids = json.load(open(os.path.join(src, "seg_ids.json")))
    out, kept = {}, 0
    for i, t in enumerate(tracks):
        ok = margin[i] > args.min_score
        cat = seed_labels[best_seed[i]] if ok else None
        kept += bool(ok)
        gt = ids.get(str(t), {})
        out[str(t)] = dict(category=cat, margin=float(margin[i]),
                           score=float(Sim[i, best_seed[i]]),
                           top_distractor=labels[n_seed + int(Sim[i, n_seed:].argmax())],
                           gt_category=gt.get("category"), gt_name=gt.get("name"))
    json.dump(out, open(os.path.join(src, "clip_categories.json"), "w"), indent=1)
    print("시드 카테고리로 채택 %d / %d" % (kept, len(tracks)))

    # --- seg_ids 재작성: **category 만** 갈아끼운다 (extent·motion 은 그대로) -----
    new = {}
    for k, v in ids.items():
        r = dict(v)
        c = out.get(k)
        r["category"] = c["category"] if c else None
        r["clip_margin"] = c["margin"] if c else None
        new[k] = r
    p = os.path.join(src, "seg_ids_clip.json")
    json.dump(new, open(p, "w"))
    print("→ %s" % p)

    # --- 채점: GT 카테고리가 시드인 트랙에서 얼마나 맞혔나 -----------------------
    tp = fp = fn = 0
    conf = {}
    for k, c in out.items():
        g = (ids.get(k, {}).get("category") or "").lower()
        gz = SEED_CATEGORIES.get(g)
        pz = SEED_CATEGORIES.get(c["category"]) if c["category"] else None
        if gz and pz:
            if gz == pz:
                tp += 1
            else:
                fp += 1
                conf[(gz, pz)] = conf.get((gz, pz), 0) + 1
        elif gz and not pz:
            fn += 1
        elif pz and not gz:
            fp += 1
            conf[("(비시드)", pz)] = conf.get(("(비시드)", pz), 0) + 1
    print("\n구역 시드 기준 — 맞음 %d · 틀림 %d · 놓침 %d" % (tp, fp, fn))
    for (a, b), n in sorted(conf.items(), key=lambda x: -x[1])[:8]:
        print("   %-10s → %-10s %d" % (a, b, n))


if __name__ == "__main__":
    main()
