#!/usr/bin/env python3
"""프레임 단위 CLIP 제로샷 **방 분류** → 구역 시드.

    $P scripts/clip_rooms.py --seq <name>

물체를 분류해 시드를 만드는 길(`clip_zero_shot.py`)은 침대에서 막혔다 — FastSAM 이
매트리스 **표면 조각**을 주는데 CLIP 은 그걸 베개와 못 가른다. 침실 시드가 0개가 되고
이웃의 가짜 kitchen 시드(Thermostat→microwave)가 그 공간을 삼켰다.

여기서는 물체를 건너뛰고 **프레임 전체를 방으로 분류**한다. "이 사진은 침실이다"는
"이 조각은 매트리스다"보다 CLIP 이 훨씬 잘하는 종류의 질문이고, 씬그래프도 이미
`zone_from_observer` 로 같은 결을 쓰고 있다.

시드 위치는 카메라 앞 `--ahead` m 지점이다. 관찰자 자신의 위치를 쓰면 문간에서
옆방을 들여다볼 때 라벨이 자기 방에 찍힌다.

출력: <seq>/zone_seeds_clip.json  (build_graph.py --zone-seeds 로 먹인다)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 방 라벨 → 기존 SEED_CATEGORIES 키로 되돌린다(regions.py 를 건드리지 않으려고).
ROOM_LABELS = {
    "a bedroom": ("bed frame", "bedroom"),
    "a kitchen": ("refrigerator", "kitchen"),
    "a living room": ("couch", "living"),
    "a dining room": ("dining table", "dining"),
    "a bathroom": ("toilet", "bathroom"),
    "a home office": ("desk", "office"),
}
# 방이 아닌 것 — 클로즈업·복도·바깥은 시드로 쓰면 안 된다
JUNK = ["a hallway", "a close-up of an object", "a close-up of a wall",
        "a blurry photo", "the outdoors", "a staircase", "a ceiling"]
TEMPLATES = ["a photo of {}", "a photo taken inside {}",
             "an indoor photo of {}", "{}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--pose", default="pose/poses.txt")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--ahead", type=float, default=1.6, help="시드를 카메라 앞 몇 m 에 찍나")
    ap.add_argument("--smooth", type=int, default=9, help="라벨 시간 다수결 창(프레임)")
    ap.add_argument("--min-prob", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reuse", action="store_true", help="저장된 유사도 행렬 재사용")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import (CLIPImageProcessor, CLIPTokenizer,
                              CLIPTextModelWithProjection, CLIPVisionModelWithProjection)

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    poses = np.loadtxt(os.path.join(seq_dir, args.pose)).reshape(-1, 4, 4)
    rgb = os.path.join(seq_dir, "rgb")
    files = sorted(f for f in os.listdir(rgb) if f.endswith(".jpg"))
    idx = list(range(0, min(len(files), len(poses)), args.every))
    dev = torch.device(args.device)
    cache = os.path.join(seq_dir, "clip_rooms_sim.npz")
    if args.reuse and os.path.exists(cache):
        z = np.load(cache); Sim = z["sim"]; idx = list(z["idx"]); labels = list(z["labels"])
        return _seed(args, seq_dir, poses, idx, Sim)

    name = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(name)
    txt = CLIPTextModelWithProjection.from_pretrained(name, use_safetensors=True).eval().to(dev)
    proc = CLIPImageProcessor.from_pretrained(name)
    vis = CLIPVisionModelWithProjection.from_pretrained(name, use_safetensors=True).eval().to(dev)

    labels = list(ROOM_LABELS) + JUNK
    with torch.no_grad():
        T = []
        for lb in labels:
            e = txt(**tok([t.format(lb) for t in TEMPLATES], padding=True,
                          return_tensors="pt").to(dev)).text_embeds
            e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
            T.append(e / e.norm())
        T = torch.stack(T)

    sims = []
    with torch.no_grad():
        for b in range(0, len(idx), args.batch):
            imgs = [Image.open(os.path.join(rgb, files[i])).convert("RGB")
                    for i in idx[b:b + args.batch]]
            px = proc(images=imgs, return_tensors="pt").pixel_values.to(dev)
            E = vis(pixel_values=px).image_embeds
            E = E / E.norm(dim=-1, keepdim=True)
            sims.append((E @ T.T).cpu().numpy())
            if b % (args.batch * 10) == 0:
                print("  [%d/%d]" % (b, len(idx)), flush=True)
    Sim = np.concatenate(sims)                       # (N, labels)
    np.savez(os.path.join(seq_dir, "clip_rooms_sim.npz"), sim=Sim,
             idx=np.array(idx), labels=np.array(labels))

    return _seed(args, seq_dir, poses, idx, Sim)


def _seed(args, seq_dir, poses, idx, Sim):
    n_room = len(ROOM_LABELS)
    P = np.exp(Sim * 100.0)                          # CLIP 로짓 스케일
    P /= P.sum(1, keepdims=True)
    lab = P[:, :n_room].argmax(1)
    prob = P[:, :n_room].max(1)
    junk = P[:, n_room:].sum(1)

    # 시간 다수결 — 한 프레임짜리 깜빡임은 시드로 쓰면 안 된다
    if args.smooth > 1:
        h = args.smooth // 2
        sm = lab.copy()
        for i in range(len(lab)):
            w = lab[max(i - h, 0):i + h + 1]
            sm[i] = np.bincount(w).argmax()
        lab = sm

    rooms = list(ROOM_LABELS)
    raw = np.bincount(P[:, :n_room].argmax(1), minlength=n_room)
    print("\n임계 전 최빈 라벨: " + " · ".join("%s %d" % (r.replace("a ",""), n)
                                            for r, n in zip(rooms, raw)))
    print("확률 분위 %s · junk 분위 %s"
          % (np.percentile(prob, [10, 50, 90]).round(2), np.percentile(junk, [10, 50, 90]).round(2)))
    seeds, kept = [], np.zeros(len(rooms), int)
    for k, i in enumerate(idx):
        if prob[k] < args.min_prob or junk[k] > 0.5:
            continue
        T_wc = poses[i]
        p = T_wc[:3, 3] + T_wc[:3, 2] * args.ahead   # 카메라 광축 앞
        cat, zone = ROOM_LABELS[rooms[lab[k]]]
        seeds.append({"name": "frame%06d" % i, "category": cat,
                      "extent_m": [1.0, 1.0, 1.0],   # SEED_MIN_EXTENT 통과용
                      "position": [float(v) for v in p],
                      "zone": zone, "prob": float(prob[k]), "frame": int(i)})
        kept[lab[k]] += 1

    out = args.out or os.path.join(seq_dir, "zone_seeds_clip.json")
    json.dump(seeds, open(out, "w"))
    print("\n프레임 %d개 분류 → 시드 %d개" % (len(idx), len(seeds)))
    for r, n in zip(rooms, kept):
        print("   %-16s %4d 프레임" % (r, n))
    print("→ %s" % out)


if __name__ == "__main__":
    main()
