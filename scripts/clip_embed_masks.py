#!/usr/bin/env python3
"""이미 뽑아둔 마스크 메타(bbox)에 **CLIP 임베딩만** 다시 입힌다.

    python3 scripts/clip_embed_masks.py --seq /workspace/data/seq/<name> --src sam_daaam

`run_daaam_seg.py --clip` 은 세그·추적·임베딩을 한 번에 하는데, 임베딩만 다시 필요할 때
DAAAM 스택 전체를 띄우는 것은 낭비이고 실패 지점도 늘어난다(TensorRT ReID 가중치 등).
여기서는 `meta/%06d.json` 의 bbox 를 그대로 읽어 크롭·인코딩만 한다.

⚠️ decoration 과 **완전히 같은 조건**이어야 임베딩이 비교 가능하다:
    모델 open_clip ViT-B-16 / pretrained="openai", bbox 를 8px 확장, L2 정규화, fp16.
이미 있는 파일은 건너뛰므로 중단된 실행을 이어서 돌릴 수 있다.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--src", default="sam_daaam")
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="있는 파일도 다시 만든다")
    args = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image

    src = os.path.join(args.seq, args.src)
    out = os.path.join(src, "clip")
    os.makedirs(out, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(args.device).eval()

    files = sorted(f for f in os.listdir(os.path.join(src, "meta")) if f.endswith(".json"))
    todo = [f for f in files
            if args.force or not os.path.exists(os.path.join(out, f.replace(".json", ".npy")))]
    print("프레임 %d개 중 %d개 처리" % (len(files), len(todo)), flush=True)

    for n, f in enumerate(todo):
        meta = json.load(open(os.path.join(src, "meta", f)))
        dst = os.path.join(out, f.replace(".json", ".npy"))
        if not meta:
            np.save(dst, np.zeros((0, 512), np.float16))
            continue
        img = Image.open(os.path.join(args.seq, "rgb", f.replace(".json", ".jpg"))).convert("RGB")
        W, H = img.size
        crops = []
        for md in meta:
            x0, y0, x1, y1 = md["bbox"]
            crops.append(preprocess(img.crop((max(x0 - args.pad, 0), max(y0 - args.pad, 0),
                                              min(x1 + args.pad, W), min(y1 + args.pad, H)))))
        E = []
        with torch.inference_mode():
            for b in range(0, len(crops), args.batch):
                e = model.encode_image(torch.stack(crops[b:b + args.batch]).to(args.device))
                E.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
        np.save(dst, np.concatenate(E))
        if n % 100 == 0:
            print("  [%d/%d] %s 마스크 %d" % (n, len(todo), f, len(meta)), flush=True)

    print("완료 — clip/ %d개" % len(os.listdir(out)))


if __name__ == "__main__":
    main()
