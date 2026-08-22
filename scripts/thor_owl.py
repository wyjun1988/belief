#!/usr/bin/env python3
"""ProcTHOR 생성 데이터에 CLIP(장소용) · OWL(물체용) 사전계산.

    $P scripts/thor_owl.py --root data/thor --cache /tmp/thorcache

어휘는 **전 주택의 물체 유형 합집합** — 텍스트 임베딩을 한 번만 만든다.
`CoffeeTable` 같은 카멜케이스를 `coffee table` 로 풀어 질의한다.
"""
import argparse, glob, json, os, re

import numpy as np


def words(t):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--owl-batch", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    houses = sorted(glob.glob(os.path.join(args.root, "house_*")))
    types = set()
    for hd in houses:
        g = json.load(open(os.path.join(hd, "gt.json")))
        types |= {v["type"] for v in g["gt1"].values()}
    vocab = sorted(types)
    print("주택 %d · 물체 유형 %d" % (len(houses), len(vocab)), flush=True)

    import torch
    from PIL import Image
    from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                              CLIPImageProcessor, CLIPVisionModelWithProjection)
    om = "google/owlv2-base-patch16-ensemble"
    op = Owlv2Processor.from_pretrained(om)
    onet = Owlv2ForObjectDetection.from_pretrained(om).to(args.device).eval()
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cnet = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    q = ["a photo of a " + words(w) for w in vocab]
    ti = op(text=[q], images=[Image.new("RGB", (256, 256), (128, 128, 128))],
            return_tensors="pt").to(args.device)
    with torch.no_grad():
        o = onet.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                       pixel_values=ti["pixel_values"], return_dict=True)
    TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)

    def run(paths):
        ims = [Image.open(p).convert("RGB") for p in paths]
        E, S = [], []
        for i in range(0, len(ims), args.batch):
            with torch.no_grad():
                e = cnet(**cp(images=ims[i:i + args.batch], return_tensors="pt").to(
                    args.device)).image_embeds.cpu().numpy().astype(np.float32)
            E.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
        for i in range(0, len(ims), args.owl_batch):
            pv = op(images=ims[i:i + args.owl_batch],
                    return_tensors="pt")["pixel_values"].to(args.device)
            with torch.no_grad():
                fm = onet.image_embedder(pixel_values=pv)[0]
                b, ph, pw, hd = fm.shape
                lg, _ = onet.class_predictor(fm.reshape(b, ph * pw, hd),
                                             TX.unsqueeze(0).expand(b, -1, -1),
                                             MK.unsqueeze(0).expand(b, -1))
            S.append(torch.sigmoid(lg).amax(1).float().cpu().numpy())
        return np.concatenate(E), np.concatenate(S)

    for hd in houses:
        out = os.path.join(args.cache, os.path.basename(hd) + ".npz")
        if os.path.exists(out):
            continue
        p1 = sorted(glob.glob(os.path.join(hd, "s1", "*.jpg")))
        p2 = sorted(glob.glob(os.path.join(hd, "s2", "*.jpg")))
        if len(p1) < 8 or len(p2) < 4:
            continue
        e1, o1 = run(p1); e2, o2 = run(p2)
        np.savez_compressed(out, e1=e1, o1=o1, e2=e2, o2=o2,
                            vocab=np.array(vocab, object))
        print("  %s · s1 %d · s2 %d" % (os.path.basename(hd), len(p1), len(p2)), flush=True)
    print("완료")


if __name__ == "__main__":
    main()
