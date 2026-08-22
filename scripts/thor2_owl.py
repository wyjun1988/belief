#!/usr/bin/env python3
"""2차 데이터 사전계산 — 맵 프레임 + **간격 추출한** 배회 프레임.

    $P scripts/thor2_owl.py --root data/thor2 --cache /tmp/thor2cache --stride 10

⚠️ 1fps 2시간이면 7,200장이다. 전부에 OWL 을 돌리면 주택당 70분 — 6채면 7시간이다.
**`--stride 10`(10초 간격, 720장)으로 캐시하고 채점 때 더 성글게 뽑는다.**
이 자체가 실험 대상이다 — 얼마나 성글게 저장해도 답이 유지되는가.
"""
import argparse, glob, json, os, re

import numpy as np


def words(t):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--owl-batch", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    houses = sorted(glob.glob(os.path.join(args.root, "house_*")))
    types = set()
    for hd in houses:
        g = json.load(open(os.path.join(hd, "gt.json")))
        types |= {v["type"] for v in g["gt0"].values()}
    vocab = sorted(types)
    print("주택 %d · 물체 유형 %d · stride %d" % (len(houses), len(vocab), args.stride), flush=True)

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
        E, S = [], []
        for i in range(0, len(paths), args.batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + args.batch]]
            with torch.no_grad():
                e = cnet(**cp(images=ims, return_tensors="pt").to(
                    args.device)).image_embeds.cpu().numpy().astype(np.float32)
            E.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
            for k in range(0, len(ims), args.owl_batch):
                pv = op(images=ims[k:k + args.owl_batch],
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
        g = json.load(open(os.path.join(hd, "gt.json")))
        mp = sorted(glob.glob(os.path.join(hd, "map", "*.jpg")))
        lv = sorted(glob.glob(os.path.join(hd, "live", "*.jpg")))
        if len(mp) < 8 or len(lv) < 100:
            print("  %s 프레임 부족" % os.path.basename(hd), flush=True); continue
        sel = lv[::args.stride]
        ts = [int(os.path.basename(p)[:-4]) for p in sel]
        em, om_ = run(mp)
        el, ol = run(sel)
        np.savez_compressed(out, em=em, om=om_, el=el, ol=ol,
                            ts=np.array(ts), vocab=np.array(vocab, object))
        print("  %s · 맵 %d · 배회 %d/%d"
              % (os.path.basename(hd), len(mp), len(sel), len(lv)), flush=True)
    print("완료")


if __name__ == "__main__":
    main()
