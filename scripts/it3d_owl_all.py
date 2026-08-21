#!/usr/bin/env python3
"""IT3DEgo 조각의 **모든 프레임**에 OWL 을 돌려 캐시한다 (엔드투엔드용).

    $P scripts/it3d_owl_all.py --slices data/it3dego/pv --ann … --cache … --stride 3

㉞ 의 부재 측정은 창에 쓰인 28% 프레임만 OWL 을 돌렸다. 엔드투엔드는 **검색**이
마지막 목격 프레임을 스스로 찾아야 해서 창이 매번 달라진다 → 전 프레임이 필요하다.

`--stride 3` 이면 영상당 약 320장(조각 960장의 1/3). 실측 0.60 s/장(M1 Pro) ·
1.25 s/장(iMac) 이므로 영상당 3~7분이다.
"""
import argparse, io, json, os, re, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.it3d_absence import base_label, load_ann              # noqa: E402

PVRE = re.compile(r"raw_videos/(video_\d+_scene_\d+)/pv/(\d+)\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--owl-batch", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

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

    vids = []
    for f in sorted(os.listdir(args.slices)):
        if f.startswith(".") or not f.endswith(".index.json"):
            continue
        vn = f[:-len(".index.json")]
        if args.videos and vn not in args.videos:
            continue
        if os.path.exists(os.path.join(args.slices, vn + ".bin")):
            vids.append(vn)
    print("영상 %d개" % len(vids), flush=True)

    for vn in vids:
        out = os.path.join(args.cache, vn + ".all.npz")
        if os.path.exists(out):
            print("  %-20s 이미 있음" % vn, flush=True); continue
        ad = os.path.join(args.ann, vn)
        if not os.path.isdir(ad):
            print("  %-20s 어노테이션 없음" % vn, flush=True); continue
        labs, _, _ = load_ann(ad)
        words = [base_label(l) for l in labs]
        frames = []
        for r in json.load(open(os.path.join(args.slices, vn + ".index.json"))):
            m = PVRE.match(r["name"])
            if m and m.group(1) == vn:
                frames.append((int(m.group(2)), r["off"], r["size"]))
        frames.sort()
        frames = frames[::args.stride]
        q = ["a photo of a " + w for w in words]
        ti = op(text=[q], images=[Image.new("RGB", (256, 256), (128, 128, 128))],
                return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = onet.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                           pixel_values=ti["pixel_values"], return_dict=True)
        TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
        ts, E, S = [], [], []
        with open(os.path.join(args.slices, vn + ".bin"), "rb") as tf:
            for i in range(0, len(frames), args.batch):
                ims = []
                for t, off, sz in frames[i:i + args.batch]:
                    tf.seek(off)
                    try:
                        ims.append(Image.open(io.BytesIO(tf.read(sz))).convert("RGB"))
                        ts.append(t)
                    except Exception:
                        pass
                if not ims:
                    continue
                with torch.no_grad():
                    e = cnet(**cp(images=ims, return_tensors="pt").to(
                        args.device)).image_embeds.cpu().numpy().astype(np.float32)
                E.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
                for j in range(0, len(ims), args.owl_batch):
                    pv = op(images=ims[j:j + args.owl_batch],
                            return_tensors="pt")["pixel_values"].to(args.device)
                    with torch.no_grad():
                        fm = onet.image_embedder(pixel_values=pv)[0]
                        b, ph, pw, hd = fm.shape
                        lg, _ = onet.class_predictor(
                            fm.reshape(b, ph * pw, hd),
                            TX.unsqueeze(0).expand(b, -1, -1),
                            MK.unsqueeze(0).expand(b, -1))
                    S.append(torch.sigmoid(lg).amax(1).float().cpu().numpy())
        np.savez_compressed(out, ts=np.array(ts, np.int64),
                            emb=np.concatenate(E), owl=np.concatenate(S),
                            words=np.array(words, object))
        print("  %-20s 프레임 %d · 물체 %d" % (vn, len(ts), len(words)), flush=True)


if __name__ == "__main__":
    main()
