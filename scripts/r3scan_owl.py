#!/usr/bin/env python3
"""3RScan 스캔별 OWL 검출 사전계산 (전역 어휘 386개).

    $P scripts/r3scan_owl.py --root /Volumes/exDisk/3rscan --cache … --nframes 24

`sequence.zip` 을 **풀지 않고** zipfile 로 프레임을 직접 읽는다(스캔당 443프레임 ·
83MB). 어휘가 386개뿐이라 텍스트 임베딩을 **한 번만** 만들고 프레임마다
image_embedder 를 1회 돌린다(㊱ 과 같은 경로).

⚠️ exFAT 에서 `._<이름>` AppleDouble 이 딸려온다 — 디렉터리 목록에서 점으로
시작하는 것은 반드시 건너뛴다(㊱ 에서 이것 때문에 전수 실행이 죽었다).
"""
import argparse, io, json, os, zipfile

import numpy as np


def frames_of(zp, n):
    """zip 안 color.jpg 를 균등 간격으로 n 장."""
    with zipfile.ZipFile(zp) as z:
        names = sorted(x for x in z.namelist() if x.endswith(".color.jpg"))
        if not names:
            return [], []
        pick = [names[i] for i in np.linspace(0, len(names) - 1, min(n, len(names))).astype(int)]
        return pick, [z.read(x) for x in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nframes", type=int, default=24)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--owl-batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    vocab = json.load(open(os.path.join(args.root, "vocab.json")))
    om = "google/owlv2-base-patch16-ensemble"
    op = Owlv2Processor.from_pretrained(om)
    onet = Owlv2ForObjectDetection.from_pretrained(om).to(args.device).eval()
    q = ["a photo of a " + w for w in vocab]
    ti = op(text=[q], images=[Image.new("RGB", (256, 256), (128, 128, 128))],
            return_tensors="pt").to(args.device)
    with torch.no_grad():
        o = onet.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                       pixel_values=ti["pixel_values"], return_dict=True)
    TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
    print("어휘 %d · 텍스트 임베딩 %s 캐시" % (len(vocab), tuple(TX.shape)), flush=True)

    sd = os.path.join(args.root, "scans")
    ids = sorted(x for x in os.listdir(sd) if not x.startswith("."))
    if args.limit:
        ids = ids[:args.limit]
    done = 0
    for sid in ids:
        out = os.path.join(args.cache, sid + ".npz")
        zp = os.path.join(sd, sid, "sequence.zip")
        if os.path.exists(out) or not os.path.exists(zp) or os.path.getsize(zp) < 1000:
            continue
        try:
            names, blobs = frames_of(zp, args.nframes)
        except Exception as e:
            print("  %s zip 오류 %s" % (sid[:8], e), flush=True); continue
        if len(blobs) < 4:
            continue
        ims = []
        for b in blobs:
            try:
                ims.append(Image.open(io.BytesIO(b)).convert("RGB"))
            except Exception:
                pass
        if len(ims) < 4:
            continue
        S = []
        for i in range(0, len(ims), args.owl_batch):
            pv = op(images=ims[i:i + args.owl_batch],
                    return_tensors="pt")["pixel_values"].to(args.device)
            with torch.no_grad():
                fm = onet.image_embedder(pixel_values=pv)[0]
                b_, ph, pw, hd = fm.shape
                lg, _ = onet.class_predictor(fm.reshape(b_, ph * pw, hd),
                                             TX.unsqueeze(0).expand(b_, -1, -1),
                                             MK.unsqueeze(0).expand(b_, -1))
                S.append(torch.sigmoid(lg).amax(1).float().cpu().numpy())
        np.savez_compressed(out, owl=np.concatenate(S),
                            frames=np.array(names[:len(ims)], object))
        done += 1
        if done % 10 == 0:
            print("  %d 스캔 완료 (%s)" % (done, sid[:8]), flush=True)
    print("완료 %d" % done)


if __name__ == "__main__":
    main()
