#!/usr/bin/env python3
"""SAM 세그멘테이션 — GT 마스크를 실제 모델로 대체 (GPU 쪽에서 실행).

    python3 scripts/run_sam.py --seq /workspace/data/seq/<name> --model fastsam
    python3 scripts/run_sam.py --seq ... --model sam2 --clip

출력 (GT 와 같은 형식이라 이후 파이프라인이 그대로 읽는다):
    sam_<model>/seg/%06d.png   uint16 마스크 id — **프레임 안에서만 유효**(추적 안 함)
    sam_<model>/meta/%06d.json 마스크별 area/bbox/score
    sam_<model>/clip/%06d.npy  (M,D) CLIP 임베딩 — 연관·분류에 쓴다 (--clip)

⚠️ GT 마스크는 지금 **세 가지**를 공짜로 주고 있었다: ①마스크 ②프레임 간 연관(인스턴스
id) ③카테고리(구역 시드). SAM 은 ①만 준다. ②는 추적/ReID, ③은 CLIP 제로샷이 맡아야
하고, 그래서 이 스크립트는 마스크와 함께 CLIP 임베딩을 남긴다.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def _fastsam(model_path, device):
    from ultralytics import FastSAM
    m = FastSAM(model_path)

    def run(img_path, imgsz, conf, iou):
        r = m(img_path, device=device, retina_masks=True, imgsz=imgsz, conf=conf, iou=iou,
              verbose=False)[0]
        if r.masks is None:
            return np.zeros((0, 0, 0), bool), np.zeros(0)
        return r.masks.data.cpu().numpy().astype(bool), r.boxes.conf.cpu().numpy()
    return run


def _sam2(model_path, device, points_per_side=16, min_area=400):
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", model_path, device=device)
    gen = SAM2AutomaticMaskGenerator(sam, points_per_side=points_per_side,
                                     min_mask_region_area=min_area,
                                     pred_iou_thresh=0.8, stability_score_thresh=0.9)

    def run(img_path, imgsz, conf, iou):
        img = np.array(Image.open(img_path).convert("RGB"))
        with torch.inference_mode():
            anns = gen.generate(img)
        if not anns:
            return np.zeros((0, 0, 0), bool), np.zeros(0)
        M = np.stack([a["segmentation"] for a in anns])
        S = np.array([a.get("predicted_iou", 1.0) for a in anns])
        return M, S
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="fastsam", choices=["fastsam", "sam2"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=704)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--min-area", type=int, default=250)
    ap.add_argument("--max-frac", type=float, default=0.35,
                    help="이 비율 넘게 화면을 덮는 마스크는 배경으로 보고 버린다")
    ap.add_argument("--clip", action="store_true", help="마스크별 CLIP 임베딩도 저장")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = args.out or os.path.join(args.seq, "sam_" + args.model)
    for sub in ("seg", "meta") + (("clip",) if args.clip else ()):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    w = args.weights or ("FastSAM-x.pt" if args.model == "fastsam"
                         else "/workspace/checkpoints/sam2.1_hiera_large.pt")
    run = _fastsam(w, args.device) if args.model == "fastsam" else _sam2(w, args.device)

    clip_model = preprocess = None
    if args.clip:
        import open_clip
        import torch
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        clip_model = clip_model.to(args.device).eval()

    files = sorted(os.listdir(os.path.join(args.seq, "rgb")))
    if args.limit:
        files = files[:args.limit]
    tot_masks = 0
    for n, f in enumerate(files):
        i = int(os.path.splitext(f)[0])
        p = os.path.join(args.seq, "rgb", f)
        M, S = run(p, args.imgsz, args.conf, args.iou)
        H, W = (M.shape[1:] if len(M) else Image.open(p).size[::-1])
        keep = []
        for k in range(len(M)):
            a = int(M[k].sum())
            if a < args.min_area or a > args.max_frac * H * W:
                continue
            keep.append(k)
        # 큰 것부터 그려서 작은 것이 위에 오도록 (작은 물체가 큰 가구에 먹히지 않게)
        keep.sort(key=lambda k: -int(M[k].sum()))
        seg = np.zeros((H, W), np.uint16)
        meta = []
        for lid, k in enumerate(keep, start=1):
            seg[M[k]] = lid
        for lid, k in enumerate(keep, start=1):
            m = seg == lid
            a = int(m.sum())
            if a < args.min_area:                 # 위에 덮여 사라진 것
                continue
            ys, xs = np.nonzero(m)
            meta.append({"id": lid, "area": a, "score": float(S[k]) if len(S) else 1.0,
                         "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
        Image.fromarray(seg).save(os.path.join(out, "seg", "%06d.png" % i))
        json.dump(meta, open(os.path.join(out, "meta", "%06d.json" % i), "w"))
        tot_masks += len(meta)

        if args.clip and meta:
            import torch
            img = Image.open(p).convert("RGB")
            crops = []
            for md in meta:
                x0, y0, x1, y1 = md["bbox"]
                pad = 8
                crops.append(preprocess(img.crop((max(x0 - pad, 0), max(y0 - pad, 0),
                                                  min(x1 + pad, W), min(y1 + pad, H)))))
            with torch.inference_mode():
                E = clip_model.encode_image(torch.stack(crops).to(args.device))
                E = (E / E.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)
            np.save(os.path.join(out, "clip", "%06d.npy" % i), E)
        if n % 100 == 0:
            print("  [%d/%d] 마스크 %d개 (누적 평균 %.1f)"
                  % (n, len(files), len(meta), tot_masks / max(n + 1, 1)), flush=True)

    json.dump({"model": args.model, "frames": len(files), "total_masks": tot_masks,
               "masks_per_frame": tot_masks / max(len(files), 1),
               "params": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                          "min_area": args.min_area, "max_frac": args.max_frac}},
              open(os.path.join(out, "sam_meta.json"), "w"))
    print("SAM_DONE  프레임 %d  마스크 %d  (프레임당 %.1f) → %s"
          % (len(files), tot_masks, tot_masks / max(len(files), 1), out))


if __name__ == "__main__":
    main()
