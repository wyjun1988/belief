#!/usr/bin/env python3
"""DAAAM 의 세그멘테이션+추적을 그대로 써서 GT 마스크를 대체한다 (GPU 쪽 실행).

    source /opt/ros/jazzy/setup.bash && source /workspace/ros2_ws/install/setup.bash
    python3 scripts/run_daaam_seg.py --seq /workspace/data/seq/<name> --out sam_daaam

DAAAM 파이프라인이 쓰는 두 서비스를 순서대로 돌린다:
    SegmentationService  FastSAM  → 마스크
    TrackingService      BotSort + CLIP ReID → **track id**

GT 마스크가 공짜로 주던 세 가지 중 ①마스크·②프레임 간 연관을 이 둘이 대신한다.
track id 를 픽셀값으로 써서 GT 와 **같은 형식**(`seg/%06d.png` uint16)으로 내보내므로
이후 파이프라인(그래프 빌더·구역·이벤트·belief)이 코드 변경 없이 그대로 읽는다.

남는 것은 ③카테고리(구역 시드용 "냉장고→kitchen")뿐이고, 그건 CLIP 제로샷이 맡는다
(`--clip` 으로 트랙별 임베딩을 남긴다).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None, help="DAAAM pipeline config yaml")
    ap.add_argument("--model", default=None, help="예: fastsam/FastSAM-x.pt")
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clip", action="store_true")
    args = ap.parse_args()

    from daaam.config import PipelineConfig
    from daaam.segmentation.services import SegmentationService
    from daaam.tracking.services import TrackingService

    cfg = (PipelineConfig.from_yaml(args.config, validate_files=False)
           if args.config else PipelineConfig())
    if args.model:
        cfg.segmentation.model_name = args.model
    cfg.segmentation.min_mask_region_area = args.min_area

    # ⚠️ `TrackingService` 생성이 **CUDA_VISIBLE_DEVICES 를 -1 로 바꿔버린다**
    # (boxmot 의 TensorRT ReID 백엔드 초기화 부작용). 그대로 두면 그 뒤 FastSAM 이
    # "Invalid CUDA device=0" 로 죽고 마스크가 0개가 된다. 트래커를 먼저 만들고
    # 환경변수를 되돌린 다음 세그멘터를 만든다.
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    trk_svc = TrackingService(cfg.tracking)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != _cvd:
        print("  (CUDA_VISIBLE_DEVICES %s → %s 로 복원)"
              % (os.environ.get("CUDA_VISIBLE_DEVICES"), _cvd), flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = _cvd
    seg_svc = SegmentationService(cfg.segmentation)
    print("세그멘터 %s / 추적 BotSort" % cfg.segmentation.model_name, flush=True)

    out = args.out or os.path.join(args.seq, "sam_daaam")
    os.makedirs(os.path.join(out, "seg"), exist_ok=True)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    if args.clip:
        os.makedirs(os.path.join(out, "clip"), exist_ok=True)

    clip_model = preprocess = None
    if args.clip:
        import open_clip
        import torch
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        clip_model = clip_model.cuda().eval()

    files = sorted(os.listdir(os.path.join(args.seq, "rgb")))
    if args.limit:
        files = files[:args.limit]

    n_masks, n_tracks, seen_ids = 0, 0, set()
    for n, f in enumerate(files):
        i = int(os.path.splitext(f)[0])
        img = np.array(Image.open(os.path.join(args.seq, "rgb", f)).convert("RGB"))
        dets, masks = seg_svc.segment(img)
        if len(dets) == 0:
            Image.fromarray(np.zeros(img.shape[:2], np.uint16)).save(
                os.path.join(out, "seg", "%06d.png" % i))
            json.dump([], open(os.path.join(out, "meta", "%06d.json" % i), "w"))
            continue
        tracks = trk_svc.update(np.asarray(dets, dtype=np.float32), img)
        n_masks += len(masks)

        # BotSort 출력: [x1,y1,x2,y2,track_id,conf,cls,det_idx] — 마지막 열이 검출 인덱스
        H, W = img.shape[:2]
        seg = np.zeros((H, W), np.uint16)
        meta = []
        rows = []
        for t in np.asarray(tracks):
            if len(t) < 8:
                continue
            tid, di = int(t[4]), int(t[7])
            if 0 <= di < len(masks):
                rows.append((tid, di, float(t[5])))
        # 큰 것부터 그려 작은 물체가 위에 오게 한다
        rows.sort(key=lambda r: -int(masks[r[1]].sum()))
        for tid, di, conf in rows:
            seg[np.asarray(masks[di], bool)] = tid
        for tid, di, conf in rows:
            m = seg == tid
            a = int(m.sum())
            if a < args.min_area:
                continue
            ys, xs = np.nonzero(m)
            meta.append({"track_id": tid, "area": a, "conf": conf,
                         "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
            seen_ids.add(tid)
        n_tracks += len(meta)
        Image.fromarray(seg).save(os.path.join(out, "seg", "%06d.png" % i))
        json.dump(meta, open(os.path.join(out, "meta", "%06d.json" % i), "w"))

        if args.clip and meta:
            import torch
            pil = Image.fromarray(img)
            crops = []
            for md in meta:
                x0, y0, x1, y1 = md["bbox"]
                crops.append(preprocess(pil.crop((max(x0 - 8, 0), max(y0 - 8, 0),
                                                  min(x1 + 8, W), min(y1 + 8, H)))))
            with torch.inference_mode():
                E = clip_model.encode_image(torch.stack(crops).cuda())
                E = (E / E.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)
            np.save(os.path.join(out, "clip", "%06d.npy" % i), E)

        if n % 100 == 0:
            print("  [%d/%d] 마스크 %d 트랙 %d (누적 고유 id %d)"
                  % (n, len(files), len(masks), len(meta), len(seen_ids)), flush=True)

    json.dump({"segmenter": cfg.segmentation.model_name, "frames": len(files),
               "masks": n_masks, "tracked": n_tracks, "unique_track_ids": len(seen_ids),
               "tracks_per_frame": n_tracks / max(len(files), 1)},
              open(os.path.join(out, "sam_meta.json"), "w"))
    print("DAAAM_SEG_DONE  프레임 %d  트랙 %d  고유 id %d  → %s"
          % (len(files), n_tracks, len(seen_ids), out))


if __name__ == "__main__":
    main()
