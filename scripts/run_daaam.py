#!/usr/bin/env python3
"""DAAAM(+Khronos+Hydra) 를 ADT 시퀀스에 돌린다 — RunPod 쪽에서 실행.

    source /opt/ros/jazzy/setup.bash && source /workspace/ros2_ws/install/setup.bash
    python3 scripts/run_daaam.py --seq /workspace/data/seq/<name> \
        --out /workspace/results/<tag> --hydra-config configs/hydra_adt_khronos.yaml

1차 구성(사용자 결정): **기하·동역학만**. 세그멘테이션은 ADT GT, DAM 캡셔닝은 끈다.
그래서 이 실행이 묻는 것은 하나다 — "DA3 뎁스로 Khronos 액티브 윈도우가 4D 그래프를
갱신하는가". 시맨틱(오픈보캡 설명)은 P4.

DAAAM 저장소는 건드리지 않는다. 데이터셋 타입 등록과 세그멘터 교체 두 지점만
런타임에 끼워 넣는다(`kx/bridge/`).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="내보낸 시퀀스 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="DAAAM pipeline config yaml")
    ap.add_argument("--hydra-config",
                    default=os.path.join(ROOT, "configs", "hydra_adt_khronos.yaml"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--depth-dir", default="depth",
                    help="시퀀스 안에서 쓸 뎁스 폴더 (애블레이션 시 depth_t23 등)")
    ap.add_argument("--min-mask-area", type=int, default=200)
    ap.add_argument("--max-extent-m", type=float, default=2.5,
                    help="이보다 큰 GT 인스턴스(벽·바닥)는 물체 노드로 올리지 않는다")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # 애블레이션용 뎁스 폴더는 depth/ 심링크로 갈아끼운다 (로더는 depth/ 만 본다).
    if args.depth_dir != "depth":
        link = os.path.join(args.seq, "depth")
        target = os.path.join(args.seq, args.depth_dir)
        if not os.path.isdir(target):
            sys.exit("없는 뎁스 폴더: %s" % target)
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.isdir(link):
            sys.exit("depth/ 가 실제 폴더다 — 애블레이션은 심링크 전제. 수동 정리 필요.")
        os.symlink(args.depth_dir, link)

    from kx.bridge.adt_dataset import AdtDataset, register
    from kx.bridge.gt_segmenter import patch_daaam

    register()
    seg = patch_daaam(args.seq, min_area=args.min_mask_area,
                      max_extent_m=args.max_extent_m, mode="masks_only")

    from daaam.config import PipelineConfig
    from daaam.hydra.runner import HydraPipelineRunner

    cfg = (PipelineConfig.from_yaml(args.config, validate_files=False)
           if args.config else PipelineConfig())
    # 1차는 시맨틱 없이 — DAM/VLM 을 끄면 24GB 로 넉넉하다.
    for section in ("grounding", "workers"):
        sec = getattr(cfg, section, None)
        for attr in ("enabled", "num_grounding_workers"):
            if sec is not None and hasattr(sec, attr):
                setattr(sec, attr, False if attr == "enabled" else 0)

    ds = AdtDataset(args.seq, {"fps": args.fps})
    os.makedirs(args.out, exist_ok=True)
    runner = HydraPipelineRunner(
        config=cfg,
        dataset=ds,
        hydra_config_path=args.hydra_config,
        output_dir=args.out,
        target_fps=args.fps,
        dataset_name="adt",
        match_ros_log_dir=False,
        zmq_url=None,                 # ZMQ 구독자를 띄우지 않는다
        show_progress=True,
    )
    stats = runner.run(max_frames=args.limit)
    runner.shutdown()
    print("GT 마스크 미스(워밍업 등): %d" % seg.misses)
    print(json.dumps(stats, indent=1, default=str)[:1200] if stats else "(stats 없음)")
    print("DAAAM_RUN_DONE -> %s" % args.out)


if __name__ == "__main__":
    main()
