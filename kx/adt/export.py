"""ADT 시퀀스 → DAAAM `ImageSequenceDataset` 레이아웃으로 내보내기.

DAAAM 은 `rgb/ depth/ pose/ camera_info.json` 폴더를 보면 `image_sequence` 로 자동
인식한다(`daaam/datasets/factory.py:_detect_dataset_type`). 여기서는 그 규약에 맞춰
쓰되, **`depth/` 는 비워 둔다** — 거기 들어갈 것은 P1 의 DA3 추정 뎁스이고,
ADT 의 진짜 뎁스는 `gt/depth/` 로 격리해 **평가에만** 쓴다. 입력으로 새면 실험이 무의미해진다.

출력:
    rgb/000000.jpg          정립 선형 RGB
    pose/poses.txt          한 줄 16값 = T_world_camera (ImageSequenceDataset 규약)
    camera_info.json        선형 K
    frames.json             프레임 인덱스 ↔ 디바이스 타임스탬프
    gt/depth/000000.png     uint16 mm  (평가 전용)
    gt/seg/000000.png       uint16 로컬 인스턴스 id
    gt/seg_ids.json         로컬 id → ADT instance id + 카테고리
    export.json             설정과 통계
"""
import json
import os
from typing import Optional

import numpy as np
from PIL import Image
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects import adt

from kx.adt.calib import camera_info, make_rig

RGB_STREAM = "214-1"


def _uniform_indices(ts_ns, fps, t_lo, t_hi):
    """디바이스 타임스탬프 목록에서 목표 fps 에 가장 가까운 프레임들을 고른다.

    ADT RGB 는 30Hz 라 3프레임마다 뽑으면 되지만, 드롭 프레임이 있으면 인덱스
    간격이 어긋난다. 균일 시간 격자에 최근접 매칭하는 편이 안전하다.

    ⚠️ VRS 스트림은 GT 보다 **먼저 시작한다**(파티 seq102 기준 11.7초). 그 구간을
    그대로 뽑으면 provider 가 경계값으로 클램프한 포즈·마스크를 조용히 돌려주기
    때문에 앞부분 100여 프레임이 통째로 오염된다. GT 유효 구간으로 먼저 자른다.
    """
    ts = np.asarray(ts_ns, dtype=np.int64)
    ts = ts[(ts >= t_lo) & (ts <= t_hi)]
    if len(ts) < 2:
        raise RuntimeError("GT 유효 구간 [%d, %d] 안에 RGB 프레임이 없다" % (t_lo, t_hi))
    step_ns = int(round(1e9 / fps))
    grid = np.arange(ts[0], ts[-1] + 1, step_ns, dtype=np.int64)
    idx = np.searchsorted(ts, grid)
    idx = np.clip(idx, 1, len(ts) - 1)
    pick = np.where(np.abs(ts[idx] - grid) < np.abs(ts[idx - 1] - grid), idx, idx - 1)
    return np.unique(ts[pick])


def export_depth(seq_dir: str, out_dir: str) -> dict:
    """이미 내보낸 시퀀스에 GT 뎁스만 덧붙인다.

    뎁스 아카이브는 시퀀스당 3~8GB 라 RGB 보다 한참 늦게 도착한다. RGB 내보내기를
    다시 돌리지 않고 `frames.json` 의 타임스탬프를 그대로 재사용한다 — 인덱스가
    어긋나면 평가가 통째로 틀어지므로 프레임 선정은 절대 다시 하지 않는다.
    """
    frames = json.load(open(os.path.join(out_dir, "frames.json")))
    cam = json.load(open(os.path.join(out_dir, "camera_info.json")))

    paths_provider = adt.AriaDigitalTwinDataPathsProvider(seq_dir)
    stats_path = os.path.join(out_dir, "export.json")
    prev = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    paths = paths_provider.get_datapaths(prev.get("skeleton_variant", False))
    provider = adt.AriaDigitalTwinDataProvider(paths)
    if not provider.has_depth_images():
        raise RuntimeError("뎁스 VRS 없음: %s (fetch_adt.py --parts depth)" % seq_dir)

    stream_id = StreamId(RGB_STREAM)
    rig = make_rig(provider.get_aria_camera_calibration(stream_id),
                   size=cam["width"], focal=float(cam["fx"]))
    dst = os.path.join(out_dir, "gt", "depth")
    os.makedirs(dst, exist_ok=True)

    ok = 0
    for fr in frames["frames"]:
        d = provider.get_depth_image_by_timestamp_ns(int(fr["t_ns"]), stream_id)
        if not d.is_valid():
            continue
        Image.fromarray(rig.depth(d.data().to_numpy_array())).save(
            os.path.join(dst, "%06d.png" % fr["index"]))
        ok += 1

    prev.update({"has_depth_gt": True, "depth_frames": ok})
    with open(stats_path, "w") as f:
        json.dump(prev, f, indent=1)
    return {"depth_frames": ok, "of": len(frames["frames"])}


def export_sequence(
    seq_dir: str,
    out_dir: str,
    fps: float = 10.0,
    size: int = 704,
    focal: float = 350.0,
    with_depth: bool = True,
    with_seg: bool = True,
    skeleton: Optional[bool] = None,
    limit: Optional[int] = None,
    quality: int = 92,
) -> dict:
    paths_provider = adt.AriaDigitalTwinDataPathsProvider(seq_dir)
    if skeleton is None:
        # 사람이 있는 세션이면 스켈레톤 판 세그멘테이션을 쓴다 — Khronos 가 다루는
        # 단기 동역학의 주인공이 사람이므로 마스크에 사람이 있어야 한다.
        skeleton = paths_provider.get_num_skeletons() > 0
    paths = paths_provider.get_datapaths(skeleton)
    if paths is None:
        raise RuntimeError("get_datapaths 실패: %s" % seq_dir)
    provider = adt.AriaDigitalTwinDataProvider(paths)

    stream_id = StreamId(RGB_STREAM)
    src_calib = provider.get_aria_camera_calibration(stream_id)
    if src_calib is None:
        raise RuntimeError("RGB 캘리브 없음 (stream %s)" % RGB_STREAM)
    rig = make_rig(src_calib, size=size, focal=focal)

    ts_all = provider.get_aria_device_capture_timestamps_ns(stream_id)
    t_lo, t_hi = provider.get_start_time_ns(), provider.get_end_time_ns()
    picks = _uniform_indices(ts_all, fps, t_lo, t_hi)
    n_dropped = int(np.sum(np.asarray(ts_all, dtype=np.int64) < t_lo))
    if limit:
        picks = picks[:limit]

    want_depth = with_depth and provider.has_depth_images()
    want_seg = with_seg and provider.has_segmentation_images()

    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pose"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)          # P1 이 채운다
    if want_depth:
        os.makedirs(os.path.join(out_dir, "gt", "depth"), exist_ok=True)
    if want_seg:
        os.makedirs(os.path.join(out_dir, "gt", "seg"), exist_ok=True)

    pose_lines, frames = [], []
    seg_lut: dict = {}                 # ADT instance id → 로컬 uint16 id
    skipped = {"no_pose": 0, "no_image": 0, "no_depth": 0, "no_seg": 0}
    kept = 0

    for t_ns in (int(t) for t in picks):
        img = provider.get_aria_image_by_timestamp_ns(t_ns, stream_id)
        if not img.is_valid():
            skipped["no_image"] += 1
            continue
        pose = provider.get_aria_3d_pose_by_timestamp_ns(t_ns)
        if not pose.is_valid():
            skipped["no_pose"] += 1
            continue

        T_world_device = pose.data().transform_scene_device.to_matrix()
        T_world_camera = T_world_device @ rig.T_device_camera

        i = kept
        Image.fromarray(rig.rgb(img.data().to_numpy_array())).save(
            os.path.join(out_dir, "rgb", "%06d.jpg" % i), quality=quality
        )
        pose_lines.append(" ".join("%.9g" % v for v in T_world_camera.reshape(-1)))

        if want_depth:
            d = provider.get_depth_image_by_timestamp_ns(t_ns, stream_id)
            if d.is_valid():
                Image.fromarray(rig.depth(d.data().to_numpy_array())).save(
                    os.path.join(out_dir, "gt", "depth", "%06d.png" % i)
                )
            else:
                skipped["no_depth"] += 1

        if want_seg:
            s = provider.get_segmentation_image_by_timestamp_ns(t_ns, stream_id)
            if s.is_valid():
                lab = rig.labels(s.data().to_numpy_array())
                local = np.zeros(lab.shape, dtype=np.uint16)
                for iid in np.unique(lab):
                    if iid == 0:
                        continue
                    key = int(iid)
                    if key not in seg_lut:
                        seg_lut[key] = len(seg_lut) + 1     # 0 = background
                    local[lab == iid] = seg_lut[key]
                Image.fromarray(local).save(os.path.join(out_dir, "gt", "seg", "%06d.png" % i))
            else:
                skipped["no_seg"] += 1

        frames.append({
            "index": i,
            "t_ns": t_ns,
            "pose_quality": float(pose.data().quality_score),
        })
        kept += 1

    with open(os.path.join(out_dir, "pose", "poses.txt"), "w") as f:
        f.write("\n".join(pose_lines) + "\n")
    with open(os.path.join(out_dir, "camera_info.json"), "w") as f:
        json.dump(camera_info(rig, fps), f, indent=1)
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump({"fps": fps, "frames": frames}, f)

    if want_seg:
        ids = {}
        for iid, local in sorted(seg_lut.items(), key=lambda kv: kv[1]):
            try:
                info = provider.get_instance_info_by_id(iid)
                ids[str(local)] = {
                    "instance_id": iid,
                    "name": info.name,
                    "category": info.category,
                    "motion_type": str(info.motion_type).rsplit(".", 1)[-1],
                    "instance_type": str(info.instance_type).rsplit(".", 1)[-1],
                }
            except Exception:                       # 스켈레톤 등 물체 테이블에 없는 id
                ids[str(local)] = {"instance_id": iid, "name": "?", "category": "?",
                                   "motion_type": "?", "instance_type": "?"}
        with open(os.path.join(out_dir, "gt", "seg_ids.json"), "w") as f:
            json.dump(ids, f, indent=1)

    stats = {
        "sequence": os.path.basename(seq_dir.rstrip("/")),
        "seq_dir": seq_dir,
        "frames": kept,
        "requested": int(len(picks)),
        "skipped": skipped,
        "vrs_frames_before_gt": n_dropped,      # GT 시작 전 잘라낸 RGB 프레임 수
        "gt_window_ns": [int(t_lo), int(t_hi)],
        "fps": fps,
        "size": size,
        "focal": focal,
        "hfov_deg": rig.hfov_deg(),
        "skeleton_variant": bool(skeleton),
        "has_depth_gt": bool(want_depth),
        "has_seg_gt": bool(want_seg),
        "n_seg_instances": len(seg_lut),
        "duration_s": (frames[-1]["t_ns"] - frames[0]["t_ns"]) / 1e9 if frames else 0.0,
        "mps_semidense": os.path.join(seq_dir, "mps", "slam", "semidense_points.csv.gz"),
    }
    with open(os.path.join(out_dir, "export.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats
