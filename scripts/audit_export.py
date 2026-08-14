#!/usr/bin/env python3
"""내보내기 무결성 게이트 — 통과 못하면 그 뒤 단계는 전부 헛일이다.

home-jepa 에서 배운 것: 치명 버그 6건 중 다수가 좌표·시점 관례였고, 전부 감사가 잡았다.
여기서 검사하는 것은 **캘리브·포즈·회전 체인이 실제로 맞물리는가** 하나다.

  A. 개수 일치 (rgb / poses / frames / gt·seg / gt·depth)
  B. 포즈 연속성 — 10Hz 사이 점프
  C. **투영 일치** — GT 3D 중심을 우리 선형 K + T_world_camera 로 투영한 픽셀이
     GT 마스크 무게중심과 맞는가. 언디스토트·cw90 회전·T_device_camera 를 한꺼번에 검증한다.
  D. **뎁스 일치** (GT 뎁스가 있을 때) — 마스크 위 GT 뎁스 중앙값이 카메라→물체 거리와 맞는가.
     z-깊이 관례인지 사거리(range) 관례인지도 여기서 확정된다.

사용:  $P scripts/audit_export.py --seq <name>[_smoke] [--every 10]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "data", "seq")

# 게이트 임계 — 704px 기준. 마스크 무게중심과 3D 중심 투영은 원리적으로 완전히
# 같지 않으므로(비대칭 메시·부분 가림) 중앙값으로 본다.
MAX_MEDIAN_PX = 25.0
MAX_MEDIAN_DEPTH_REL = 0.15
MAX_POSE_JUMP_M = 0.5


def load(seq_dir):
    frames = json.load(open(os.path.join(seq_dir, "frames.json")))
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))
    ids = json.load(open(os.path.join(seq_dir, "gt", "seg_ids.json")))
    return frames, cam, poses, gt, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--every", type=int, default=10, help="C/D 검사 프레임 간격")
    ap.add_argument("--min-area", type=int, default=500)
    ap.add_argument("--max-extent", type=float, default=1.5,
                    help="이보다 큰 물체(벽·바닥)는 중심 투영 검사에서 뺀다")
    args = ap.parse_args()

    seq_dir = os.path.join(SEQ, args.seq) if not os.path.isdir(args.seq) else args.seq
    frames, cam, poses, gt, ids = load(seq_dir)
    K = np.array(cam["intrinsics"])
    W, H = cam["width"], cam["height"]
    fails = []

    # --- A. 개수 -------------------------------------------------------------
    n = len(frames["frames"])
    counts = {"frames": n, "poses": len(poses),
              "rgb": len(os.listdir(os.path.join(seq_dir, "rgb")))}
    for sub in ("seg", "depth"):
        p = os.path.join(seq_dir, "gt", sub)
        if os.path.isdir(p):
            counts["gt/" + sub] = len(os.listdir(p))
    print("A. counts: %s" % counts)
    if len({v for k, v in counts.items() if k != "gt/depth"}) != 1:
        fails.append("개수 불일치: %s" % counts)
    if gt["n_frames"] != n:
        fails.append("gt/objects.json 프레임 수 %d != %d" % (gt["n_frames"], n))

    # --- B. 포즈 연속성 ------------------------------------------------------
    d = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    print("B. pose step  median %.4f m   max %.4f m" % (np.median(d), d.max()))
    if d.max() > MAX_POSE_JUMP_M:
        fails.append("포즈 점프 %.3f m > %.2f" % (d.max(), MAX_POSE_JUMP_M))

    # --- C/D. 투영·뎁스 ------------------------------------------------------
    inst = gt["instances"]
    local2iid = {int(k): str(v["instance_id"]) for k, v in ids.items()}
    have_depth = os.path.isdir(os.path.join(seq_dir, "gt", "depth"))

    px_err, dep_rel_z, dep_rel_r, used = [], [], [], 0
    for i in range(0, n, args.every):
        seg_p = os.path.join(seq_dir, "gt", "seg", "%06d.png" % i)
        if not os.path.exists(seg_p):
            continue
        seg = np.array(Image.open(seg_p))
        dep = None
        if have_depth:
            dp = os.path.join(seq_dir, "gt", "depth", "%06d.png" % i)
            if os.path.exists(dp):
                dep = np.array(Image.open(dp)).astype(np.float64) / 1000.0   # mm → m

        T = poses[i]
        Rwc, twc = T[:3, :3], T[:3, 3]
        for local in np.unique(seg):
            if local == 0:
                continue
            rec = inst.get(local2iid.get(int(local), ""))
            if rec is None or rec["extent_m"] is None:
                continue
            if max(rec["extent_m"]) > args.max_extent:
                continue
            m = seg == local
            if m.sum() < args.min_area:
                continue
            ys, xs = np.nonzero(m)
            if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
                continue                                  # 화면 밖으로 잘린 물체는 제외

            p_cam = Rwc.T @ (np.array(rec["positions"][i]) - twc)
            if p_cam[2] < 0.15:
                continue
            uv = K[:2, :2] @ (p_cam[:2] / p_cam[2]) + K[:2, 2]
            px_err.append(float(np.hypot(uv[0] - xs.mean(), uv[1] - ys.mean())))
            used += 1

            if dep is not None:
                dv = dep[m]
                dv = dv[dv > 0]
                if len(dv) > 20:
                    med = float(np.median(dv))
                    dep_rel_z.append(abs(med - p_cam[2]) / p_cam[2])
                    r = float(np.linalg.norm(p_cam))
                    dep_rel_r.append(abs(med - r) / r)

    if px_err:
        px = np.array(px_err)
        print("C. proj err  n=%d  median %.1f px  p90 %.1f px" % (used, np.median(px), np.percentile(px, 90)))
        if np.median(px) > MAX_MEDIAN_PX:
            fails.append("투영 오차 중앙값 %.1f px > %.0f — 캘리브/회전/포즈 체인 의심"
                         % (np.median(px), MAX_MEDIAN_PX))
    else:
        fails.append("투영 검사 표본 0 — 마스크나 GT 매칭이 깨졌다")

    if dep_rel_z:
        mz, mr = float(np.median(dep_rel_z)), float(np.median(dep_rel_r))
        conv = "z-depth" if mz <= mr else "range"
        print("D. depth rel err  median  z:%.3f  range:%.3f  → %s 관례" % (mz, mr, conv))
        if min(mz, mr) > MAX_MEDIAN_DEPTH_REL:
            fails.append("GT 뎁스 상대오차 %.3f > %.2f" % (min(mz, mr), MAX_MEDIAN_DEPTH_REL))
    elif have_depth:
        print("D. depth: 표본 없음")
    else:
        print("D. depth: GT 뎁스 미다운로드 — 생략")

    print()
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        sys.exit(1)
    print("PASS: export integrity OK (%s)" % args.seq)


if __name__ == "__main__":
    main()
