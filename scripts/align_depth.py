#!/usr/bin/env python3
"""DA3 원시 뎁스 → 전역 정합된 미터 뎁스.

    $P scripts/align_depth.py --seq <name> --mode t23 --out depth
    $P scripts/align_depth.py --seq <name> --mode t2  --out depth_t2    # 애블레이션

모드:
    none  정합 없음 — DA3 출력을 그대로 (윈도우 경계에서 스케일이 튄다)
    t2    프레임별 앵커 affine (전역 절대 정렬)
    t23   t2 + 시간축 2차 차분 정칙화 (앵커 빈곤 프레임 보간 포함)

동적 물체 픽셀의 앵커는 버린다. 앵커는 정적 지도에서 온 것이라, 사람이 그 앞을
지나가는 순간 "벽까지 거리"와 "사람까지 거리"를 같은 픽셀에서 비교하게 된다.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.align import apply_affine, fit_frame, gate, smooth_sequence   # noqa: E402
from kx.depth.anchors import AnchorProjector, load_semidense, scene_bbox   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dynamic_mask(seq_dir, i, dyn_locals, dilate=9):
    p = os.path.join(seq_dir, "gt", "seg", "%06d.png" % i)
    if not os.path.exists(p) or not dyn_locals:
        return None
    seg = np.array(Image.open(p))
    m = np.isin(seg, dyn_locals)
    if not m.any():
        return None
    return cv2.dilate(m.astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--raw", default="depth_raw")
    ap.add_argument("--out", default="depth")
    ap.add_argument("--mode", default="t23", choices=["none", "t2", "t23"])
    ap.add_argument("--lam", type=float, default=1.0, help="T3 평활 강도")
    ap.add_argument("--no-dyn-mask", action="store_true")
    ap.add_argument("--pose", default="pose/poses.txt",
                    help="앵커 투영에 쓸 포즈. 모델 포즈로 정합하려면 여기를 바꾼다 —"
                         " 앵커는 이 포즈로 프레임에 투영되므로 포즈가 곧 정합 품질이다")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    poses = np.loadtxt(os.path.join(seq_dir, args.pose)).reshape(-1, 4, 4)
    K = np.array(cam["intrinsics"])
    W, H = cam["width"], cam["height"]
    raw_dir = os.path.join(seq_dir, args.raw)
    out_dir = os.path.join(seq_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)
    # 스모크 실행에서는 앞 N 프레임만 추론해 두므로, 있는 것만 처리한다.
    have = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(raw_dir) if f.endswith(".npy"))
    n = (have[-1] + 1) if have else 0
    if not have:
        sys.exit("원시 뎁스가 없다: %s" % raw_dir)

    if args.mode == "none":
        for i in have:
            d = np.load(os.path.join(raw_dir, "%06d.npy" % i)).astype(np.float32)
            _write(out_dir, i, d)
        json.dump({"mode": "none"}, open(os.path.join(seq_dir, args.out + "_align.json"), "w"))
        print("wrote %d frames (no alignment)" % len(have))
        return

    # 동적 인스턴스 로컬 id
    dyn_locals = []
    ids_p = os.path.join(seq_dir, "gt", "seg_ids.json")
    if os.path.exists(ids_p) and not args.no_dyn_mask:
        ids = json.load(open(ids_p))
        dyn_locals = [int(k) for k, v in ids.items()
                      if v.get("motion_type") != "STATIC"]   # DYNAMIC + 스켈레톤(?) 전부

    stats = json.load(open(os.path.join(seq_dir, "export.json")))
    xyz, w = load_semidense(stats["mps_semidense"], bbox=scene_bbox(poses))
    proj = AnchorProjector(xyz, w, K, W, H)
    print("anchors: %d points, dynamic instances masked: %d" % (len(xyz), len(dyn_locals)))

    rng = np.random.default_rng(0)
    A, B, NI, RM = np.zeros(n), np.zeros(n), np.zeros(n, int), np.full(n, np.nan)
    IR = np.zeros(n)
    for i in have:
        d = np.load(os.path.join(raw_dir, "%06d.npy" % i)).astype(np.float32)
        u, v, z, ww = proj.frame(poses[i])
        if len(z) == 0:
            continue
        dm = dynamic_mask(seq_dir, i, dyn_locals)
        if dm is not None:
            keep = ~dm[v, u]
            u, v, z, ww = u[keep], v[keep], z[keep], ww[keep]
        r = fit_frame(d, u, v, z, ww, rng)
        if r is None:
            continue
        A[i], B[i], NI[i], RM[i], IR[i] = r
        if i % 100 == 0:
            print("  frame %4d  a=%.4f b=%+.4f  inliers=%4d  rmse_inv=%.4f"
                  % (i, A[i], B[i], NI[i], RM[i]), flush=True)

    raw_ok = NI > 0
    ok = gate(A, B, NI, RM, IR)
    print("fit ok %d/%d, 게이트 통과 %d (파탄 %d프레임은 시간 보간)"
          % (raw_ok.sum(), n, ok.sum(), int(raw_ok.sum() - ok.sum())))
    NI = np.where(ok, NI, 0)
    if args.mode == "t23":
        A, B, _ = smooth_sequence(A, B, NI, lam=args.lam)
    else:
        # t2 단독: 실패한 프레임은 이웃에서 선형 보간 (그래야 뎁스가 비지 않는다)
        idx = np.arange(n)
        A = np.interp(idx, idx[ok], A[ok])
        B = np.interp(idx, idx[ok], B[ok])

    for i in have:
        d = np.load(os.path.join(raw_dir, "%06d.npy" % i)).astype(np.float32)
        _write(out_dir, i, apply_affine(d, A[i], B[i]))

    scale_drift = float(np.std(A[ok] / max(np.median(A[ok]), 1e-9)))
    
    doc = {"mode": args.mode, "lam": args.lam, "frames": n,
           "fit_frames": int(ok.sum()), "median_inliers": float(np.median(NI[ok])),
           "median_rmse_inv": float(np.nanmedian(RM)),
           "gated_out": int(raw_ok.sum() - ok.sum()),
           "scale_rel_std": scale_drift,
           "a": np.round(A, 6).tolist(), "b": np.round(B, 6).tolist(),
           "n_inlier": NI.tolist()}
    json.dump(doc, open(os.path.join(seq_dir, args.out + "_align.json"), "w"))
    print("mode=%s  fit %d/%d  median inliers %.0f  scale rel-std %.4f  → %s"
          % (args.mode, ok.sum(), n, np.median(NI[ok]), scale_drift, out_dir))


def _write(out_dir, i, depth_m):
    mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(os.path.join(out_dir, "%06d.png" % i))


if __name__ == "__main__":
    main()
