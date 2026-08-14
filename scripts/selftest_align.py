#!/usr/bin/env python3
"""정합 기계의 자체 검증 — DA3 없이, GT 뎁스만으로.

두 가지를 확인한다.

  A. **앵커 정합성**: MPS 반-조밀 앵커를 프레임에 투영한 거리 z 가 그 픽셀의 GT 뎁스와
     맞는가. 여기서 어긋나면 프레임 관례·투영·포인트 필터 중 하나가 틀린 것이고,
     그 위에 올릴 T2/T3 는 전부 무의미하다.

  B. **왕복 복원**: GT 뎁스에 윈도우별 임의 affine 을 씌워 "DA3 의 윈도우 드리프트"를
     흉내 낸 뒤, T2/T3 가 원본을 되찾는지 본다. DA3 를 기다리지 않고 정합 코드를
     검증할 수 있고, 복원 오차의 하한도 여기서 나온다.

사용:  $P scripts/selftest_align.py --seq <name> [--frames 60]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.align import apply_affine, fit_frame, smooth_sequence   # noqa: E402
from kx.depth.anchors import AnchorProjector, load_semidense, scene_bbox   # noqa: E402
from kx.depth.metrics import depth_errors   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 2.0, 5.0, 20.0])
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"])
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    stats = json.load(open(os.path.join(seq_dir, "export.json")))
    gt_dir = os.path.join(seq_dir, "gt", "depth")

    # 연속 구간을 쓴다. 띄엄띄엄 뽑으면 프레임마다 다른 윈도우가 돼서 T3(시간 정칙화)를
    # 평가할 수 없다 — 실제로는 stride 만큼 이어지는 프레임이 같은 윈도우를 공유한다.
    avail = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(gt_dir))
    start = max((len(avail) - args.frames) // 2, 0)
    idx = avail[start:start + args.frames]
    print("frames: %d 연속 (%d..%d, 전체 %d)" % (len(idx), idx[0], idx[-1], len(avail)))

    xyz, w = load_semidense(stats["mps_semidense"], bbox=scene_bbox(poses))
    proj = AnchorProjector(xyz, w, K, cam["width"], cam["height"])
    print("anchors: %d points" % len(xyz))

    # --- A. 앵커 vs GT 뎁스 --------------------------------------------------
    rel, counts = [], []
    per_frame = []
    for i in idx:
        gt = np.array(Image.open(os.path.join(gt_dir, "%06d.png" % i))).astype(np.float32) / 1000.0
        u, v, z, ww = proj.frame(poses[i])
        if len(z) == 0:
            continue
        g = gt[v, u]
        ok = (g > 0.2) & (g < 10) & (z > 0.2) & (z < 10)
        if ok.sum() < 30:
            continue
        r = np.abs(z[ok] - g[ok]) / g[ok]
        rel.append(np.median(r))
        counts.append(int(ok.sum()))
        per_frame.append((i, gt, u, v, z, ww))
    rel = np.array(rel)
    print("\nA. anchor vs GT depth   frames=%d  anchors/frame median=%d" % (len(rel), int(np.median(counts))))
    print("   median rel err: %.4f   p90: %.4f   (frame-median 분포)" % (np.median(rel), np.percentile(rel, 90)))
    a_ok = np.median(rel) < 0.05
    print("   → %s" % ("OK (앵커가 실제 표면 위에 있다)" if a_ok
                       else "FAIL — 프레임 관례·필터 확인 필요"))

    # --- B. 왕복 복원 --------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n = len(per_frame)
    stride = max(args.window // 2, 1)               # da3_runner 기본과 같은 stride
    nwin = max((n + stride - 1) // stride, 1)
    # 윈도우마다 다른 (scale, shift) — DA3 윈도우 드리프트의 대역폭을 흉내
    wa = rng.normal(1.0, 0.08, nwin)
    wb = rng.normal(0.0, 0.02, nwin)
    A_hat, B_hat, NI = np.zeros(n), np.zeros(n), np.zeros(n, int)
    corrupted, truth = [], []
    for k, (i, gt, u, v, z, ww) in enumerate(per_frame):
        wi = min(k // stride, nwin - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(gt > 0, 1.0 / np.maximum(gt, 1e-6), 0.0)
            bad = np.where(gt > 0, 1.0 / np.maximum(wa[wi] * inv + wb[wi], 1e-6), 0.0)
        bad = np.nan_to_num(bad, posinf=0.0, neginf=0.0).astype(np.float32)
        corrupted.append(bad)
        truth.append(gt)
        r = fit_frame(bad, u, v, z, ww, rng)
        if r is not None:
            A_hat[k], B_hat[k], NI[k] = r[0], r[1], r[2]   # fit_frame 은 5-튜플

    print("\nB. round-trip (윈도우 %d개 stride %d, scale σ=0.08 shift σ=0.02 주입)"
          % (nwin, stride))
    print("   fit ok on %d/%d frames" % ((NI > 0).sum(), n))

    base = [depth_errors(corrupted[k], truth[k]) for k in range(n)]
    base = float(np.mean([b["absrel"] for b in base if b]))

    variants = [("T2 only", (A_hat, B_hat))]
    for lam in args.lams:
        A_s, B_s, _ = smooth_sequence(A_hat, B_hat, NI, lam=lam)
        variants.append(("T2+T3 λ=%g" % lam, (A_s, B_s)))

    for label, (AA, BB) in variants:
        errs = []
        for k in range(n):
            if label == "T2 only" and NI[k] == 0:
                continue
            e = depth_errors(apply_affine(corrupted[k], AA[k], BB[k]), truth[k])
            if e:
                errs.append(e["absrel"])
        print("   %-12s AbsRel %.4f   (정합 전 %.4f)" % (label, np.mean(errs), base))

    print()
    if not a_ok:
        sys.exit(1)
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
