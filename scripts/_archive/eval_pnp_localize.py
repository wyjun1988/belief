#!/usr/bin/env python3
"""물체 랜드마크로 **매 프레임 독립 측위** — PnP. 드리프트가 원리적으로 안 쌓인다.

    $P scripts/eval_pnp_localize.py --seq <name> --landmarks graph_gtdepth.json

부트스트랩이 정적 물체의 3D 위치를 준다. 운영 때는 그 물체들의 **2D 관측**만 있으면
알려진 3D ↔ 관측된 2D 대응으로 카메라 6DoF 를 푼다(PnP + RANSAC).

v1 의 절벽은 포즈였고(위치오차 0.066 → 1.482 m), 원인은 **윈도우를 이어붙이며 오차가
누적**되는 것이었다. PnP 는 매 프레임을 지도에 **독립적으로** 맞추므로 그 누적이 없다.

⚠️ 관측점을 무엇으로 잡느냐가 정확도를 좌우한다. 물체의 3D 중심을 투영한 점과
**마스크 중심**은 다르다 — 가려지거나 잘리면 마스크 중심이 물체 중심에서 밀린다.
그래서 (a) 잘린 물체 제외, (b) 큰 물체 제외(중심 편의가 크기에 비례), (c) RANSAC
으로 남은 이상치를 거른다.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_PX = 300          # 이보다 작은 마스크는 중심이 불안정
BORDER = 4            # 이미지 가장자리에 닿으면 잘린 것 — 중심이 밀린다
MAX_EXTENT = 2.5      # m. 큰 물체는 '보이는 면 중심'이 3D 중심에서 멀다


def observations(seg, ids, meta3d, W, H):
    """마스크 → [(3D 점, 2D 점, local_id)] — 잘리거나 너무 크거나 작은 것은 뺀다."""
    out = []
    u, c = np.unique(seg, return_counts=True)
    for a, n in zip(u, c):
        a = int(a)
        if a == 0 or n < MIN_PX:
            continue
        rec = meta3d.get(a)
        if rec is None:
            continue
        p3, ext = rec
        if ext is not None and max(ext) > MAX_EXTENT:
            continue
        ys, xs = np.nonzero(seg == a)
        if xs.min() < BORDER or ys.min() < BORDER or xs.max() > W - 1 - BORDER \
                or ys.max() > H - 1 - BORDER:
            continue                                  # 잘린 물체
        out.append((p3, np.array([xs.mean(), ys.mean()], float), a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--landmarks", default="graph_gtdepth.json",
                    help="랜드마크 3D 위치 출처. GT 대신 v1 그래프를 쓸 수 있다")
    ap.add_argument("--seg", default="gt/seg")
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--min-pts", type=int, default=4)
    ap.add_argument("--reproj", type=float, default=25.0, help="RANSAC 재투영 허용(px)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(sd, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    W, H = cam["width"], cam["height"]
    poses = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    ids = json.load(open(os.path.join(sd, args.seg_ids)))
    g = json.load(open(os.path.join(sd, args.landmarks)))
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]

    # 랜드마크: 정적이고 배치가 하나뿐인 노드 (부트스트랩 산출물에 해당)
    meta3d = {}
    for local, m in ids.items():
        gi = str(m.get("gt_instance") or m.get("instance_id"))
        rec = gt.get(gi)
        if rec is None or rec["motion_type"] != "static" or rec.get("moves"):
            continue
        o = g["objects"].get(str(m.get("instance_id"))) or g["objects"].get(gi)
        if not o or not o.get("placements"):
            continue
        meta3d[int(local)] = (np.array(o["placements"][0]["position"], float),
                              o.get("extent_m"))
    print("랜드마크 %d개 (출처 %s)" % (len(meta3d), args.landmarks))

    rows = []
    for i in range(0, len(poses), args.every):
        sp = os.path.join(sd, args.seg, "%06d.png" % i)
        if not os.path.exists(sp):
            continue
        obs = observations(np.array(Image.open(sp)), ids, meta3d, W, H)
        if len(obs) < args.min_pts:
            rows.append(dict(frame=i, n=len(obs), ok=False))
            continue
        P3 = np.array([o[0] for o in obs], np.float64)
        P2 = np.array([o[1] for o in obs], np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            P3, P2, K, None, flags=cv2.SOLVEPNP_SQPNP,
            reprojectionError=args.reproj, iterationsCount=500, confidence=0.999)
        if not ok or inl is None or len(inl) < args.min_pts:
            rows.append(dict(frame=i, n=len(obs), ok=False))
            continue
        R, _ = cv2.Rodrigues(rvec)
        # solvePnP 는 world→camera. 카메라 중심 = -R^T t
        C = (-R.T @ tvec).ravel()
        T = poses[i]
        e_pos = float(np.linalg.norm(C - T[:3, 3]))
        Rg = T[:3, :3].T                              # world→camera (GT)
        cosang = (np.trace(R @ Rg.T) - 1) / 2
        e_rot = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        rows.append(dict(frame=i, n=len(obs), ok=True, n_inl=int(len(inl)),
                         err_pos=e_pos, err_rot=e_rot))

    good = [r for r in rows if r.get("ok")]
    cov = len(good) / max(len(rows), 1)
    print("프레임 %d · **측위 성공 %.1f%%** · 관측점 중앙 %.0f개 · 인라이어 중앙 %.0f개"
          % (len(rows), 100 * cov,
             np.median([r["n"] for r in rows]),
             np.median([r["n_inl"] for r in good]) if good else 0))
    if good:
        ep = np.array([r["err_pos"] for r in good])
        er = np.array([r["err_rot"] for r in good])
        print("  위치오차  중앙 **%.3f m** · p90 %.3f · <0.5m %.2f · <1m %.2f"
              % (np.median(ep), np.percentile(ep, 90), (ep < 0.5).mean(), (ep < 1).mean()))
        print("  자세오차  중앙 **%.2f°** · p90 %.2f · <10° %.2f"
              % (np.median(er), np.percentile(er, 90), (er < 10).mean()))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
