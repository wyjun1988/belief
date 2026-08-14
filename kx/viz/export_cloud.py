"""GT 뎁스 + GT 포즈 + RGB → 색깔 있는 정적 점구름 (웹 뷰어용 바이너리).

    $P kx/viz/export_cloud.py --seq <name> --every 5 --voxel 0.035

동적 물체는 **뺀다** — 착용자가 들고 다니는 물건은 누적하면 궤적을 따라 번져서
배경 재구성을 더럽힌다. 그 물체들은 씬그래프 노드로 따로 그린다.

출력: `cloud.bin` (pos **uint16** ×3 + rgb uint8 ×3 = 9 B/점) + `cloud.json` (양자화 메타).
16비트 양자화는 3.5cm 복셀 대비 충분히 촘촘하다(집 크기 ~15m → 0.23mm 해상도).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--depth", default="gt/depth")
    ap.add_argument("--pose", default="pose/poses.txt")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2, help="픽셀 서브샘플")
    ap.add_argument("--voxel", type=float, default=0.035)
    ap.add_argument("--zmax", type=float, default=6.0)
    ap.add_argument("--max-points", type=int, default=520000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    out_dir = args.out or os.path.join(seq_dir, "cloud")
    os.makedirs(out_dir, exist_ok=True)
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    poses = np.loadtxt(os.path.join(seq_dir, args.pose)).reshape(-1, 4, 4)
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    # seg PNG 는 **로컬 라벨**이라 GT 인스턴스 id 로 바로 못 거른다 — seg_ids 로 사상한다.
    ids = json.load(open(os.path.join(seq_dir, args.seg_ids)))
    dyn = {k for k, v in gt.items() if v.get("motion_type", "").lower() == "dynamic"}
    drop = [int(local) for local, m in ids.items() if str(m.get("instance_id")) in dyn]
    print("동적 인스턴스 %d개 제외 (로컬 라벨 기준)" % len(drop))

    H = W = cam["height"]
    vv, uu = np.mgrid[0:H:args.stride, 0:W:args.stride]
    uu, vv = uu.ravel(), vv.ravel()
    xn = (uu - K[0, 2]) / K[0, 0]
    yn = (vv - K[1, 2]) / K[1, 1]

    acc = {}                                    # voxel key -> [r,g,b,n]
    for i in range(0, len(poses), args.every):
        dp = os.path.join(seq_dir, args.depth, "%06d.png" % i)
        rp = os.path.join(seq_dir, "rgb", "%06d.jpg" % i)
        sp = os.path.join(seq_dir, "gt", "seg", "%06d.png" % i)
        if not (os.path.exists(dp) and os.path.exists(rp)):
            continue
        d = np.array(Image.open(dp)).astype(np.float32)[vv, uu] / 1000.0
        rgb = np.asarray(Image.open(rp).convert("RGB"))[vv, uu]
        ok = (d > 0.25) & (d < args.zmax)
        if os.path.exists(sp):
            seg = np.array(Image.open(sp))[vv, uu]
            ok &= ~np.isin(seg, list(drop))
        if ok.sum() == 0:
            continue
        d, c = d[ok], rgb[ok]
        P = np.stack([xn[ok] * d, yn[ok] * d, d], 1) @ poses[i][:3, :3].T + poses[i][:3, 3]
        key = np.floor(P / args.voxel).astype(np.int64)
        key = (key[:, 0] + 8192) * 16384 * 16384 + (key[:, 1] + 8192) * 16384 + (key[:, 2] + 8192)
        u, inv = np.unique(key, return_inverse=True)
        n = np.bincount(inv)
        s = np.stack([np.bincount(inv, c[:, j].astype(np.float64)) for j in range(3)], 1)
        for j, k in enumerate(u):
            a = acc.get(k)
            if a is None:
                acc[k] = [s[j, 0], s[j, 1], s[j, 2], n[j]]
            else:
                a[0] += s[j, 0]; a[1] += s[j, 1]; a[2] += s[j, 2]; a[3] += n[j]
        if (i // args.every) % 20 == 0:
            print("  f%-5d 복셀 %d" % (i, len(acc)))

    keys = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
    vals = np.array([acc[k] for k in keys], float)
    if len(keys) > args.max_points:                 # 많이 본 복셀부터 남긴다
        sel = np.argsort(-vals[:, 3])[:args.max_points]
        keys, vals = keys[sel], vals[sel]
    kz = keys % 16384 - 8192
    ky = (keys // 16384) % 16384 - 8192
    kx = (keys // (16384 * 16384)) % 16384 - 8192
    P = (np.stack([kx, ky, kz], 1) + 0.5) * args.voxel
    C = np.clip(vals[:, :3] / vals[:, 3:4], 0, 255).astype(np.uint8)

    # ⚠️ uint16 로 저장한다. 처음에 int16 에 0~65000 을 넣었더니 절반이 음수로 감겨
    # 씬의 먼 쪽이 통째로 엉뚱한 자리에 복제됐다(2026-08-14).
    lo, hi = P.min(0), P.max(0)
    scale = max((hi - lo).max(), 1e-6) / 65000.0
    Q = np.clip(np.round((P - lo) / scale), 0, 65535).astype(np.uint16)
    buf = np.empty((len(Q), 9), np.uint8)
    buf[:, :6] = Q.view(np.uint8).reshape(-1, 6)
    buf[:, 6:] = C
    buf.tofile(os.path.join(out_dir, "cloud.bin"))
    json.dump({"n": int(len(Q)), "lo": lo.tolist(), "scale": float(scale),
               "voxel": args.voxel, "depth": args.depth, "every": args.every},
              open(os.path.join(out_dir, "cloud.json"), "w"), indent=1)
    print("점 %d개 → %s (%.1f MB)" % (len(Q), out_dir, len(Q) * 9 / 1e6))


if __name__ == "__main__":
    main()
