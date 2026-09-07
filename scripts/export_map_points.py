#!/usr/bin/env python3
"""스캔 지도(GT 스캔 포즈 삼각측량 rec_gt)의 3D 점을 맵 프레임별 (k, u, v, 깊이) 로 내보낸다 → build_initmap 의 MAP_POINTS(SfM 점 깊이·DA 척도 자가보정) 입력.
    python scripts/export_map_points.py <house_dir> --work ~/khcache/hloc-c4/<house> --out ~/khcache/mappts/<house>
깊이 = 카메라 좌표 z (미러·월드 규약과 무관). 스캔 프레임 rgb/%06d.jpg 의 번호 = 맵 프레임 k (hssd_to_seq_reloc 규약, k < n_map)."""
import argparse, json, os, numpy as np, pycolmap
ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seq", default=None)
a = ap.parse_args(); hn = os.path.basename(os.path.realpath(a.house)); os.makedirs(a.out, exist_ok=True)
seq = a.seq or os.path.join("data/seq", "c4_" + hn); n_map = json.load(open(os.path.join(seq, "camera_info.json")))["n_map"]
rec = pycolmap.Reconstruction(os.path.join(a.work, "rec_gt"))
K, U, V, D = [], [], [], []
for im in rec.images.values():
    k = int(os.path.basename(im.name)[:-4])
    if k >= n_map: continue
    cfw = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
    R = np.asarray(cfw.rotation.matrix()); t = np.asarray(cfw.translation)
    for p in im.points2D:
        if not p.has_point3D(): continue
        X = np.asarray(rec.points3D[p.point3D_id].xyz); z = float((R @ X + t)[2])
        if 0.2 < z < 15: K.append(k); U.append(float(p.xy[0])); V.append(float(p.xy[1])); D.append(z)
np.savez_compressed(os.path.join(a.out, "map_points_%s.npz" % hn), k=np.array(K, np.int32), u=np.array(U, np.float32), v=np.array(V, np.float32), d=np.array(D, np.float32))
print("%s: 맵 프레임 %d/%d 에 점 %d (프레임당 중앙 %d)" % (hn, len(set(K)), n_map, len(K), int(np.median(np.bincount(np.array(K)) [np.bincount(np.array(K)) > 0])) if K else 0))
