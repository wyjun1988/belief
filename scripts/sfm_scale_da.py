#!/usr/bin/env python3
"""SfM 척도를 GT 없이 — 단안 메트릭 깊이(DA-V2 metric)로. 재구성의 3D 점을 프레임에 투영한 SfM 깊이 z 와 같은 픽셀의
DA 깊이 비율 중앙값 = 집의 척도. sim3(GT 맵포즈) 척도와 대조해 "GT 척도 ÷ DA 척도" 가 집마다 일정한지 본다
(일정하면 데이터셋 상수 1개 = 카메라 보정 수준이지 집별 GT 가 아니다).

    python scripts/sfm_scale_da.py data/hssd20S2 house_0000 house_0001 ... [--n 40]
"""
import argparse, json, os, sys
import numpy as np, torch
import pycolmap
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("houses", nargs="+")
ap.add_argument("--n", type=int, default=40); ap.add_argument("--sfm", default=os.path.expanduser("~/khcache/sfm"))
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoImageProcessor.from_pretrained(a.model)
mdl = AutoModelForDepthEstimation.from_pretrained(a.model, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
print("모델 %s · %s" % (a.model.split("/")[-1], DEV), flush=True)

def da_depth(img):
    inp = proc(images=img, return_tensors="pt").to(DEV)
    with torch.no_grad(): d = mdl(**inp).predicted_depth
    d = torch.nn.functional.interpolate(d[None], size=img.size[::-1], mode="bicubic", align_corners=False)[0, 0]
    return d.float().cpu().numpy()

rows = []
for hn in a.houses:
    d = os.path.join(a.sfm, hn, "rec_all")
    subs = [s for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
    rec = max((pycolmap.Reconstruction(os.path.join(d, s)) for s in subs), key=lambda r: r.num_reg_images())
    ims = sorted((im for im in rec.images.values() if im.has_pose and im.name.startswith("live/")), key=lambda im: im.name)
    step = max(1, len(ims) // a.n); ims = ims[::step][:a.n]
    fr = []
    for im in ims:
        p2 = [p for p in im.points2D if p.has_point3D()]
        if len(p2) < 8: continue
        X = np.array([rec.points3D[p.point3D_id].xyz for p in p2]); xy = np.array([p.xy for p in p2])
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        z = (cfw * X)[:, 2]
        img = Image.open(os.path.join(a.root, hn, im.name)).convert("RGB"); D = da_depth(img)
        u = np.clip(xy[:, 0].round().astype(int), 0, D.shape[1] - 1); v = np.clip(xy[:, 1].round().astype(int), 0, D.shape[0] - 1)
        ok = (z > 0.3) & (D[v, u] > 0.3)
        if ok.sum() >= 8: fr.append(float(np.median(D[v, u][ok] / z[ok])))
    s_da = float(np.median(fr)) if fr else float("nan")
    sm = json.load(open(os.path.join(a.sfm, hn, "summary_%s.json" % hn)))
    s_gt = sm["scale"]
    rows.append((hn, len(fr), s_da, s_gt, s_gt / s_da if fr else float("nan"), float(np.std(fr) / np.median(fr)) if fr else float("nan")))
    print("%s  프레임 %d · DA 척도 %.3f · GT(sim3) 척도 %.3f · GT/DA %.3f · 프레임간 산포 %.2f" % rows[-1], flush=True)
k = np.array([r[4] for r in rows if np.isfinite(r[4])])
if len(k):
    kc = float(np.median(k)); print("GT/DA 비율: 중앙 %.3f · 집별 편차 %s" % (kc, ["%+.0f%%" % (100 * (x / kc - 1)) for x in k]))
    print("→ 상수 %.3f 하나로 척도를 잡으면 집별 척도 오차 = 위 편차 (방 수준 위치엔 ±10%% 면 충분)" % kc)
