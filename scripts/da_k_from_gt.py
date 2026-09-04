#!/usr/bin/env python3
"""DA-V2 metric 깊이의 데이터셋 보정 상수 K = GT거리 / DA깊이 (§146: HSSD 0.468) — **COLMAP 없이** gt.json 의 매핑 프레임만으로.
매핑 프레임 gt.map[i] 에는 보이는 물체의 box(픽셀)와 dist(카메라→물체 수평거리, 시뮬 GT)가 있다. 박스 중심의 DA 깊이(5×5 중앙값)와
dist 의 비율 중앙값을 낸다. 렌더러가 바뀌면(OG) 1채로 한 번만 돌려 --da-k 에 넣는다. 카메라 보정 수준의 데이터셋 상수이지 집별 GT 가 아니다.

    python scripts/da_k_from_gt.py <house_dir> [--n 40]
"""
import argparse, json, os
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"); ap.add_argument("--min-d", type=float, default=0.8); ap.add_argument("--max-d", type=float, default=8.0)
a = ap.parse_args()
hd = a.house.rstrip("/"); g = json.load(open(os.path.join(hd, "gt.json"))); gm = g["map"]
maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg"))
assert len(maps) == len(gm), "gt.map %d ≠ map 프레임 %d" % (len(gm), len(maps))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoImageProcessor.from_pretrained(a.model); mdl = AutoModelForDepthEstimation.from_pretrained(a.model).to(DEV).eval()
idx = [k for k, m in enumerate(gm) if m.get("box") and m.get("dist")]
idx = idx[:: max(1, len(idx) // a.n)][:a.n]; rat = []; per = []
for k in idx:
    m = gm[k]; img = Image.open(os.path.join(hd, "map", maps[k])).convert("RGB"); W, H = img.size
    with torch.no_grad(): d = mdl(**proc(images=img, return_tensors="pt").to(DEV)).predicted_depth
    D = torch.nn.functional.interpolate(d[None], size=(H, W), mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
    fr = []
    for oid, b in m["box"].items():
        gd = m["dist"].get(oid)
        if gd is None or not (a.min_d <= gd <= a.max_d): continue
        u = int(np.clip((b[0] + b[2]) / 2, 2, W - 3)); v = int(np.clip((b[1] + b[3]) / 2, 2, H - 3))
        z = float(np.median(D[v-2:v+3, u-2:u+3]))
        if z > 0.1: fr.append(gd / z)              # dist 는 수평거리, DA 는 광축 깊이 — 눈높이 물체는 근사 동일
    if fr: per.append(float(np.median(fr))); rat += fr
K = float(np.median(rat)) if rat else float("nan")
print("K = GT/DA 중앙 %.3f · 표본 %d(프레임 %d) · 프레임간 산포 %.2f  (HSSD 기준 0.468)" % (K, len(rat), len(per), float(np.std(per) / (np.median(per) + 1e-9)) if per else 0))
print("→ vggt_reloc/cut3r_reloc --da-k %.3f  ·  sfm_reloc --da-k %.3f" % (K, K))
