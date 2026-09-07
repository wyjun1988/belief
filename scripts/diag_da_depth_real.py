#!/usr/bin/env python3
"""실사(ADT)에서 단안 메트릭 깊이(DA-V2 metric indoor)가 GT 깊이 대비 어떤 척도로 나오는지 — 시뮬 상수 0.468 의 실사 근거 확인.
    python scripts/diag_da_depth_real.py data/seq/Apartment_release_decoration_seq137_M1292 [N]
프레임 N장 균등 표본. 출력: 프레임별 중앙 비율(GT/DA), 전체 중앙·사분위, AbsRel, δ1(1.25). GT 깊이는 uint16 mm."""
import os, sys, glob, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
seq = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
mname = os.environ.get("DA_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
pr = AutoImageProcessor.from_pretrained(mname); md = AutoModelForDepthEstimation.from_pretrained(mname).to(dev).eval()
rgbs = sorted(glob.glob(os.path.join(seq, "rgb", "*.jpg"))); pick = rgbs[::max(1, len(rgbs) // N)][:N]
ratios, absrel, d1 = [], [], []; absrel_s, d1_s = [], []; SCALE = float(os.environ.get("DA_K", "0.468"))   # 시뮬 상수를 그대로 적용해 본다
for p in pick:
    gt = np.array(Image.open(os.path.join(seq, "gt", "depth", os.path.basename(p)[:-4] + ".png"))).astype(np.float32) / 1000.0
    img = Image.open(p).convert("RGB"); inp = pr(images=img, return_tensors="pt").to(dev)
    with torch.no_grad(): D = md(**inp).predicted_depth
    D = torch.nn.functional.interpolate(D[None], size=gt.shape, mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
    ok = (gt > 0.3) & (gt < 8) & (D > 0.1)
    r = gt[ok] / D[ok]; ratios.append(float(np.median(r)))
    absrel.append(float(np.mean(np.abs(D[ok] - gt[ok]) / gt[ok]))); d1.append(float(np.mean(np.maximum(D[ok] / gt[ok], gt[ok] / D[ok]) < 1.25)))
    Ds = D * SCALE; absrel_s.append(float(np.mean(np.abs(Ds[ok] - gt[ok]) / gt[ok]))); d1_s.append(float(np.mean(np.maximum(Ds[ok] / gt[ok], gt[ok] / Ds[ok]) < 1.25)))
ratios = np.array(ratios)
print("%s · %d장 · %s" % (os.path.basename(seq), len(pick), mname.split("/")[-1]))
print("GT/DA 척도비 중앙 %.3f (사분위 %.3f~%.3f · 프레임 표준편차 %.3f) · AbsRel %.3f · δ1 %.3f" % (
    np.median(ratios), np.quantile(ratios, .25), np.quantile(ratios, .75), ratios.std(), np.mean(absrel), np.mean(d1)))
print("시뮬 상수 %.3f 적용 후: AbsRel %.3f · δ1 %.3f" % (SCALE, np.mean(absrel_s), np.mean(d1_s)))
print("→ 실사 비율이 시뮬 상수와 같으면 상수는 렌더 편향이 아니라 모델의 화각 가정(90° HFOV) 체계 오차 — 실사로 전이된다")
