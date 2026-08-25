#!/usr/bin/env python3
"""**앵커를 개체(instance)로 식별한다.** 정적 물체마다 exemplar 를 만들어 질의로 쓴다.

    THOR_ROOT=data/thor3 python scripts/exp_anchor_exemplar.py

⚠️ **왜 필요한가.** 지금 앵커는 글자로만 찾는다("Chair"). 그런데 한 집에 Chair 가
여럿이라 **어느 의자인지 모른다.** 그래서 세 가지가 연달아 막혔다:

  · 앵커 배치를 증거로 쓰기 — 타입만 알면 3D 좌표를 못 고른다 (0.644 → 0.613)
  · 2차 검색으로 프레임 늘리기 — 자리 서명이 뭉툭해 딴 자리가 딸려온다 (0.625 → 0.522)
  · A→B→C 분해에서 개체 모호성 몫이 +0.051

exemplar 는 **재생성 없이** 만들 수 있다. live 프레임의 `anch` 에 정적 물체별
화면 위치가 기록돼 있으므로, 그 위치의 패치 임베딩을 뽑으면 개체별 질의가 된다.

⚠️ 시그모이드를 쓰지 않는다 — `class_head` 의 shift/scale 은 텍스트 스케일에 맞춰
학습돼 exemplar 에서는 포화한다. 정규화 내적을 그대로 쓴다(§68).
"""
import glob, json, os, sys
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
OUT = os.environ.get("ACACHE_PREFIX", "/tmp/ax_")
STRIDE = int(os.environ.get("STRIDE", "8"))
MAXA = int(os.environ.get("MAXA", "60"))        # 집당 앵커 개체 상한
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available() else "cpu")
CK = "google/owlv2-base-patch16-ensemble"
pr = Owlv2Processor.from_pretrained(CK)
md = Owlv2ForObjectDetection.from_pretrained(CK).to(DEV).eval()
print("앵커 exemplar · %s · stride %d" % (DEV, STRIDE), flush=True)


def feats(ims):
    pv = pr(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        fm = md.image_embedder(pixel_values=pv)[0]
        b, ph, pw, hd = fm.shape
        ce = md.class_head.dense0(fm.reshape(b, ph * pw, hd))
        ce = ce / (torch.linalg.norm(ce, dim=-1, keepdim=True) + 1e-6)
    return ce, ph, pw


for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    out = OUT + hn + ".npz"
    if os.path.exists(out):
        continue
    g = json.load(open(hd + "/gt.json"))
    sm = g.get("scene_meta")
    if not sm:
        print("  %s scene_meta 없음 — 건너뜀" % hn, flush=True); continue
    lv = sorted(glob.glob(hd + "/live/*.jpg"))[::STRIDE]
    ts = [int(os.path.basename(p)[:-4]) for p in lv]
    live = {m["t"]: m for m in g["live"]}
    # ── 앵커마다 **가장 잘 보인 프레임**을 고른다. 화면 중앙에 가까울수록 잘림이 적다 ──
    best = {}
    for k, t in enumerate(ts):
        for a, c in (live[t].get("anch") or {}).items():
            if not c or a not in sm["static"]:
                continue
            q = -abs(c[0] - 192) - abs(c[1] - 192)      # 384px 중앙 기준
            if q > best.get(a, (-1e9,))[0]:
                best[a] = (q, k, c)
    anch = sorted(best, key=lambda a: -best[a][0])[:MAXA]
    if not anch:
        print("  %s 앵커 없음" % hn, flush=True); continue
    # ── exemplar 추출: 프레임별로 묶어 한 번씩만 통과 ──
    byfr = {}
    for a in anch:
        byfr.setdefault(best[a][1], []).append(a)
    QE = {}
    for k, alist in byfr.items():
        ce, ph, pw = feats([Image.open(lv[k]).convert("RGB")])
        for a in alist:
            c = best[a][2]
            cy = min(max(int(c[1] / 384 * ph), 0), ph - 1)
            cx = min(max(int(c[0] / 384 * pw), 0), pw - 1)
            QE[a] = ce[0, cy * pw + cx]
    keys = list(QE)
    Q = torch.stack([QE[a] for a in keys])
    Q = Q / (torch.linalg.norm(Q, dim=-1, keepdim=True) + 1e-6)
    # ── 전 프레임에 대해 앵커 개체 점수 + 화면 위치 ──
    SC, PP = [], []
    for i in range(0, len(lv), 4):
        ce, ph, pw = feats([Image.open(p).convert("RGB") for p in lv[i:i + 4]])
        with torch.no_grad():
            sim = ce @ Q.transpose(0, 1)              # (batch, 패치, 앵커)
        SC.append(sim.amax(1).float().cpu().numpy())
        PP.append(sim.argmax(1).int().cpu().numpy())
        if i % 160 == 0:
            print("  %s %d/%d" % (hn, i, len(lv)), flush=True)
    np.savez_compressed(out, s=np.concatenate(SC), p=np.concatenate(PP),
                        ph=ph, pw=pw, ts=np.array(ts), anch=np.array(keys, object))
    print("  %s 완료 · 앵커 개체 %d · 프레임 %d" % (hn, len(keys), len(lv)), flush=True)
print("완료")
