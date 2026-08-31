#!/usr/bin/env python3
"""기하 투영 사다리 ③ — 거리의 실물화: DA-V2 metric mono-depth @ 타겟 패치. (GPU)

    THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ QC_PREFIX=/tmp/t7_q_ \\
      SCORES=t1_scores_t7ac.jsonl OUT_JSONL=geo_depth_t7.jsonl \\
      python scripts/geo_depth.py

투영이 실제로 만지는 프레임만 골라 depth 를 잰다:
  a) t1_scores 의 걸은 후보 전부(오검출 포함 — §108 의 GT 누출 제거)
  b) 타겟별 GT 목격 최신 8 (georoom 풀)
  c) 타겟별 top50 최신 5 (find_recent 풀)
z-depth → 수평거리: d = z·sqrt(x̂² + (cosθ − ŷ·sinθ)²), θ=카메라 기울기 10°.

**지도 앵커 역깊이 아핀 정합(우리 기술 kx.depth.align 의 THOR 적응판)**:
1차 실측 상대오차 0.91 = 계통 스케일 실패. 바닐라 DA 가 아니라 우리 정합을 쓴다 —
  · 역깊이 공간 1/ẑ = a·(1/z_pred) + b  (곱셈 스케일만으론 near/far 편향 못 잡음)
  · **상대 잔차** RANSAC (절대 잔차는 "먼 점만 맞추는 평평한 퇴화 해" 를 낳는다)
  · a≤0.05 퇴화 가드 → 스케일-only 후퇴
가구 앵커는 프레임당 몇 개뿐이라(원판의 프레임당 30+ 반조밀 앵커와 다름) 쌍을
**채 단위로 모아** 아핀을 풀고, 프레임은 잔차 배율(중앙값, [0.6,1.6] 클립)만 보정.
GT 는 진단 출력 전용 — 정합엔 지도 좌표만 쓴다.
"""
import glob, json, os
import numpy as np
from collections import Counter
from PIL import Image
import torch

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
A3P = os.environ.get("A3_PREFIX", "/tmp/t7_a_")
QCP = os.environ.get("QC_PREFIX", "/tmp/t7_q_")
AXP = os.environ.get("AX_PREFIX", "/tmp/t7_x_")
SC = os.environ.get("SCORES", "t1_scores_t7ac.jsonl")
OUTJ = os.environ.get("OUT_JSONL", "geo_depth_t7.jsonl")
MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
TILT = np.radians(float(os.environ.get("TILT", "10")))
BATCH = int(os.environ.get("BATCH", "8"))

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(MODEL)
mdl = AutoModelForDepthEstimation.from_pretrained(MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
print("모델 %s · %s" % (MODEL, DEV), flush=True)

vsc = {}
if os.path.exists(SC):
    for l in open(SC):
        d = json.loads(l)
        vsc.setdefault(d["house"], []).append(d)
else:
    print("⚠ SCORES 없음(%s) — a) 생략" % SC)

out = open(OUTJ, "w")
errs = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    ts, vocab, nT = za["ts"], list(za["vocab"]), int(za["nT"])
    P, ph, pw = za["p"], int(za["ph"]), int(za["pw"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    sm = g.get("scene_meta") or {}
    stp = {k: v["pos"] for k, v in sm.get("static", {}).items() if v.get("pos")}
    ZX = np.load(AXP + hn + ".npz", allow_pickle=True) if os.path.exists(AXP + hn + ".npz") else None
    axids = [a for a in (list(ZX["anch"]) if ZX is not None else []) if a in sm.get("static", {})]
    if ZX is not None and axids:
        _cols = [k for k, a in enumerate(list(ZX["anch"])) if a in sm["static"]]
        AXS = ZX["s"][:, _cols]; AXS = AXS - np.median(AXS, axis=0, keepdims=True)
        AXPp = ZX["p"][:, _cols]
    else:
        AXS = None
    live = {m["t"]: m for m in g["live"]}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    mvs = {}
    for m in g["moves"]: mvs[m["oid"]] = m["t"]
    lv = {int(os.path.basename(p)[:-4]): p for p in glob.glob(hd + "/live/*.jpg")}
    tsl = [int(t) for t in ts]
    need = {}                                   # (i, oid) → ti
    def ti_of(oid):
        t_ = g["gt0"][oid]["type"]
        return vocab.index(t_) if t_ in vocab else None
    for r in vsc.get(hn, []):                   # a) 걸은 후보 (오검출 포함)
        ti = ti_of(r["oid"])
        if ti is None: continue
        for e in r["scored"]: need[(int(e[0]), r["oid"])] = ti
    for j, oid in enumerate(QT):                # b) GT 목격 8 · c) 최신 5
        v0 = g["gt0"].get(oid)
        if not v0 or not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        if oid not in mvs: continue
        ti = vocab.index(v0["type"]); t0 = mvs[oid]
        sights = sorted([i for i, t in enumerate(tsl)
                         if t > t0 and oid in (live.get(t, {}).get("vis") or [])],
                        key=lambda i: -tsl[i])[:8]
        TS = QS[:, j] + STx[:, j]
        rec5 = sorted(np.argsort(-TS)[:50], key=lambda i: -tsl[i])[:5]
        for i in list(sights) + [int(x) for x in rec5]: need[(int(i), oid)] = ti
    frames = sorted({i for i, _o in need})
    frames = [i for i in frames if tsl[i] in lv]
    zmap = {}
    for b0 in range(0, len(frames), BATCH):
        bi = frames[b0:b0 + BATCH]
        ims = [Image.open(lv[tsl[i]]).convert("RGB") for i in bi]
        W, H = ims[0].size
        with torch.no_grad():
            inp = proc(images=ims, return_tensors="pt").to(DEV)
            pd = mdl(**{k: (v.half() if DEV == "cuda" and v.dtype == torch.float32 else v)
                        for k, v in inp.items()}).predicted_depth
            pd = torch.nn.functional.interpolate(pd.unsqueeze(1).float(), size=(H, W),
                                                 mode="bilinear", align_corners=False)[:, 0]
        for k, i in enumerate(bi): zmap[i] = pd[k].cpu().numpy()
    def kfac(cx, cy, W, H):
        f = W / 2.0
        xh = (cx - W / 2.0) / f; yh = (cy - H / 2.0) / f
        return float(np.sqrt(xh ** 2 + (np.cos(TILT) - yh * np.sin(TILT)) ** 2))
    def zpatch(Z, cx, cy, h2):
        return float(np.median(Z[max(0, cy - h2):cy + h2, max(0, cx - h2):cx + h2]))
    # ── 앵커 쌍 수집: x = 1/z_pred, y = 1/z_기대 (지도 수평거리 → z 환산) ──
    pairs = []; fpairs = {}
    for i in zmap:
        if AXS is None: break
        m = live.get(tsl[i], {}); ap = m.get("apos")
        if ap is None: continue
        Z = zmap[i]; H, W = Z.shape; c2 = max(4, (W // pw) // 2)
        for k in np.where(AXS[i] >= 0.15)[0]:
            a = axids[k]
            if a not in stp: continue
            cx = int((AXPp[i, k] % pw + .5) / pw * W); cy = int((AXPp[i, k] // pw + .5) / ph * H)
            zp = zpatch(Z, cx, cy, c2)
            d_map = float(np.hypot(stp[a][0] - ap[0], stp[a][1] - ap[1]))
            zexp = d_map / kfac(cx, cy, W, H)
            if zp > 0.1 and zexp > 0.1:
                pairs.append((1.0 / zp, 1.0 / zexp))
                fpairs.setdefault(i, []).append((1.0 / zp, 1.0 / zexp))
    # ── 채 단위 역깊이 아핀 (상대 잔차 RANSAC + 퇴화 가드) ──
    A_, B_ = 1.0, 0.0
    if len(pairs) >= 10:
        X = np.array([p_[0] for p_ in pairs]); Y = np.array([p_[1] for p_ in pairs])
        rng_ = np.random.default_rng(0); best = (0, 1.0, 0.0)
        for _ in range(200):
            i1, i2 = rng_.integers(0, len(X), 2)
            if abs(X[i1] - X[i2]) < 1e-6: continue
            a_ = (Y[i1] - Y[i2]) / (X[i1] - X[i2]); b_ = Y[i1] - a_ * X[i1]
            nin = int(np.sum(np.abs((a_ * X + b_) / Y - 1) < 0.10))
            if nin > best[0]: best = (nin, a_, b_)
        nin, a_, b_ = best
        if nin >= max(10, int(0.2 * len(X))) and a_ > 0.05:
            m_ = np.abs((a_ * X + b_) / Y - 1) < 0.10
            M = np.stack([X[m_] / Y[m_], 1.0 / Y[m_]], 1)
            sol, *_ = np.linalg.lstsq(M, np.ones(m_.sum()), rcond=None)
            A_, B_ = float(sol[0]), float(sol[1])
        else:                                    # 퇴화/합의 부족 → 스케일-only
            A_, B_ = float(np.median(Y / X)), 0.0
    elif pairs:
        X = np.array([p_[0] for p_ in pairs]); Y = np.array([p_[1] for p_ in pairs])
        A_, B_ = float(np.median(Y / X)), 0.0
    # 프레임 잔차 배율 (아핀 후, 앵커 2개 이상)
    fr_ = {}
    for i, ps in fpairs.items():
        if len(ps) >= 2:
            r = float(np.median([y / (A_ * x + B_) for x, y in ps]))
            fr_[i] = float(np.clip(r, 0.6, 1.6))
    cell = None
    for (i, oid), ti in sorted(need.items()):
        if i not in zmap: continue
        Z = zmap[i]; H, W = Z.shape
        if cell is None: cell = W // pw
        cx = int((P[i, ti] % pw + .5) / pw * W); cy = int((P[i, ti] // pw + .5) / ph * H)
        k_ = kfac(cx, cy, W, H)
        zp = zpatch(Z, cx, cy, max(4, cell // 2))
        d_raw = zp * k_
        inv = (A_ * (1.0 / zp) + B_) * fr_.get(i, 1.0)
        d = (1.0 / inv) * k_ if inv > 1e-6 else d_raw
        gt = (live.get(tsl[i], {}).get("dist") or {}).get(oid)
        out.write(json.dumps(dict(house=hn, t=tsl[i], oid=oid, d=round(float(d), 2),
                                  d_raw=round(float(d_raw), 2),
                                  gt=gt)) + "\n")
        if gt: errs.append((gt, abs(d - gt) / gt, d_raw / gt))
    out.flush()
    print("  아핀 a=%.3f b=%.3f · 쌍 %d · 프레임보정 %d/%d"
          % (A_, B_, len(pairs), len(fr_), len(zmap)), flush=True)
    print("%s 프레임 %d · 표본 %d" % (hn, len(frames), len(need)), flush=True)
out.close()
errs = np.array(errs) if errs else np.zeros((0, 3))
print("\nGT 대조 %d건 · 보정후 상대오차 중앙값 %.2f · raw 비율(d_raw/gt) 중앙값 %.2f"
      % (len(errs), float(np.median(errs[:, 1])) if len(errs) else -1,
         float(np.median(errs[:, 2])) if len(errs) else -1))
for lo, hi, tag in ((0, 2, "<2m"), (2, 5, "2-5m"), (5, 99, "5m+")):
    sub = errs[(errs[:, 0] >= lo) & (errs[:, 0] < hi)]
    if len(sub): print("  %-5s 상대오차 중앙값 %.2f (n=%d)" % (tag, float(np.median(sub[:, 1])), len(sub)))
print("→", OUTJ)
