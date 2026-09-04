#!/usr/bin/env python3
"""VGGT 재국소화 어댑터 — 지도 1회 + live 프레임별 등록. COLMAP(sfm_reloc.py) 과 **같은 표**로 재려고 만든 것.

구조(우리 과제에 맞춘 이유): 1fps live 는 프레임 간 겹침이 없어 추적이 불가능하다. 그래서
  (1) 매핑워크 프레임을 **한 번의 전방 통과**로 재구성해 지도 포즈를 얻고,
  (2) live 는 [지도 앵커 K장 + live 묶음 C장] 을 함께 넣어 묶음마다 독립으로 풀고, 앵커의 기지 포즈로 좌표를 옮긴다.
증분 등록이 아니므로 드리프트도, COLMAP 을 무너뜨린 **지도 접힘**도 구조적으로 덜하다(§153).

    python scripts/vggt_reloc.py data/og20/house_0000 --out raw_house_0000.jsonl
    # 그 뒤 정렬·평가·요약은 COLMAP 판과 동일한 코드로:
    python scripts/sfm_reloc.py data/og20/house_0000 --from-poses raw_house_0000.jsonl \
        --scale da --align sites --out ~/khcache/vggt/pose_house_0000.jsonl

출력 raw jsonl: {"name": "map/0000.jpg" | "live/000000.jpg", "c": [x,y,z], "f": [x,y,z]}  (미터, y-up 월드)
  c = 카메라 중심, f = 광축 전방 벡터. sfm_reloc --from-poses 가 이 형식을 읽는다.
척도: VGGT 는 up-to-scale → **DA-V2 metric 깊이와의 비율 중앙값**으로 미터화(§146 과 같은 원리, GT 불필요).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("house")
ap.add_argument("--out", default=None)
ap.add_argument("--global-live-step", type=int, default=0, help="단일 전역 통과: 지도 전부 + live N장마다 1장을 **한 번에** 푼다(0=끔). 이 live 프레임들은 이후 묶음의 앵커가 된다 — 시간상 이웃이라 서로 겹친다(지점 회전 앵커의 무시차 문제 회피)")
ap.add_argument("--map-max", type=int, default=96, help="지도 재구성에 넣을 매핑 프레임 상한(메모리). 초과하면 균등 표본")
ap.add_argument("--anchor-mode", default="window", choices=["window", "topk"], help="window=매핑워크의 **연속 구간**(서로 겹쳐 VGGT 가 묶어 풀 수 있다) · topk=기술자 상위(흩어져 있어 서로 연결 불가 → rms>1 로 전부 기권)")
ap.add_argument("--selfcheck", action="store_true", help="지도 프레임 부분집합을 두 번째 통과로 다시 풀어 지도 통과와의 sim3 잔차를 본다 — 실패 원인이 검색인지 규약/재현성인지 가른다")
ap.add_argument("--anchors", type=int, default=8, help="live 묶음마다 함께 넣는 지도 앵커 수")
ap.add_argument("--chunk", type=int, default=8, help="한 번에 푸는 live 프레임 수")
ap.add_argument("--query-frames", default=None, help="json {house:[t,…]} — 이 프레임들만 국소화(평가기가 포즈를 쓰는 검증 후보 프레임). scripts/query_frames.py 가 만든다. 1fps 전부를 이어 붙이지 않는다")
ap.add_argument("--live-step", type=int, default=1, help="live N장마다 1장 (긴 에피소드)")
ap.add_argument("--res", type=int, default=518, help="VGGT 입력 해상도(기본 518)")
ap.add_argument("--model", default="facebook/VGGT-1B")
ap.add_argument("--inlier-th", type=float, default=0.35, help="앵커 RANSAC 인라이어 문턱(정규화 잔차)")
ap.add_argument("--min-inliers", type=int, default=5, help="정렬에 필요한 최소 앵커 수")
ap.add_argument("--rms-max", type=float, default=0.5, help="앵커 정렬 rms(상대) 상한 — 넘으면 그 묶음 기권")
ap.add_argument("--da-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
ap.add_argument("--da-n", type=int, default=24, help="척도 추정에 쓸 프레임 수")
a = ap.parse_args()

hd = a.house.rstrip("/"); hn = os.path.basename(hd)
T0 = time.time()
def log(*x): print("[%6.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DT = torch.bfloat16 if (DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
model = VGGT.from_pretrained(a.model).to(DEV).eval()
log("VGGT 적재 · %s · %s" % (DEV, DT))

maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg"))
lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg"))
n_live0 = len(lives)
if a.query_frames:
    _q = json.load(open(a.query_frames)).get(hn, [])
    _qs = set("%06d.jpg" % t for t in _q); lives = [f for f in lives if f in _qs]
    log("질의 프레임 모드: %d/%d 장만 국소화" % (len(lives), n_live0))
elif a.live_step > 1: lives = lives[::a.live_step]
log("매핑 %d · live %d(원본 %d)" % (len(maps), len(lives), n_live0))

import tempfile, shutil
_STAGE = tempfile.mkdtemp(prefix="vggt_order_")
def _load(paths):
    """순서 보장 + 전처리 일관성. 두 가지를 동시에 만족해야 한다:
      (1) load_and_preprocess_images 는 버전에 따라 경로를 **내부 정렬**한다 → 순서가 깨진다.
      (2) 그렇다고 한 장씩 부르면 **묶음 공통 크기 리사이즈**가 사라져 프레임마다 스케일·주점이 달라지고
          기하가 통째로 깨진다(2026-09-04 자가검사에서 연속 앵커도 rms>1 이 나온 원인).
    → 임시 디렉터리에 `%03d_` 접두사로 심볼릭 링크를 걸어 **정렬해도 내 순서와 같게** 만들고, 전처리는 묶음으로 한다."""
    for f in os.listdir(_STAGE): os.unlink(os.path.join(_STAGE, f))
    lp = []
    for k, p_ in enumerate(paths):
        d_ = os.path.join(_STAGE, "%04d_%s" % (k, os.path.basename(p_)))
        os.symlink(os.path.abspath(p_), d_); lp.append(d_)
    return load_and_preprocess_images(sorted(lp))
def run(paths):
    """VGGT 한 번 → (중심 (N,3), 전방 (N,3), 깊이 (N,H,W) 또는 None)"""
    im = _load(paths).to(DEV)
    with torch.no_grad():
        with torch.autocast(device_type=("cuda" if DEV == "cuda" else "cpu"), dtype=DT, enabled=(DEV == "cuda")):
            pred = model(im)
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], im.shape[-2:])
    E = extri[0].float().cpu().numpy()          # (N,3,4) world→cam
    R = E[:, :3, :3]; t = E[:, :3, 3]
    C = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)     # 카메라 중심
    F = np.transpose(R, (0, 2, 1))[:, :, 2]                         # 광축(카메라 +z) → 월드
    U = -np.transpose(R, (0, 2, 1))[:, :, 1]                        # 카메라 up(−y) → 월드 (중력 추정용)
    D = pred["depth"][0].float().cpu().numpy()[..., 0] if "depth" in pred else None
    run.last_U = U
    return C, F, D

# ── 1. 지도: 한 번의 전방 통과 (+ 선택: live 표본을 같은 통과에 넣어 전역 앵커로) ──
midx = list(range(len(maps)))
if len(midx) > a.map_max: midx = midx[:: int(np.ceil(len(midx) / a.map_max))][:a.map_max]
mpaths = [os.path.join(hd, "map", maps[i]) for i in midx]
gl_idx = list(range(0, len(lives), a.global_live_step)) if a.global_live_step > 0 else []
gpaths = [os.path.join(hd, "live", lives[i]) for i in gl_idx]
Call, Fall, Dall = run(mpaths + gpaths); Uall = run.last_U
Cm, Fm = Call[:len(mpaths)], Fall[:len(mpaths)]; Dm = Dall[:len(mpaths)] if Dall is not None else None
Cg, Fg = Call[len(mpaths):], Fall[len(mpaths):]
log("전역 통과 %d프레임 (지도 %d + live 표본 %d) · 지도 중심 산포 %.2f" % (len(mpaths) + len(gpaths), len(mpaths), len(gpaths), float(np.std(Cm))))

# ── 2. 척도: DA 메트릭 깊이 / VGGT 깊이 비율 중앙값 (GT 불필요) ──
scale = 1.0
if Dm is not None:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dp = AutoImageProcessor.from_pretrained(a.da_model)
    dm = AutoModelForDepthEstimation.from_pretrained(a.da_model).to(DEV).eval()
    rat = []
    for k in range(0, len(mpaths), max(1, len(mpaths) // a.da_n)):
        img = Image.open(mpaths[k]).convert("RGB")
        with torch.no_grad(): dd = dm(**dp(images=img, return_tensors="pt").to(DEV)).predicted_depth
        H, W = Dm[k].shape
        dd = torch.nn.functional.interpolate(dd[None], size=(H, W), mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
        v = Dm[k]; ok = (v > 0.05) & (dd > 0.05)
        if ok.sum() > 500: rat.append(float(np.median(dd[ok] / v[ok])))
    if rat: scale = float(np.median(rat))
    log("척도(DA/VGGT) %.3f · 프레임 %d · 산포 %.2f" % (scale, len(rat), float(np.std(rat) / (np.median(rat) + 1e-9))))
    del dm
Cm = Cm * scale; Cg = Cg * scale

def _um(X, Y):
    n = len(X); cx, cy = X.mean(0), Y.mean(0); H = (X - cx).T @ (Y - cy) / n
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    var = float(((X - cx) ** 2).sum() / n); sc = float((S[:2].sum() + d * S[2]) / max(var, 1e-9))
    return sc, R, cy - sc * (R @ cx)
if a.selfcheck:
    # 지도 프레임의 연속 구간을 **두 번째 통과**로 다시 풀어, 지도 통과와 sim3 로 맞춰 본다.
    # 잔차가 작으면 VGGT 규약·재현성은 정상이고 문제는 앵커 검색 → window 모드로 해결된다.
    C0r, _, _ = run(mpaths)                      # 동일 입력 재실행 — 재현성 확인 (여기서 실패하면 모델/전처리 문제)
    _sc, _R0, _t0 = _um(C0r, Cm / scale if scale else Cm)
    _r0 = np.linalg.norm((_sc * (_R0 @ C0r.T)).T + _t0 - (Cm / scale if scale else Cm), axis=1)
    log("자가검사 동일 %d장 재실행: 잔차 중앙 %.4f · 정규화 rms %.4f (0 에 가까워야 정상)" % (
        len(mpaths), float(np.median(_r0)), float(np.sqrt((_r0**2).mean())) / (float(np.std(Cm / (scale or 1))) + 1e-9)))
    for tag, idx in (("연속 8장", list(range(0, 8))), ("연속 8장(중간)", list(range(len(midx) // 2, len(midx) // 2 + 8))),
                     ("흩어진 8장", list(range(0, len(midx), max(1, len(midx) // 8)))[:8])):
        C2, F2, _ = run([mpaths[i] for i in idx])
        sc, R2, t2 = _um(C2, Cm[idx])
        r = np.linalg.norm((sc * (R2 @ C2.T)).T + t2 - Cm[idx], axis=1)
        log("자가검사 %s: 잔차 중앙 %.3fm · 정규화 rms %.3f" % (tag, float(np.median(r)), float(np.sqrt((r**2).mean())) / (float(np.std(Cm[idx])) + 1e-9)))
    log("→ '연속' 이 작고 '흩어진' 이 크면 원인은 앵커 검색(= --anchor-mode window 로 해결). 둘 다 크면 VGGT 규약/재현성 문제.")

# ── 3. 검색용 전역 기술자 (의존성 0: 32x32 회색조 정규화 코사인) ──
def desc(path):
    g = np.asarray(Image.open(path).convert("L").resize((32, 32)), np.float32).ravel()
    g -= g.mean(); n = np.linalg.norm(g) + 1e-9
    return g / n
Dm_desc = np.stack([desc(p) for p in mpaths])

# ── 4. live: [앵커 K + 묶음 C] 를 함께 풀고 앵커로 sim3 이동 ──
def umeyama_rigid(X, Y):
    """Umeyama sim3. ⚠️ 척도는 **분산**(제곱합/n)으로 나눈다 — 제곱합으로 나누면 앵커 수만큼 작아져
    잔차가 폭발하고 모든 묶음이 기권한다(2026-09-04 RTX 첫 실행에서 158/158 기권한 원인)."""
    n = len(X); cx, cy = X.mean(0), Y.mean(0)
    H = (X - cx).T @ (Y - cy) / n
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    var = float(((X - cx) ** 2).sum() / n)
    s = float((S[:2].sum() + d * S[2]) / max(var, 1e-9))
    return s, R, cy - s * (R @ cx)
rows = [dict(name="map/" + maps[i], c=[round(float(v), 4) for v in Cm[k]], f=[round(float(v), 4) for v in Fm[k]], u=[round(float(v), 4) for v in Uall[k]])
        for k, i in enumerate(midx)]
gl_set = set(gl_idx)
for k, i in enumerate(gl_idx):          # 전역 통과에 들어간 live 는 그대로 확정 포즈
    rows.append(dict(name="live/" + lives[i], c=[round(float(v), 4) for v in Cg[k]], f=[round(float(v), 4) for v in Fg[k]], u=[round(float(v), 4) for v in Uall[len(mpaths) + k]]))
nchunk = int(np.ceil(len(lives) / a.chunk)); nok = 0; _rmss = []; _nins = []; last_c = None
for ci in range(nchunk):
    grp = lives[ci * a.chunk:(ci + 1) * a.chunk]
    if not grp: continue
    gp = [os.path.join(hd, "live", f) for f in grp]
    if gl_idx:
        # 전역 앵커: 이 묶음과 **시간상 가장 가까운** 전역-통과 live 프레임들 (서로 겹치는 이웃)
        _c0 = ci * a.chunk + len(grp) // 2
        _near = sorted(range(len(gl_idx)), key=lambda k: abs(gl_idx[k] - _c0))[:a.anchors]
        paths = [gpaths[k] for k in _near] + gp
        try: C, F, _ = run(paths)
        except Exception as e: log("묶음 %d 실패: %s" % (ci, str(e)[:80])); continue
        order = np.array(_near); Cm_anchor = Cg[order]
    else:
        Cm_anchor = None
    q = np.stack([desc(p) for p in gp]).mean(0)
    _by_desc = np.argsort(-(Dm_desc @ q))
    if gl_idx: pass
    elif a.anchor_mode == "window":
        # ⚠️ 흩어진 앵커는 서로 시야가 겹치지 않아 VGGT 가 한 묶음 안에서 연결하지 못한다(정규화 rms > 1, 전부 기권).
        # 매핑워크는 순서가 곧 경로이므로 **연속 구간**을 쓰면 앵커끼리 겹쳐 국소적으로 강체가 된다.
        _c = int(_by_desc[0]) if last_c is None else int(np.argmin(np.linalg.norm(Cm - last_c, axis=1)))
        _h = a.anchors // 2
        _lo = max(0, min(_c - _h, len(midx) - a.anchors)); order = np.arange(_lo, min(_lo + a.anchors, len(midx)))
    elif last_c is not None:
        _by_pos = np.argsort(np.linalg.norm(Cm - last_c, axis=1))
        order = np.array(list(dict.fromkeys(list(_by_pos[:a.anchors // 2]) + list(_by_desc[:a.anchors])))[:a.anchors])
    else:
        order = np.array(list(_by_desc[:a.anchors]))
    if not gl_idx:
        paths = [mpaths[i] for i in order] + gp
        try:
            C, F, _ = run(paths)
        except Exception as e:
            log("묶음 %d 실패: %s" % (ci, str(e)[:80])); continue
        Cm_anchor = Cm[order]
    na = len(order)
    # 앵커 전부/전무 정렬은 한 장만 어긋나도(방 전환·급회전 구간) 묶음 전체를 버린다.
    # → 3점 RANSAC 으로 **맞는 앵커만** 골라 정렬한다. 최소 인라이어 수는 --min-inliers.
    _A, _B = C[:na], Cm_anchor; _sc = float(np.std(_B)) + 1e-9
    _best = None
    _rng = np.random.default_rng(0)
    for _ in range(60 if na > 3 else 1):
        _i3 = _rng.choice(na, 3, replace=False) if na > 3 else np.arange(na)
        try: _s3, _R3, _t3 = umeyama_rigid(_A[_i3], _B[_i3])
        except Exception: continue
        _e = np.linalg.norm((_s3 * (_R3 @ _A.T)).T + _t3 - _B, axis=1) / _sc
        _in = _e < a.inlier_th
        if _best is None or _in.sum() > _best.sum(): _best = _in
    if _best is None or _best.sum() < a.min_inliers: _best = np.ones(na, bool)
    _use = _best if _best.sum() >= a.min_inliers else np.ones(na, bool)
    s_, R_, t_ = umeyama_rigid(_A[_use], _B[_use])
    res = np.linalg.norm((s_ * (R_ @ _A[_use].T)).T + t_ - _B[_use], axis=1)
    rms = float(np.sqrt((res ** 2).mean())) / _sc
    _nin = int(_use.sum())
    _rmss.append(rms); _nins.append(_nin)
    if ci == 0:
        log("첫 묶음 진단: 앵커 %d(인라이어 %d) · 척도 %.3f · 잔차(m) %s · 정규화 rms %.3f · 앵커 산포 %.2f" % (
            na, _nin, s_, np.round(res, 2).tolist(), rms, float(np.std(Cm_anchor))))
    if rms > a.rms_max: continue                      # 앵커가 안 맞는 묶음은 기권
    nok += 1
    for k, f in enumerate(grp):
        if (ci * a.chunk + k) in gl_set: continue          # 전역 통과에서 이미 확정
        c = s_ * (R_ @ C[na + k]) + t_; fv = R_ @ F[na + k]
        rows.append(dict(name="live/" + f, c=[round(float(v), 4) for v in c], f=[round(float(v), 4) for v in fv]))
        last_c = c
    if ci % 20 == 0: log("묶음 %d/%d · 채택 %d" % (ci + 1, nchunk, nok))
out = a.out or os.path.join(os.path.dirname(hd), "raw_%s.jsonl" % hn)
with open(out, "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
if _rmss:
    _q = np.percentile(_rmss, [10, 50, 90])
    log("앵커 정렬 rms 분위 10/50/90 = %.3f / %.3f / %.3f (상한 %.2f) · 인라이어 중앙 %d/%d" % (_q[0], _q[1], _q[2], a.rms_max, int(np.median(_nins)) if _nins else 0, a.anchors))
    if nok == 0: log("⚠️ 전부 기권 — rms 중앙값이 상한보다 크면 --rms-max 를 그 값 이상으로. 중앙값이 1 이상이면 정렬 자체가 실패(앵커 검색 불량).")
log("→ %s · 지도 %d · live %d/%d (묶음 채택 %d/%d)" % (out, len(midx), len(rows) - len(midx), n_live0, nok, nchunk))
