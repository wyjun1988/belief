#!/usr/bin/env python3
"""데이터로 만드는 평면도 — 매핑워크의 SfM 포즈 + 단안 메트릭 깊이(DA-V2 × 데이터셋 상수)로 바닥 점유격자를 만들고,
문(좁은 통로)에서 방을 나눈다. 방 이름은 등록 때 사용자가 지점마다 붙인 라벨(여기선 gt.map[i].room 이 그 역할).
**GT 폴리곤은 쓰지 않는다** — 채점(카메라방 적중)에만 GT live.room 을 쓴다.

    python scripts/floorplan_from_map.py data/hssd20S2/house_0000 [--sfm ~/khcache/sfm] [--res 0.1] [--door 0.45]

출력: <sfm>/<house>/floorplan.npz — labels(H×W int, -1=미정/벽), names, origin[x,z], res · 카메라방 적중률(live)
좌표: sfm_reloc 가 내보낸 우리 규약(apos=[x,z] 미러 프레임, yaw 0°=+z 시계증가). 카메라 우측 벡터 부호는 GT 없이 자가검사
(live 카메라 위치가 점유 셀에 덜 떨어지는 쪽)로 고른다.
"""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image
from scipy import ndimage as ndi
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("--sfm", default=os.path.expanduser("~/khcache/sfm"))
ap.add_argument("--res", type=float, default=0.1); ap.add_argument("--door", type=float, default=0.45, help="침식 반경(m) — 문폭의 절반")
ap.add_argument("--k", type=float, default=0.468, help="DA 메트릭 깊이 보정 상수 (§146)"); ap.add_argument("--eye", type=float, default=1.5)
ap.add_argument("--gtpose", action="store_true", help="진단용: 지점·live 포즈를 GT 로 (투영 규약 vs 포즈 문제 분리)")
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"); ap.add_argument("--step", type=int, default=6)
a = ap.parse_args()
hd = a.house.rstrip("/"); hn = os.path.basename(hd); wd = os.path.join(a.sfm, hn)
g = json.load(open(os.path.join(hd, "gt.json"))); live = {m["t"]: m for m in g["live"]}
mp = [json.loads(l) for l in open(os.path.join(wd, "map_pose_%s.jsonl" % hn))]
site_room = {"map/%04d.jpg" % i: m["room"] for i, m in enumerate(g["map"])}
lp = [json.loads(l) for l in open(os.path.join(a.sfm, "pose_%s_da.jsonl" % hn))]      # live SfM 포즈(척도 DA)
if a.gtpose:
    mp = [dict(r, apos=r["apos_gt"], yaw=r["yaw_gt"]) for r in mp]
    lp = [dict(house=hn, t=r["t"], apos=live[r["t"]]["apos"], yaw=live[r["t"]]["yaw"]) for r in lp]
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
proc = AutoImageProcessor.from_pretrained(a.model); mdl = AutoModelForDepthEstimation.from_pretrained(a.model).to(DEV).eval()

def depth(path):
    img = Image.open(path).convert("RGB"); inp = proc(images=img, return_tensors="pt").to(DEV)
    with torch.no_grad(): d = mdl(**inp).predicted_depth
    return torch.nn.functional.interpolate(d[None], size=img.size[::-1], mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy() * a.k

# ── 1. 프레임별 "바닥 레이저": DA 깊이는 잡음(상대오차 ~0.24)이라 높이 문턱으로 바닥/장애물을 가르면 방 안이 장애물로 찬다.
#    대신 바닥 픽셀의 깊이는 기하로 정해진다(d_floor(v) = eye·fx/(v-cy), 피치 0). DA 는 "이 픽셀이 바닥인가"(비율 검사)에만 쓰고,
#    열마다 아래에서 위로 바닥이 이어지는 끝 = 자유 공간의 끝(장애물). 거리엔 기하값을 쓴다.
W = H = None; scans = []           # (apos, yaw, 열 방향 x_c/z_c 단위, d_free, 장애물 여부)
for r in mp:
    D = depth(os.path.join(hd, r["name"]))
    if W is None:
        H, W = D.shape; fx = W / 2.0; cx, cy = W / 2.0, H / 2.0
        vs = np.arange(H - 1, int(cy) + 3, -1); d_floor = a.eye * fx / (vs - cy)          # 아래(가까움)→위(멂)
        us = np.arange(0, W, a.step)
    ang = (us - cx) / fx                                   # 카메라 x/z 비율
    Dc = D[vs][:, us]                                      # (rows, cols)
    # DA 절대 깊이는 바닥에서 35% 짧다(SfM 점으로 맞춘 상수가 바닥엔 안 맞음). 열 안의 **상대** 깊이로 판정:
    # 맨 아래 10행(거의 항상 바닥, 1.5~1.6m)의 깊이를 기준으로 D(v)/D(bottom) 이 d_floor(v)/d_floor(bottom) 을 따르면 바닥.
    # 기울기 판정: 바닥에선 열을 따라 올라갈수록 깊이가 d_floor 처럼 커지고(로그-로그 기울기 ≈1, 압축돼도 >0.4),
    # 벽·가구에선 깊이가 멈춘다(기울기 ≈0). DA 의 절대값·압축과 무관하게 **바닥이 끝나는 행**만 찾고 거리는 기하로.
    lnD = np.log(np.maximum(Dc, 0.05)); lnF = np.log(d_floor)
    win = 12
    dfree = np.empty(len(us)); obst = np.ones(len(us), bool)
    for k in range(len(us)):
        col = lnD[:, k]; top = 0; flat = 0
        for i in range(win, len(col)):
            sl = (col[i] - col[i - win]) / (lnF[i] - lnF[i - win] + 1e-9)
            if sl > 0.4: top = i; flat = 0
            else:
                flat += 1
                if flat >= 4: break
        dfree[k] = min(d_floor[top], 8.0); obst[k] = d_floor[top] < 8.0 and top < len(col) - 3
    scans.append((r["apos"], r["yaw"], ang, dfree, obst))
_df = np.concatenate([sc[3] for sc in scans]); _ob = np.concatenate([sc[4] for sc in scans])
print("바닥 레이저: 열 %d · 장애물로 끝남 %.2f · 자유거리 중앙 %.2fm · 8m 상한 %.2f" % (len(_df), _ob.mean(), np.median(_df), (_df >= 8.0).mean()))
def world(apos, yaw, xc, zc, sign):
    y = np.radians(yaw); f = np.array([np.sin(y), np.cos(y)]); rt = sign * np.array([-np.cos(y), np.sin(y)])   # hssd_to_seq 규약(미러 프레임)
    return apos[0] + xc * rt[0] + zc * f[0], apos[1] + xc * rt[1] + zc * f[1]
xs = np.concatenate([np.r_[p[0][0] - 9, p[0][0] + 9] for p in scans]); zs = np.concatenate([np.r_[p[0][1] - 9, p[0][1] + 9] for p in scans])
ox, oz = xs.min(), zs.min(); GW = int((xs.max() - ox) / a.res) + 1; GH = int((zs.max() - oz) / a.res) + 1
def cell(x, z): return np.clip(((z - oz) / a.res).astype(int), 0, GH - 1), np.clip(((x - ox) / a.res).astype(int), 0, GW - 1)

PTS = np.load(os.path.join(wd, "points_%s.npz" % hn))
_wall = (PTS["h"] > 0.25) & (PTS["h"] < 2.0)
print("SfM 점 %d · 벽/가구 높이대 %d" % (len(PTS["h"]), _wall.sum()))
def build(sign):
    # 합의 점유: 바닥 판정이 한 열에서 잘못 끊기면 방 한가운데 가짜 장애물이 찍힌다 → 셀마다 자유 증거(광선 통과) 수와
    # 장애물 증거 수를 세어, 장애물이 자유보다 많을 때만 점유로 본다.
    fc = np.zeros((GH, GW), np.int32); oc = np.zeros((GH, GW), np.int32)
    for apos, yaw, ang, dfree, obst in scans:
        for k in range(len(ang)):
            dstop = max(0.25, min(dfree[k], 7.0) - 0.25)
            n = max(2, int(dstop / (a.res * 0.5)))
            zc = np.linspace(0.2, dstop, n); xc = zc * ang[k]
            x, z = world(apos, yaw, xc, zc, sign); i, j = cell(x, z); np.add.at(fc, (i, j), 1)
            if obst[k]:
                zo = dfree[k] + 0.05; x, z = world(apos, yaw, np.array([zo * ang[k]]), np.array([zo]), sign); i, j = cell(x, z); np.add.at(oc, (i, j), 1)
    # 벽·가구는 SfM 3D 점(메트릭 정렬)으로: 셀당 점 2개 이상
    wi, wj = cell(PTS["x"][_wall], PTS["z"][_wall]); wc = np.zeros((GH, GW), np.int32); np.add.at(wc, (wi, wj), 1)
    occ = (wc >= 2) | ((oc >= 2) & (oc > fc)); free = (fc > 0) & ~occ
    return free, occ
# 우측 벡터 부호 자가검사: live 카메라 위치가 장애물 셀에 떨어지는 비율이 낮은 쪽
lx = np.array([r["apos"][0] for r in lp]); lz = np.array([r["apos"][1] for r in lp]); li, lj = cell(lx, lz)
best = None
for sign in (1.0, -1.0):
    free, occ = build(sign); bad = (float(occ[li, lj].mean()), int(free.sum()))   # 동률이면 자유셀이 적은(흩어지지 않은) 쪽
    if best is None or bad < best[0]: best = (bad, sign, free, occ)
bad, sign, free, occ = best
bad = bad[0]; print("우측 부호 %+.0f (live 카메라가 장애물 셀에 %.2f) · 격자 %dx%d @%.2fm · 자유 %d · 장애물 %d" % (sign, bad, GH, GW, a.res, free.sum(), occ.sum()))
free[li, lj] = True                                        # live 궤적도 자유 공간 증거
free = ndi.binary_closing(free, iterations=1) & ~occ        # 잔구멍 메움(가구 그림자)
# ── 2. 문에서 나누기: 침식 → 연결성분 → 자유셀을 최근접 성분에 배정 ──
er = ndi.binary_erosion(free, iterations=max(1, int(round(a.door / a.res))))
comp, nc = ndi.label(er)
seed = comp > 0
_, (ii, jj) = ndi.distance_transform_edt(~seed, return_indices=True)
lab = np.where(free, comp[ii, jj], 0)
# ── 3. 성분에 지점 라벨 붙이기 (다수결) ──
from collections import Counter, defaultdict
votes = defaultdict(Counter)
for r in mp:
    i, j = cell(np.array([r["apos"][0]]), np.array([r["apos"][1]])); c = int(lab[i[0], j[0]])
    if c > 0: votes[c][site_room[r["name"]]] += 1
names = {c: v.most_common(1)[0][0] for c, v in votes.items()}
# 이름 없는 성분(지점이 없는 조각)은 최근접 이름 있는 성분으로
named = np.isin(lab, list(names)) 
if names and (~named & (lab > 0)).any():
    _, (ii2, jj2) = ndi.distance_transform_edt(~named, return_indices=True)
    lab = np.where((lab > 0) & ~named, lab[ii2, jj2], lab)
print("성분 %d · 이름 붙은 성분 %d · 방 라벨 %s" % (nc, len(names), sorted(set(names.values()))))
# ── 4. 채점: live 카메라방 ──
ok = n = 0; miss = Counter()
for r in lp:
    i, j = cell(np.array([r["apos"][0]]), np.array([r["apos"][1]])); c = int(lab[i[0], j[0]])
    if c == 0:                                             # 자유셀 밖(장애물/미관측) → 최근접 라벨 셀
        _, (a1, b1) = ndi.distance_transform_edt(lab == 0, return_indices=True) if n == 0 and False else (None, (None, None))
    pred = names.get(c)
    if pred is None:
        d2 = np.argwhere(lab > 0); k = np.argmin((d2[:, 0] - i[0]) ** 2 + (d2[:, 1] - j[0]) ** 2); pred = names.get(int(lab[d2[k, 0], d2[k, 1]]))
    gtr = live[r["t"]]["room"]; n += 1; ok += pred == gtr
    if pred != gtr: miss[(gtr, pred)] += 1
print("%s 카메라방 적중(데이터 평면도) %.2f (n=%d) · 주요 오류 %s" % (hn, ok / n, n, miss.most_common(3)))
# 그림: 자유=흰 · 장애물=검정 · 방 성분=색 · 지점=파랑 · live 카메라=빨강
rgb = np.full((GH, GW, 3), 120, np.uint8); rgb[free] = 255; rgb[occ] = 0
rng_ = np.random.default_rng(1); cols = {c: rng_.integers(80, 230, 3) for c in names}
for c, col in cols.items(): rgb[lab == c] = col
for r in mp:
    i, j = cell(np.array([r["apos"][0]]), np.array([r["apos"][1]])); rgb[max(i[0]-1,0):i[0]+2, max(j[0]-1,0):j[0]+2] = (0, 0, 255)
rgb[li, lj] = (255, 0, 0)
Image.fromarray(rgb[::-1]).resize((GW * 3, GH * 3), Image.NEAREST).save(os.path.join(wd, "floorplan.png"))
np.savez_compressed(os.path.join(wd, "floorplan.npz"), lab=lab, names=json.dumps(names), origin=np.array([ox, oz]), res=a.res, sign=sign)
