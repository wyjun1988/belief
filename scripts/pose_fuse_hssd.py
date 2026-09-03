#!/usr/bin/env python3
"""포즈 융합 — SfM 의 상대 이동량(카메라 좌표) + 앵커 투표 yaw(절대 방위) → 위치 적분. GT 대비 ATE.

    THOR_ROOT=data/hssd20S2 A3_PREFIX=... AX_PREFIX=... python scripts/pose_fuse_hssd.py \\
        --seq data/seq/hs2_house_0000 --house data/hssd20S2/house_0000 --pose pose/poses_da3_anchor.txt --out pose_fused.jsonl

§138: DA3 는 직진에선 정확하고 급회전에서 회전을 놓친다. 그래서 SfM 의 회전은 버리고
  · 프레임 t→t+1 이동 벡터를 SfM 카메라 좌표계(t 기준 전방 z·우 x)로 꺼내고
  · 그 프레임의 절대 yaw 는 앵커 투표(존재 게이트, eval_online 과 동일)로, 없으면 앞뒤 투표를 보간
  · 위치 = 누적합. 스케일은 GT 와 sim3 로 맞춘 단일 스칼라 (배포에선 착용 높이/초기맵 앵커).
앵커가 루프 클로저 역할을 한다 — 회전이 매 프레임 절대값으로 묶이므로 드리프트가 쌓이지 않는다.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kx.depth.pose_stitch import umeyama  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True); ap.add_argument("--house", required=True)
ap.add_argument("--pose", default="pose/poses_da3_anchor.txt"); ap.add_argument("--out", default=None)
a = ap.parse_args()
A3P = os.path.expanduser(os.environ.get("A3_PREFIX")); AXP = os.path.expanduser(os.environ.get("AX_PREFIX"))
W = float(os.environ.get("FRAME_W", "768")); F = W / 2
EX, TY, DP = float(os.environ.get("ANCH_EX", "0.80")), float(os.environ.get("ANCH_TY", "0.10")), int(os.environ.get("ANCH_DP", "2"))
hn = os.path.basename(os.path.realpath(a.house))
g = json.load(open(os.path.join(a.house, "gt.json"))); live = {m["t"]: m for m in g["live"]}; st = g["scene_meta"]["static"]
za = np.load(A3P + hn + ".npz", allow_pickle=True); zx = np.load(AXP + hn + ".npz", allow_pickle=True)
S, P, ts, vocab, pw = za["s"], za["p"], za["ts"], list(za["vocab"]), int(za["pw"])
SX, PX, anch = zx["s"], zx["p"], list(zx["anch"])
tcol = {an: (vocab.index(st[an]["type"]) if an in st and st[an]["type"] in vocab else None) for an in anch}
def br(dx, dz): return np.degrees(np.arctan2(dx, dz))
def pb(cx): return np.degrees(np.arctan((cx - W / 2) / F))
def vote(hyps, win=10.0):
    best = None
    for y, w in hyps:
        mem = [(y2, w2) for y2, w2 in hyps if abs((y2 - y + 180) % 360 - 180) <= win]; sw = sum(w2 for _, w2 in mem)
        if best is None or sw > best[0]: best = (sw, mem)
    if not best: return None
    ys = np.radians([y for y, _ in best[1]]); ws = np.array([w for _, w in best[1]])
    return float(np.degrees(np.arctan2((np.sin(ys) * ws).sum(), (np.cos(ys) * ws).sum()))) % 360
# 앵커 투표 yaw (위치는 **SfM 적분값을 써야** 하지만 방위 역산엔 카메라 위치가 필요 → 1차: GT 위치로 투표(상한),
# 2차: 적분 위치로 재투표(실물). 아래는 2차까지 반복한다.
def vote_yaw_at(i, apos):
    hyps = []
    for k, an in enumerate(anch):
        c = tcol.get(an)
        if an not in st or SX[i, k] < EX or c is None or S[i, c] < TY: continue
        pe, pt = int(PX[i, k]), int(P[i, c])
        if abs(pe % pw - pt % pw) > DP or abs(pe // pw - pt // pw) > DP: continue
        pos = st[an]["pos"]; px = (pe % pw + .5) / pw * W
        hyps.append(((br(pos[0] - apos[0], pos[2] - apos[1]) - pb(px)) % 360, float(SX[i, k])))
    return vote(hyps)
frames = sorted(int(f[:-4]) for f in os.listdir(os.path.join(a.seq, "rgb")) if f.endswith(".jpg"))
Pm = np.loadtxt(os.path.join(a.seq, a.pose)).reshape(-1, 4, 4); n = min(len(Pm), len(frames))
tidx = {int(t): i for i, t in enumerate(ts)}                     # 캐시 프레임(stride 4) 인덱스
gt = np.array([[live[frames[i]]["apos"][0], live[frames[i]]["apos"][1]] for i in range(n)])
ygt = np.array([live[frames[i]]["yaw"] for i in range(n)])
# SfM 상대 이동을 프레임 t 카메라 좌표로: d_cam = R_t^T (c_{t+1} - c_t)  (c2w 가정; x우·y상·z전방)
C = Pm[:n, :3, 3]; R = Pm[:n, :3, :3]
d_cam = np.array([R[i].T @ (C[i + 1] - C[i]) for i in range(n - 1)])
def integrate(yaw):
    pos = np.zeros((n, 2)); 
    for i in range(n - 1):
        y = np.radians(yaw[i]); fwd = np.array([np.sin(y), np.cos(y)]); right = np.array([np.cos(y), -np.sin(y)])
        step = fwd * d_cam[i][2] + right * d_cam[i][0]           # 카메라 z(전방)·x(우) 성분만 (y 는 높이)
        pos[i + 1] = pos[i] + step
    return pos
def yaw_series(apos_of):
    ys = np.full(n, np.nan)
    for i in range(n):
        j = tidx.get(frames[i])
        if j is None: continue
        v = vote_yaw_at(j, apos_of(i))
        if v is not None: ys[i] = v
    # 보간(각도): 가장 가까운 투표값으로 채움
    idx = np.where(~np.isnan(ys))[0]
    if len(idx) == 0: return ys, 0.0
    for i in range(n):
        if np.isnan(ys[i]): ys[i] = ys[idx[np.argmin(np.abs(idx - i))]]
    return ys, len(idx) / n
def evaluate(name, pos, yaw):
    sc, Rm, tm = umeyama(np.c_[pos[:, 0], np.zeros(n), pos[:, 1]], np.c_[gt[:, 0], np.zeros(n), gt[:, 1]])
    al = (sc * (Rm @ np.c_[pos[:, 0], np.zeros(n), pos[:, 1]].T)).T + tm
    ate = np.hypot(al[:, 0] - gt[:, 0], al[:, 2] - gt[:, 1]); ye = np.abs((yaw - ygt + 180) % 360 - 180)
    print("%-34s ATE 중앙 %.2f m · <0.5m %.2f · <1m %.2f | yaw 중앙 %.1f° · 스케일 %.2f" % (name, np.median(ate), (ate < 0.5).mean(), (ate < 1).mean(), np.median(ye), sc))
    return al
# 0) SfM 회전 그대로 (대조)
yaw_sfm = np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2])) % 360
evaluate("SfM 그대로 (회전·이동 모두 SfM)", C[:, [0, 2]], yaw_sfm)
# 1) 상한: GT 위치로 투표한 yaw + SfM 이동량
ys1, cov = yaw_series(lambda i: gt[i]); pos1 = integrate(ys1); evaluate("융합 · 투표(GT위치) 커버 %.2f" % cov, pos1, ys1)
# 2) 실물: 적분 위치로 재투표 (2회 반복)
al = pos1
for it in range(2):
    sc, Rm, tm = umeyama(np.c_[al[:, 0], np.zeros(n), al[:, 1]] if al.shape[1] == 2 else al, np.c_[gt[:, 0], np.zeros(n), gt[:, 1]])
    aligned = ((sc * (Rm @ (np.c_[al[:, 0], np.zeros(n), al[:, 1]] if al.shape[1] == 2 else al).T)).T + tm)[:, [0, 2]]
    ys2, cov2 = yaw_series(lambda i: aligned[i]); pos2 = integrate(ys2)
    al3 = evaluate("융합 · 투표(적분위치) 반복%d 커버 %.2f" % (it + 1, cov2), pos2, ys2); al = al3
if a.out:
    with open(a.out, "w") as f:
        for i in range(n): f.write(json.dumps(dict(house=hn, t=int(frames[i]), apos=[round(float(al[i, 0]), 3), round(float(al[i, 2]), 3)], yaw=round(float(ys2[i]), 2))) + "\n")
    print("→", a.out)
