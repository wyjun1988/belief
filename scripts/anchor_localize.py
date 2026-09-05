#!/usr/bin/env python3
"""앵커 방위 삼각측량 국소화 — SfM 없이 프레임마다 (x, z, yaw) 를 앵커 검출만으로 푼다.

    THOR_ROOT=data/hssd20S2 A3_PREFIX=... AX_PREFIX=... python scripts/anchor_localize.py data/hssd20S2/house_0000 --out pose.jsonl

프레임 i 의 게이트 통과 앵커 k(존재 게이트: exemplar≥ANCH_EX·타입 검출≥ANCH_TY·패치 일치≤ANCH_DP)마다
  atan2(ax_k − x, az_k − z) = yaw + atan((u_k − cx)/f)
가 성립한다(우리 규약). 미지수 3개(x, z, yaw): 앵커 ≥3 이면 직접, 2 이면 직전 해를 위치 사전분포로 풀고,
RANSAC(앵커 부분집합)으로 오검출을 뺀다. 지도 좌표는 scene_meta.static(배포에선 초기맵 좌표).
GT 위치·yaw 는 채점 전용. 출력은 eval_online 의 POSE_JSONL 형식.
"""
import argparse, itertools, json, os
import numpy as np
from scipy.optimize import least_squares

ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("--out", default=None)
ap.add_argument("--min-anch", type=int, default=2); a = ap.parse_args()
A3P = os.path.expanduser(os.environ.get("A3_PREFIX")); AXP = os.path.expanduser(os.environ.get("AX_PREFIX"))
W = float(os.environ.get("FRAME_W", "768")); F = W / 2
EX, TY, DP = float(os.environ.get("ANCH_EX", "0.80")), float(os.environ.get("ANCH_TY", "0.10")), int(os.environ.get("ANCH_DP", "2"))
hn = os.path.basename(os.path.realpath(a.house))
g = json.load(open(os.path.join(a.house, "gt.json"))); live = {m["t"]: m for m in g["live"]}; st = g["scene_meta"]["static"]
za = np.load(A3P + hn + ".npz", allow_pickle=True); zx = np.load(AXP + hn + ".npz", allow_pickle=True)
S, P, ts, vocab, pw = za["s"], za["p"], za["ts"], list(za["vocab"]), int(za["pw"])
SX, PX, anch = zx["s"], zx["p"], list(zx["anch"])
tcol = {an: (vocab.index(st[an]["type"]) if an in st and st[an]["type"] in vocab else None) for an in anch}
def pb(cx): return np.degrees(np.arctan((cx - W / 2) / F))
def obs_at(i):
    out = []
    for k, an in enumerate(anch):
        c = tcol.get(an)
        if an not in st or SX[i, k] < EX or c is None or S[i, c] < TY: continue
        pe, pt = int(PX[i, k]), int(P[i, c])
        if abs(pe % pw - pt % pw) > DP or abs(pe // pw - pt // pw) > DP: continue
        pos = st[an]["pos"]; out.append((float(pos[0]), float(pos[2]), pb((pe % pw + .5) / pw * W), float(SX[i, k])))
    return out
def resid(x, obs, prior=None, wprior=0.0):
    px, pz, yaw = x; r = []
    for ax, az, b, w in obs:
        r.append(w * (((np.degrees(np.arctan2(ax - px, az - pz)) - yaw - b + 180) % 360) - 180) / 5.0)   # 5° 단위
    if prior is not None and wprior > 0: r += [wprior * (px - prior[0]), wprior * (pz - prior[1])]
    return np.array(r)
def solve(obs, init, prior=None, wprior=0.0):
    best = None
    subsets = [obs] if len(obs) <= 3 else [list(c) for c in itertools.combinations(obs, 3)][:40]
    for sub in subsets:
        try: r = least_squares(resid, init, args=(sub, prior, wprior), loss="soft_l1", f_scale=2.0)
        except Exception: continue
        # 전체 관측에 대한 인라이어 수로 채점
        full = np.abs(resid(r.x, obs)) * 5.0; inl = int((full < 8).sum())
        if best is None or (inl, -full.mean()) > (best[0], -best[1]): best = (inl, full.mean(), r.x)
    return best
frames_i = [i for i in range(len(ts)) if int(ts[i]) in live]
sol = {}; prev = None
for i in frames_i:
    obs = obs_at(i)
    if len(obs) < a.min_anch or (len(obs) == 2 and prev is None): continue
    init = np.array([prev[0], prev[1], prev[2]]) if prev is not None else np.array([0.0, 0.0, 0.0])
    if prev is None:   # 첫 프레임: 지도 중심 초기값 여러 yaw 시도
        cands = [solve(obs, np.array([np.mean([o[0] for o in obs]), np.mean([o[1] for o in obs]), y0])) for y0 in (0, 90, 180, 270)]
        cands = [c for c in cands if c]; best = max(cands, key=lambda c: (c[0], -c[1])) if cands else None
    else:
        best = solve(obs, init, prior=prev[:2], wprior=0.5 if len(obs) == 2 else 0.1)
    if not best or best[0] < min(2, len(obs)): continue
    x = best[2]; sol[int(ts[i])] = (float(x[0]), float(x[1]), float(x[2] % 360), len(obs), best[0]); prev = x
# 채점
if sol:
    T = sorted(sol); E = np.array([[sol[t][0], sol[t][1]] for t in T]); G = np.array([live[t]["apos"] for t in T])
    ate = np.hypot(*(E - G).T); ye = np.abs((np.array([sol[t][2] for t in T]) - np.array([live[t]["yaw"] for t in T]) + 180) % 360 - 180)
    print("%s 캐시 프레임 %d · 해 %d (커버 %.2f) · 앵커/프레임 중앙 %d" % (hn, len(frames_i), len(sol), len(sol) / len(frames_i), int(np.median([sol[t][3] for t in T]))))
    print("  위치 오차 중앙 %.2f m · <0.5m %.2f · <1m %.2f | yaw 오차 중앙 %.1f° · <10° %.2f" % (np.median(ate), (ate < .5).mean(), (ate < 1).mean(), np.median(ye), (ye < 10).mean()))
    for k in (3, 4):
        m = np.array([sol[t][3] >= k for t in T])
        if m.any(): print("  앵커 ≥%d 프레임(%d): 위치 중앙 %.2f m · yaw 중앙 %.1f°" % (k, m.sum(), np.median(ate[m]), np.median(ye[m])))
    if a.out:
        with open(a.out, "w") as f:
            for t in T: f.write(json.dumps(dict(house=hn, t=int(t), apos=[round(sol[t][0], 3), round(sol[t][1], 3)], yaw=round(sol[t][2], 2), n_anch=sol[t][3])) + "\n")
        print("→", a.out)
else: print("해 없음")
