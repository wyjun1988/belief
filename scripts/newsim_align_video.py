#!/usr/bin/env python3
"""새 시뮬레이터 스캔 영상 ↔ 주석 행 정렬 (2026-09-10). vol8_02 스캔은 인코더가 프레임을 떨어뜨려 ego.mp4 7,332장 / annotations 9,000행인데
annotations.video_frame 은 0..8999 그대로라 인덱스로 뽑으면 100 s 이후 GT 박스가 엉뚱한 곳에 찍힌다(선형 재사상도 실패: 드롭이 불균일).
방법: 영상 프레임 차분 d_j 는 카메라 이동(Δpos·Δyaw·Δpitch)과 R² 0.70 으로 맞는다(드롭 없는 vol8_01 보정) → 단조 정렬 DP(비디오 j ↔ 주석 행 A(j),
A(j)−A(j−1)−1 = 건너뛴 행 수 s∈[0,Smax]). 회전 구간이 랜드마크가 되어 누적 드리프트가 고정된다.
  python scripts/newsim_align_video.py <scan_episode_dir> [--out video_align.json] [--simulate 0.185]   # 출력: {"n_video","n_ann","video2ann","ann2video"}
  --simulate: 드롭 없는 스캔(vol8_01)에서 프레임을 임의로 떨어뜨려 복원 오차를 잰다(검증)."""
import os, sys, json, argparse, numpy as np, cv2, time
ap = argparse.ArgumentParser(); ap.add_argument("scan"); ap.add_argument("--out", default=""); ap.add_argument("--simulate", default="")
ap.add_argument("--flow", type=int, default=1); ap.add_argument("--smax", type=int, default=60); ap.add_argument("--sbig", type=int, default=900); ap.add_argument("--dbig", type=float, default=25.0); ap.add_argument("--w", default="1.17,1.05,7.83,4.22"); ap.add_argument("--prefix", type=int, default=900); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); W0 = np.array([float(v) for v in a.w.split(",")])
def video_small(mp4):
    c = cv2.VideoCapture(mp4); out = []
    while True:
        ok, fr = c.read()
        if not ok: break
        out.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (192, 144), interpolation=cv2.INTER_AREA))
    return np.stack(out).astype(np.float32)
def diffs(sm): return np.r_[0.0, np.mean(np.abs(sm[1:] - sm[:-1]), axis=(1, 2))]
def shifts(sm):
    """연속 프레임 위상상관 이동(px, 192×144) — 부호 있는 yaw(가로)·pitch(세로) 랜드마크. 응답이 낮으면(평평한 벽) 신뢰 낮음."""
    win = cv2.createHanningWindow((sm.shape[2], sm.shape[1]), cv2.CV_32F); dx = [0.0]; dy = [0.0]; rs = [0.0]
    for j in range(1, len(sm)):
        (x, y), r = cv2.phaseCorrelate(sm[j - 1], sm[j], win); dx.append(x); dy.append(y); rs.append(r)
    return np.array(dx), np.array(dy), np.array(rs)
def motion(cams):
    P = np.array([c["cameras"][0]["location"] for c in cams], float); R = np.array([c["cameras"][0]["rotation_pyr_deg"] for c in cams], float)
    dp = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]; sy = np.r_[0, (np.diff(R[:, 1]) + 180) % 360 - 180]; spi = np.r_[0, np.diff(R[:, 0])]
    return np.cumsum(dp), np.cumsum(np.abs(sy)), np.cumsum(np.abs(spi)), np.cumsum(sy), np.cumsum(spi)
def align(d, cums, w, smax, prefix, slack=30, flow=None, sbig=0, dbig=25.0):
    """d: 비디오 차분(Nv) · cums: 주석 누적 이동(|Δpos|,|Δyaw|,|Δpitch|, 부호 yaw, 부호 pitch)(Na) · flow: (dx,dy,resp) 위상상관 → A(j) 주석 행(Nv)."""
    Nv, Na = len(d), len(cums[0]); D = Na - Nv
    if D <= 0: return np.arange(Nv)
    cp, cy, cpi, csy, cspi = cums
    def pred(i0, i1): return w[0] + w[1] * (cp[i1] - cp[i0]) + w[2] * (cy[i1] - cy[i0]) + w[3] * (cpi[i1] - cpi[i0])
    # 이 영상의 잡음 바닥·스케일: 드롭 전 앞구간(항등 정렬 가정)에서 d ≈ a + b·pred 로 재보정
    j = np.arange(1, min(prefix, Nv)); p0 = pred(j - 1, j); X = np.c_[np.ones(len(j)), p0]; ab, *_ = np.linalg.lstsq(X, d[j], rcond=None)
    res = d[j] - X @ ab; sig0 = max(1.5, float(np.std(res))); print("  앞구간 재보정 d = %.2f + %.2f·pred · 잔차 σ %.2f" % (ab[0], ab[1], sig0), flush=True)
    use_flow = flow is not None
    if use_flow:                                             # 앞구간에서 dx ≈ kx·Δyaw, dy ≈ ky·Δpitch 보정(부호 포함)
        fx_, fy_, fr_ = flow; gy = csy[j] - csy[j - 1]; gp = cspi[j] - cspi[j - 1]; ok = fr_[j] > 0.1
        kx = float(np.sum(fx_[j][ok] * gy[ok]) / max(1e-6, np.sum(gy[ok] ** 2))); ky = float(np.sum(fy_[j][ok] * gp[ok]) / max(1e-6, np.sum(gp[ok] ** 2)))
        sx = max(0.5, float(np.std(fx_[j][ok] - kx * gy[ok]))); sy_ = max(0.5, float(np.std(fy_[j][ok] - ky * gp[ok])))
        r2x = 1 - np.var(fx_[j][ok] - kx * gy[ok]) / max(1e-9, np.var(fx_[j][ok])); print("  위상상관 보정 dx = %.2f·Δyaw (R² %.2f, σ %.2f) · dy = %.2f·Δpitch (σ %.2f) · 응답>0.1 %.0f%%" % (kx, r2x, sx, ky, sy_, 100 * ok.mean()), flush=True)
    Dmax = D + slack; O = np.arange(Dmax + 1); S = np.arange(smax + 1)
    lam0, lam1 = 1.0, 0.08                                   # 드롭 사전확률 벌점 (건너뜀 시작 1.0 + 행당 0.08)
    C = np.full(Dmax + 1, np.inf); C[0] = 0.0               # j=0 ↔ 행 0
    BP = np.zeros((Nv, Dmax + 1), np.uint8); INF = np.inf
    Sbig = np.arange(sbig + 1) if sbig > smax else S
    for jj in range(1, Nv):
        S_ = Sbig if (sbig > smax and d[jj] > dbig) else S   # 장면이 확 바뀐 프레임 = 인코더 정지 후보 → 큰 점프 허용
        i = jj + O                                           # 후보 행 (열)
        ip = i[None, :] - 1 - S_[:, None]                    # 이전 행 (행=s)
        valid = (ip >= jj - 1) & (i[None, :] < Na)          # 이전 오프셋 o−s ≥ 0, 행 범위
        ipc = np.clip(ip, 0, Na - 1); ic = np.minimum(i, Na - 1)
        pr = np.minimum(ab[0] + ab[1] * pred(ipc, ic[None, :]), 60.0)
        sig = np.maximum(sig0, 0.35 * pr)
        cost = ((d[jj] - pr) / sig) ** 2 + lam0 * (S_[:, None] > 0) + lam1 * S_[:, None]
        if use_flow and fr_[jj] > 0.1:
            px = kx * (csy[ic[None, :]] - csy[ipc]); py = ky * (cspi[ic[None, :]] - cspi[ipc]); big = (np.abs(px) > 40) | (np.abs(py) > 30)
            cost = cost + np.where(big, 4.0, ((fx_[jj] - px) / sx) ** 2 + ((fy_[jj] - py) / sy_) ** 2)
        if len(S_) > smax + 1:                               # 큰 점프: 신호 없음 → 평평한 비용(정지 1회 벌점 + 행당 소액)
            cost[smax + 1:] = 6.0 + 0.004 * S_[smax + 1:, None]
        prev = np.full((len(S_), Dmax + 1), INF)
        for s in range(len(S_)):
            if s <= Dmax: prev[s, s:] = C[:Dmax + 1 - s]
        tot = np.where(valid, prev + cost, INF); k = np.argmin(tot, axis=0); C = tot[k, O]; BP[jj] = k
    # 종단: A(Nv−1) ∈ [Na−1−slack, Na−1]
    o_end = np.arange(Dmax + 1); i_end = (Nv - 1) + o_end; ok = (i_end >= Na - 1 - slack) & (i_end <= Na - 1)
    Cend = np.where(ok, C, INF); o = int(np.argmin(Cend)); A = np.zeros(Nv, int); A[Nv - 1] = Nv - 1 + o
    for jj in range(Nv - 1, 0, -1):
        s = int(BP[jj, o]); o -= s; A[jj - 1] = (jj - 1) + o
    return A
scan = a.scan.rstrip("/"); t0 = time.time()
sm = video_small(os.path.join(scan, "ego.mp4")); cams = [json.loads(l) for l in open(os.path.join(scan, "observed_graph_updates.jsonl"))]
Na = len(cams); print("비디오 %d장 · 주석 %d행 (%.0fs)" % (len(sm), Na, time.time() - t0), flush=True)
if a.simulate:                                              # 검증: 드롭 없는 스캔에서 버스트 드롭을 흉내 → 복원 오차. "0.17,0.48,0.03,0,0" = 5구간별 드롭 비율
    prof = [float(v) for v in a.simulate.split(",")]; prof = prof * 5 if len(prof) == 1 else prof
    rng = np.random.default_rng(a.seed); keep = np.ones(len(sm), bool); jj = 1
    while jj < len(sm):
        pq = prof[min(4, 5 * jj // len(sm))]
        if rng.random() < pq / 2: L = int(rng.integers(1, 4)); keep[jj:jj + L] = False; jj += L
        jj += 1
    keep[:a.prefix] = True; truth = np.where(keep)[0]; sm = sm[keep]; print("  모의 드롭 %d장 (%.1f%%) · 남은 %d" % ((~keep).sum(), 100 * (~keep).mean(), len(sm)))
d = diffs(sm); cums = motion(cams); fl = shifts(sm) if a.flow else None; t0 = time.time(); A = align(d, cums, W0, a.smax, a.prefix, flow=fl, sbig=a.sbig, dbig=a.dbig); print("  DP %.0fs" % (time.time() - t0), flush=True)
if a.simulate:
    err = np.abs(A - truth); print("  복원 오차 |A−진짜| 중앙 %.1f 평균 %.1f p90 %.1f 최대 %d · ≤2행 %.1f%% · ≤5행 %.1f%%" % (np.median(err), err.mean(), np.percentile(err, 90), err.max(), 100 * np.mean(err <= 2), 100 * np.mean(err <= 5)))
    for q in range(5):
        sl = slice(q * len(A) // 5, (q + 1) * len(A) // 5); print("   구간 %d: 오차 중앙 %.1f · 드롭 진짜 %d 추정 %d" % (q, np.median(err[sl]), int((truth[sl][-1] - truth[sl][0]) - (sl.stop - sl.start - 1)), int(A[sl][-1] - A[sl][0] - (sl.stop - sl.start - 1))))
    sys.exit(0)
skips = np.diff(A) - 1; print("  건너뛴 행 합 %d (기대 %d) · 5구간별 %s" % (skips.sum(), Na - len(sm), [int(skips[q * len(skips) // 5:(q + 1) * len(skips) // 5].sum()) for q in range(5)]))
ann2video = np.searchsorted(A, np.arange(Na)); ann2video = np.clip(ann2video, 0, len(A) - 1)
for i in range(Na):                                          # 가장 가까운 비디오 프레임
    j = ann2video[i]
    if j > 0 and abs(A[j - 1] - i) <= abs(A[j] - i): ann2video[i] = j - 1
out = a.out or os.path.join(scan, "video_align.json")
json.dump(dict(n_video=int(len(sm)), n_ann=int(Na), video2ann=A.tolist(), ann2video=ann2video.tolist(), method="diff-motion DP", w=W0.tolist()), open(out, "w"))
print("ALIGN_DONE %s" % out)
