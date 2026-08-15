#!/usr/bin/env python3
"""물체 랜드마크로 **증분 SfM** — 지도를 키우면서 앞뒤를 되돌아 교정한다.

    $P scripts/incremental_sfm.py --seq <name>

배치 정제(refine_bootstrap.py)는 v1 스티칭 초기값에서 출발해 국소 최소점에 갇혔다
(가중치를 60배 바꿔도 결과가 소수점까지 동일했다). 틀린 랜드맵과 틀린 포즈가 서로를
정당화하기 때문이다.

여기서는 구조를 바꾼다 — **v1 을 국소적으로만 믿는다**:

    ① 씨앗    공통 랜드마크가 가장 많은 **짧은 구간**만 v1 포즈로 초기화
              (윈도우 내부는 DA3 가 일관되게 푼다. 드리프트는 윈도우 **사이**에서 생긴다)
    ② 등록    지도에 있는 랜드마크를 가장 많이 보는 프레임부터 하나씩 PnP 로 붙인다
    ③ 성장    새로 등록된 프레임 덕에 시차가 생긴 랜드마크를 **새로 삼각측량**
    ④ 교정    N 개마다 지역 BA. 등록된 프레임이 시간적으로 멀리 떨어진 곳과
              랜드마크를 공유하면(=루프) **전역 BA** 로 전체를 다시 편다

데이터에 루프는 충분하다 — 실측: 랜드마크 140개 중 43개가 30초 이상 시간을 가로지르고,
상위 5개는 span 905~915프레임에 중간 공백 225~375프레임이다.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.depth.pose_stitch import robust_umeyama                 # noqa: E402
from scripts.refine_bootstrap import (BORDER, MIN_PX, load_obs,  # noqa: E402
                                      triangulate)

MIN_TRI_VIEWS = 3
MIN_PARALLAX = 0.20      # m. 이보다 시차가 작으면 삼각측량하지 않는다
LOOP_GAP = 200           # 프레임. 이만큼 떨어진 구간과 랜드마크를 공유하면 루프로 본다


def accept_point(X, views, K, max_reproj=6.0, max_depth=25.0):
    """삼각측량 결과 검증 — **이게 없으면 지도가 통째로 오염된다.**

    시차가 모자란 관측은 점을 무한대나 카메라 뒤로 보낸다. 실측: 검증 없이 돌렸더니
    씨앗 BA 재투영 RMS 가 **166,215 px** 였다. 세 가지를 본다:
      ① 카이랄리티  모든 뷰에서 카메라 앞에 있어야 한다
      ② 거리        비현실적으로 멀면 시차 부족의 신호다
      ③ 재투영      실제로 관측을 설명하는가
    """
    if X is None or not np.all(np.isfinite(X)):
        return False
    errs = []
    for T, (u, v) in views:
        R, t = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
        Xc = R @ X + t
        if Xc[2] <= 0.05 or Xc[2] > max_depth:
            return False
        du = K[0, 0] * Xc[0] / Xc[2] + K[0, 2] - u
        dv = K[1, 1] * Xc[1] / Xc[2] + K[1, 2] - v
        errs.append(np.hypot(du, dv))
    return float(np.median(errs)) <= max_reproj


def pnp(pts3, pts2, K, reproj=8.0):
    if len(pts3) < 5:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.asarray(pts3, np.float64), np.asarray(pts2, np.float64), K, None,
        flags=cv2.SOLVEPNP_SQPNP, reprojectionError=reproj,
        iterationsCount=1000, confidence=0.9999)
    if not ok or inl is None or len(inl) < 5:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ tvec).ravel()
    return T, len(inl)


def local_ba(poses, land, obs, K, fids=None, f_scale=2.0, max_nfev=40,
             anchor=None, w_scale=1000.0):
    """등록 프레임(또는 부분집합)과 그들이 보는 랜드마크를 재최적화.

    ② fids 밖의 등록 프레임이 같은 랜드마크를 보면 그 **관측도 잔차에 넣되 포즈는
       동결**한다. 안 그러면 창 BA 가 돌 때마다 공유 랜드마크가 최근 관측 쪽으로만
       끌려가 옛 프레임 기준의 지도가 조용히 오염된다.
    ③ 단안 BA 는 첫 카메라를 고정해도 **스케일이 자유**다. 실측: 앵커 없이는 지도가
       40~80% 쪼그라든 채 수렴했다(정합스케일 1.56~1.83). anchor=(fa, fb, d0) 로
       두 카메라 간 거리를 초기값 d0 에 묶는다. 동결 프레임이 있으면 그쪽이
       게이지·스케일을 이미 잡으므로 앵커·첫카메라 고정 둘 다 쓰지 않는다.
       w_scale=100 은 huber 선형영역에서 픽셀항에 밀렸다(실측: d0 에서 23% 이탈).
       1000 이면 1% 이탈이 이미 임계(2px) 밖이라 사실상 강제 구속이 된다.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    fids = sorted(fids if fids is not None else poses)
    used = {l for f in fids for l in obs.get(f, {}) if l in land}
    frozen = [f for f in sorted(set(poses) - set(fids)) if used & set(obs.get(f, {}))]
    allf = fids + frozen
    fi = {f: i for i, f in enumerate(allf)}
    nF = len(fids)
    lids = sorted(used)
    if len(fids) < 3 or len(lids) < 6:
        return poses, land, None
    li = {l: i for i, l in enumerate(lids)}
    rows = [(fi[f], li[l], o[0], o[1]) for f in allf
            for l, o in obs.get(f, {}).items() if l in li]
    if len(rows) < 30:
        return poses, land, None
    F, L = len(allf), len(lids)

    x0 = np.zeros(F * 6 + L * 3)
    for f, i in fi.items():
        R = poses[f][:3, :3].T
        x0[i * 6:i * 6 + 3] = cv2.Rodrigues(R)[0].ravel()
        x0[i * 6 + 3:i * 6 + 6] = -R @ poses[f][:3, 3]
    for l, i in li.items():
        x0[F * 6 + i * 3:F * 6 + i * 3 + 3] = land[l]

    ci = np.array([r[0] for r in rows])
    pi = np.array([r[1] for r in rows])
    uv = np.array([[r[2], r[3]] for r in rows])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    anc = None                                  # (i_a, i_b, d0) 인덱스로 변환
    if not frozen and anchor is not None:
        fa, fb, d0 = anchor
        if fa in fi and fb in fi and d0 > 1e-3:
            anc = (fi[fa], fi[fb], d0)

    def center(x, i):
        R = cv2.Rodrigues(x[i * 6:i * 6 + 3])[0]
        return -R.T @ x[i * 6 + 3:i * 6 + 6]

    nres = len(rows) * 2 + (1 if anc else 0)

    def resid(x):
        P = x[F * 6:].reshape(L, 3)[pi]
        out = np.empty(nres)
        for c in np.unique(ci):
            m = ci == c
            R = cv2.Rodrigues(x[c * 6:c * 6 + 3])[0]
            X = P[m] @ R.T + x[c * 6 + 3:c * 6 + 6]
            z = np.maximum(X[:, 2], 1e-3)
            idx = np.nonzero(m)[0]
            out[2 * idx] = fx * X[:, 0] / z + cx - uv[m, 0]
            out[2 * idx + 1] = fy * X[:, 1] / z + cy - uv[m, 1]
        if anc:
            ia, ib, d0 = anc
            out[-1] = w_scale * (np.linalg.norm(center(x, ia) - center(x, ib)) - d0)
        return out

    S = lil_matrix((nres, len(x0)), dtype=int)
    for k, (c, p, _, _) in enumerate(rows):
        if c < nF:                              # 동결 카메라 열은 비워 둔다
            S[2 * k:2 * k + 2, c * 6:c * 6 + 6] = 1
        S[2 * k:2 * k + 2, F * 6 + p * 3:F * 6 + p * 3 + 3] = 1
    if anc:
        for i in anc[:2]:
            S[-1, i * 6:i * 6 + 6] = 1
    if not frozen:
        S[:, :6] = 0                            # 게이지: 첫 카메라 고정

    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=f_scale,
                        method="trf", max_nfev=max_nfev, verbose=0)
    x = res.x
    if not frozen:
        x[:6] = x0[:6]
    for f in fids:
        i = fi[f]
        R = cv2.Rodrigues(x[i * 6:i * 6 + 3])[0]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ x[i * 6 + 3:i * 6 + 6]
        poses[f] = T
    for l, i in li.items():
        land[l] = x[F * 6 + i * 3:F * 6 + i * 3 + 3]
    rms = float(np.sqrt(np.mean(res.fun[:len(rows) * 2] ** 2)))
    return poses, land, rms

def ate(est, gt):
    v = sorted(est)
    A = np.array([est[i][:3, 3] for i in v])
    B = np.array([gt[i][:3, 3] for i in v])
    s, R, t, _ = robust_umeyama(A, B)
    d = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    return float(np.median(d)), float(np.percentile(d, 90)), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--init", default="pose/poses_da3_lc.txt")
    ap.add_argument("--seg", default="gt/seg")
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-extent", type=float, default=1.0)
    ap.add_argument("--min-fill", type=float, default=0.0, help="마스크/bbox 채움비 하한(가림 필터)")
    ap.add_argument("--seed-len", type=int, default=12, help="씨앗에 쓸 프레임 개수")
    ap.add_argument("--seed-span", type=int, default=60, help="씨앗이 걸치는 원본 프레임 수(시간)")
    ap.add_argument("--ba-every", type=int, default=10)
    ap.add_argument("--fuse-emb", default=None,
                    help="track_emb.npz 경로(시퀀스 상대). FastSAM 트랙은 끊겨서 지도가 못 크는데,"
                         " 삼각측량된 새 점이 기존 랜드마크와 3D 근접+임베딩 일치+시간 중첩 0 이면"
                         " 같은 물체로 융합한다(re-ID = 루프 감지의 FastSAM 판)")
    ap.add_argument("--fuse-dist", type=float, default=0.5)
    ap.add_argument("--const", action="store_true",
                    help="별자리 루프병합. 기본 꺼짐 — CLIP 크롭 임베딩은 실내 소품"
                         " 판별력이 없어 cos0.90+3쌍+0.5m 게이트가 18/18 오병합했다(2026-08-16)")
    ap.add_argument("--still-dynamic", action="store_true",
                    help="motion_type 이 동적이라도 이 시퀀스에서 안 움직였으면(moves=False)"
                         " 랜드마크로 쓴다. 접시·소품류가 늘어 관측 밀도가 오른다")
    ap.add_argument("--const-cos", type=float, default=0.90,
                    help="별자리 루프 감지의 임베딩 하한 (융합 0.85 보다 높게)")
    ap.add_argument("--const-tol", type=float, default=0.5,
                    help="별자리 쌍별 거리 일치 허용(m)")
    ap.add_argument("--bridge", type=int, default=0,
                    help="관측 기근으로 PnP 가 멈추면 등록 프레임에서 이 거리(프레임) 이내를"
                         " v1 상대포즈로 건넌다. 브릿지 포즈도 BA 로 다듬어진다. 0=끔")
    ap.add_argument("--fuse-cos", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cam = json.load(open(os.path.join(sd, "camera_info.json")))
    K = np.array(cam["intrinsics"], float)
    W, H = cam["width"], cam["height"]
    gtp = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    init = np.loadtxt(os.path.join(sd, args.init)).reshape(-1, 4, 4)
    ids = json.load(open(os.path.join(sd, args.seg_ids)))
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]

    keep = set()
    for local, m in ids.items():
        rec = gt.get(str(m.get("gt_instance") or m.get("instance_id")))
        if rec and (rec["motion_type"] == "static" or args.still_dynamic) \
                and not rec.get("moves") \
                and rec.get("extent_m") and max(rec["extent_m"]) <= args.max_extent:
            keep.add(int(local))
    # min_fill 기본 0 (= 가림 필터 끔) 은 실측으로 확정한 값이다. 관측이 프레임당
    # 중앙 7개뿐인 기근 상태라 0.65 를 걸면 씨앗 랜드마크 10→3, 등록 36%→8% 로
    # 붕괴한다(2026-08-15). refine_bootstrap 의 0.65 는 관측이 풍부할 때 얘기다.
    obs = load_obs(sd, args.seg, ids, keep, args.every, W, H, args.min_fill)
    frames = sorted(obs)
    emb = None
    if args.fuse_emb:
        zz = np.load(os.path.join(sd, args.fuse_emb))
        E = zz["emb"] / np.linalg.norm(zz["emb"], axis=1, keepdims=True)
        emb = {int(t): E[i] for i, t in enumerate(zz["ids"])}
        tframes = defaultdict(set)
        for _f in frames:
            for _l in obs[_f]:
                tframes[_l].add(_f)
    print("관측 프레임 %d · 랜드마크 후보 %d · 관측 중앙 %.0f개/프레임"
          % (len(frames), len(keep), np.median([len(v) for v in obs.values()])))

    # --- ① 씨앗: 공통 랜드마크가 많으면서 **시차가 충분한** 구간 -----------------
    # ⚠️ 씨앗을 '연속 N 프레임' 으로 잡으면 안 된다. every=1 에서 12프레임은 1.2초라
    # 카메라가 거의 안 움직여 시차가 0 이고, 삼각측량이 **전부 기각**된다(실측: 씨앗
    # 랜드마크 0개, 등록 12/905). 시간 창으로 잡고 그 안에서 성기게 고른다.
    span = args.seed_span
    best, bseed = -1, None
    for i in range(len(frames)):
        seg = [f for f in frames[i:] if f - frames[i] <= span]
        if len(seg) < 4:
            continue
        seg = seg[::max(1, len(seg) // args.seed_len)][:args.seed_len]
        C = np.array([init[f][:3, 3] for f in seg])
        base = float(np.linalg.norm(C.max(0) - C.min(0)))
        if base < MIN_PARALLAX * 2:
            continue
        sets = [set(obs[f]) for f in seg]
        sc = len(set.intersection(*sets)) * 10 + sum(len(a & b) for a, b in zip(sets, sets[1:]))
        if sc > best:
            best, bseed = sc, seg
    if bseed is None:
        sys.exit("씨앗 구간을 못 찾았다 — seed-span 을 늘려라")
    seed = bseed
    poses = {f: init[f].copy() for f in seed}
    _C = np.array([init[f][:3, 3] for f in seed])
    print("씨앗 %d프레임 f%d~f%d · 기준선 %.2f m · 점수 %d"
          % (len(seed), seed[0], seed[-1],
             float(np.linalg.norm(_C.max(0) - _C.min(0))), best))

    land = {}
    for lid in keep:
        views = [(poses[f], obs[f][lid][:2]) for f in seed if lid in obs[f]]
        if len(views) < MIN_TRI_VIEWS:
            continue
        C = np.array([T[:3, 3] for T, _ in views])
        if np.linalg.norm(C.max(0) - C.min(0)) < MIN_PARALLAX:
            continue
        X = triangulate(views, K)
        if accept_point(X, views, K):
            land[lid] = X
    print("씨앗 랜드마크 %d개" % len(land))
    # 스케일 앵커: 씨앗 양끝 카메라 간 거리. v1 은 국소적으로는 미터가 맞으므로
    # 이 거리 하나로 지도 전체의 스케일을 미터에 묶는다.
    anc = (seed[0], seed[-1],
           float(np.linalg.norm(init[seed[0]][:3, 3] - init[seed[-1]][:3, 3])))
    poses, land, rms = local_ba(poses, land, obs, K, anchor=anc)
    print("씨앗 BA RMS %.2f px" % (rms or -1))

    # --- ②③④ 등록 → 성장 → 교정 -------------------------------------------------
    # ⚠️ 실패한 프레임을 todo 에서 빼면 안 된다. 지도가 작을 때 실패한 프레임이
    # 지도가 커진 뒤에는 풀릴 수 있고, 그게 증분 SfM 의 핵심이다. 대신 **직전 시도 이후
    # 지도가 자란 경우에만** 재시도해서 무한 루프를 막는다.
    n_fuse, n_fuse_ok = 0, 0
    grounded = set(land)                # PnP 뷰 ≥3 으로 삼각측량된 랜드마크 (씨앗 포함)

    def try_fuse(lid, X):
        """새 랜드마크 ↔ 기존 랜드마크 융합 게이트. 사전 병합(v1 위치 기반)은 정밀도
        32~44% 로 못 쓴다(실측) — 지도 3D 는 그보다 정확해서 게이트가 날카로워진다."""
        # ⚠️ 융합은 identity 결정이다 — 브릿지(v1 복사) 포즈로 삼각측량된 점은
        # 위치가 v1 드리프트만큼 틀어져 있어 3D 게이트가 다른 물체를 병합한다
        # (실측: bridge 100 에서 순도 10/10 → 13/26 붕괴, 가짜 루프가 전역 BA 로
        # PnP 구역까지 오염 0.067→0.214). PnP 접지된 점끼리만 융합한다.
        if emb is None or lid not in emb or lid not in grounded:
            return None
        best, bl = -1.0, None
        for l2, X2 in land.items():
            if l2 == lid or l2 not in emb or l2 not in grounded:
                continue
            if np.linalg.norm(X - X2) > args.fuse_dist:
                continue
            c = float(emb[lid] @ emb[l2])
            if c > best:
                best, bl = c, l2
        if bl is None or best < args.fuse_cos or (tframes[lid] & tframes[bl]):
            return None
        return bl

    n_const, n_const_ok = 0, 0

    def merge_into(l, o):
        """l 의 관측 전부(미래 포함)를 o 로 이관."""
        for g in tframes[l]:
            if l in obs[g]:
                obs[g][o] = obs[g].pop(l)
        tframes[o] |= tframes[l]
        land.pop(l, None)

    def constellation(f):
        """드리프트 불변 루프 감지 — 절대좌표 융합 게이트는 재방문 구역이 1m 쯤
        드리프트되면 침묵한다(실측: bridge 100 에서 루프 0회). 랜드마크 **쌍별 상대
        거리**는 국소 포즈만 맞으면 유효하므로, 임베딩 후보쌍들 가운데 상호 거리가
        일치하는 ≥3쌍을 별자리로 보고 통째로 병합한다. 병합이 곧 루프 간선이 되어
        기존 far/전역 BA 경로가 발화한다."""
        import itertools
        cur = [l for l in obs[f] if l in land and l in emb and l not in grounded]
        cand = []
        for l in cur:
            for o in grounded:
                if o not in land or o not in emb or o in obs[f]:
                    continue
                if float(emb[l] @ emb[o]) < args.const_cos or (tframes[l] & tframes[o]):
                    continue
                cand.append((l, o))
        if len(cand) < 3:
            return 0
        con = {i: set() for i in range(len(cand))}
        for (i, (l1, o1)), (j, (l2, o2)) in itertools.combinations(enumerate(cand), 2):
            if l1 == l2 or o1 == o2:
                continue
            dn = np.linalg.norm(land[l1] - land[l2])
            do = np.linalg.norm(land[o1] - land[o2])
            if abs(dn - do) <= args.const_tol:
                con[i].add(j)
                con[j].add(i)
        best = max(con, key=lambda i: len(con[i]))
        group = {best}
        for j in sorted(con[best], key=lambda j: -len(con[j])):
            if all(j in con[g] or g in con[j] for g in group):
                lg = [cand[g][0] for g in group] + [cand[j][0]]
                og = [cand[g][1] for g in group] + [cand[j][1]]
                if len(set(lg)) == len(lg) and len(set(og)) == len(og):
                    group.add(j)
        if len(group) < 3:
            return 0
        nonlocal n_const, n_const_ok
        for gI in group:
            l, o = cand[gI]
            n_const += 1
            n_const_ok += (ids.get(str(l), {}).get("gt_instance")
                           == ids.get(str(o), {}).get("gt_instance"))
            merge_into(l, o)
        return len(group)

    import bisect
    todo = [f for f in frames if f not in poses]
    tried_at = {}                       # frame → 마지막 시도 시점의 지도 크기
    n_reg, n_loop, n_retry, n_bridge = 0, 0, 0, 0
    pnp_frames = set(seed)
    closed = set()
    while todo:
        cand = [(len([l for l in obs[f] if l in land]), f) for f in todo
                if tried_at.get(f, -1) < len(land)]
        cand.sort(reverse=True)
        if cand and cand[0][0] >= 5:
            nvis, f = cand[0]
            n_retry += (f in tried_at)
            tried_at[f] = len(land)
            r = pnp([land[l] for l in obs[f] if l in land],
                    [obs[f][l][:2] for l in obs[f] if l in land], K)
            if r is None:
                continue
            poses[f], _ = r
            pnp_frames.add(f)
            for lid in obs[f]:
                if lid in land and lid not in grounded:
                    if sum(g in pnp_frames for g in poses if lid in obs[g]) >= MIN_TRI_VIEWS:
                        grounded.add(lid)
        elif args.bridge:
            # 프론티어 브릿지 — 삼각측량은 등록을, 등록은 랜드마크를 전제하는
            # 닭-달걀을 v1 상대포즈 한 걸음으로 끊는다(실측: FastSAM 트랙은 수명이
            # 짧아 등록이 씨앗 주변 15초에 갇혔다). 국소 v1 은 믿는다는 설계 그대로.
            regs = sorted(poses)
            best = None
            for f2 in todo:
                k = bisect.bisect_left(regs, f2)
                for g in regs[max(0, k - 1):k + 1]:
                    gap = abs(g - f2)
                    if gap <= args.bridge and (best is None or gap < best[0]):
                        best = (gap, f2, g)
            if best is None:
                break
            _, f, g = best
            poses[f] = poses[g] @ np.linalg.inv(init[g]) @ init[f]
            n_bridge += 1
        else:
            break
        todo.remove(f)
        n_reg += 1

        # ③ 새 프레임 덕에 시차가 생긴 랜드마크를 새로 삼각측량
        for lid in list(obs[f]):
            if lid in land:
                continue
            vf = [g for g in poses if lid in obs[g]]
            views = [(poses[g], obs[g][lid][:2]) for g in vf]
            if len(views) < MIN_TRI_VIEWS:
                continue
            C = np.array([T[:3, 3] for T, _ in views])
            if np.linalg.norm(C.max(0) - C.min(0)) < MIN_PARALLAX:
                continue
            X = triangulate(views, K)
            if not accept_point(X, views, K):
                continue
            if sum(g in pnp_frames for g in vf) >= MIN_TRI_VIEWS:
                grounded.add(lid)
            tgt = try_fuse(lid, X)
            if tgt is None:
                land[lid] = X
            else:                       # 융합: lid 의 모든 관측(미래 포함)을 tgt 로 이관
                n_fuse += 1
                n_fuse_ok += (ids.get(str(lid), {}).get("gt_instance")
                              == ids.get(str(tgt), {}).get("gt_instance"))
                for g in tframes[lid]:
                    if lid in obs[g]:
                        obs[g][tgt] = obs[g].pop(lid)
                tframes[tgt] |= tframes[lid]

        if emb is not None and args.const:
            constellation(f)

        # ④ 루프 판정: 시간적으로 먼 프레임과 랜드마크를 공유하는가.
        # 같은 루프(LOOP_GAP 버킷 쌍)는 **처음 닫힐 때 한 번만** 전역 BA 를 돈다 —
        # 안 그러면 지도가 시퀀스 전체로 퍼진 뒤 거의 매 등록이 전역 BA 가 되어
        # 비용이 프레임 수 제곱으로 자란다(실측: every 5 에서 66등록 중 39회).
        far = [g for g in poses if abs(g - f) > LOOP_GAP and (set(obs[g]) & set(obs[f]))]
        keys = {(min(f, g) // LOOP_GAP, max(f, g) // LOOP_GAP) for g in far} - closed
        if keys:
            closed |= keys
            n_loop += 1
            poses, land, rms = local_ba(poses, land, obs, K, anchor=anc)   # 전역
        elif n_reg % (3 * args.ba_every) == 0:
            # 루프 억제(④)로 전역 BA 가 39→4회로 줄자 꼬리 오차가 늘었다
            # (p90 0.705→0.933). 주기적 전역 1회가 그 중간을 잡는다.
            poses, land, rms = local_ba(poses, land, obs, K, anchor=anc)
        elif n_reg % args.ba_every == 0:
            recent = sorted(poses)[-3 * args.ba_every:]
            poses, land, rms = local_ba(poses, land, obs, K, fids=recent)

    # BA 로 움직인 뒤 여전히 설명 안 되는 랜드마크는 지도에서 뺀다
    bad = [l for l, X in land.items()
           if not accept_point(X, [(poses[f], obs[f][l][:2]) for f in poses if l in obs[f]],
                               K, max_reproj=10.0)]
    for l in bad:
        del land[l]
    if bad:
        print("정리: 재투영이 안 맞는 랜드마크 %d개 제거" % len(bad))
    poses, land, rms = local_ba(poses, land, obs, K, max_nfev=200, anchor=anc)
    med, p9, sc = ate(poses, gtp)
    if len(poses) < 0.3 * len(frames):
        print("\n⚠️ 등록률이 낮아 ATE 가 무의미하다 — 등록된 프레임이 한쪽에 몰리면"
              "\n   sim3 정합이 거의 완벽히 맞아 0.0x m 같은 허상이 나온다.")
    print("\n등록 %d/%d 프레임 (%.0f%%, PnP %d + 브릿지 %d) · 랜드마크 %d · 재시도 %d"
          " · 루프 교정 %d회 · RMS %.2f px"
          % (len(poses), len(frames), 100 * len(poses) / len(frames),
             len(pnp_frames), n_bridge, len(land), n_retry, n_loop, rms or -1))
    if emb is not None:
        print("융합 %d회 (옳음 %d) · 별자리 루프병합 %d회 (옳음 %d)"
              % (n_fuse, n_fuse_ok, n_const, n_const_ok))
    print("**ATE 중앙 %.3f m · p90 %.3f · 정합스케일 %.3f**" % (med, p9, sc))
    if n_bridge and len(pnp_frames) > 10:
        m2, p2, _ = ate({f: poses[f] for f in pnp_frames}, gtp)
        print("  (PnP 프레임만: ATE 중앙 %.3f · p90 %.3f)" % (m2, p2))

    if args.out:
        arr = np.tile(np.eye(4), (len(gtp), 1, 1))
        for i, T in poses.items():
            arr[i] = T
        np.savetxt(args.out, arr.reshape(len(gtp), 16))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
