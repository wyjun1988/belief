#!/usr/bin/env python3
"""물체 랜드마크로 **증분 SfM** — 지도를 키우면서 앞뒤를 되돌아 교정한다.

    $P scripts/incremental_sfm.py --seq <name>

배치 정제(refine_bootstrap.py)는 v1 스티칭 초기값에서 출발해 국소 최소점에 갇혔다
(가중치를 60배 바꿔도 결과가 소수점까지 동일했다). 틀린 랜드맵과 틀린 포즈가 서로를
정당화하기 때문이다.

여기서는 구조를 바꾼다 — **v1 을 국소적으로만 믿는다**:

    ① 씨앗    공통 랜드마크가 많고 시차가 충분한 **짧은 구간**만 v1 포즈로 초기화
    ② 등록    지도에 있는 랜드마크를 가장 많이 보는 프레임부터 하나씩 PnP 로 붙인다
    ③ 성장    새로 등록된 프레임 덕에 시차가 생긴 랜드마크를 **새로 삼각측량**
    ④ 교정    N 개마다 지역 BA. 루프가 처음 닫히면 **전역 BA**
    ⑤ 아일랜드 관측 사막에 막히면(--islands) 남은 구간에 새 씨앗을 심어 독립 성장,
              마지막에 RANSAC sim3 로 병합 — 사막을 v1 오도메트리로 건너지 않는다

FastSAM 트랙 실측(2026-08-16): 연결 구역 PnP-only 0.067 m, 사막을 v1 브릿지로
건너면 드리프트 수입(0.28~), 별자리 루프병합은 18/18 오병합. 아일랜드 병합은
접지된 강체 구조끼리의 정합이라 조건이 다르다.
"""
import argparse
import bisect
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
PIECE = 100000           # (c) 이동물체 구간별 가상 id: lid + PIECE*(구간+1)
MOVE_BUF = 10            # 기본 완충. GT 이동구간은 변위 임계 검출이라 경계에 잔여 움직임이 샌다


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

    # anchor: (fa,fb,d0) 하나 또는 그 리스트. 다중 앵커는 구간별 스케일 구속 —
    # v1 유래 불균일 스케일은 앵커 하나(씨앗)로는 못 편다(실측: 전역 ATE 0.76 정체).
    anc = []
    if not frozen and anchor is not None:
        alist = anchor if isinstance(anchor, list) else [anchor]
        for fa, fb, d0 in alist:
            if fa in fi and fb in fi and d0 > 1e-3:
                anc.append((fi[fa], fi[fb], d0))

    def center(x, i):
        R = cv2.Rodrigues(x[i * 6:i * 6 + 3])[0]
        return -R.T @ x[i * 6 + 3:i * 6 + 6]

    nres = len(rows) * 2 + len(anc)

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
        for kk, (ia, ib, d0) in enumerate(anc):
            out[len(rows) * 2 + kk] = w_scale * (
                np.linalg.norm(center(x, ia) - center(x, ib)) - d0)
        return out

    S = lil_matrix((nres, len(x0)), dtype=int)
    for k, (c, p, _, _) in enumerate(rows):
        if c < nF:                              # 동결 카메라 열은 비워 둔다
            S[2 * k:2 * k + 2, c * 6:c * 6 + 6] = 1
        S[2 * k:2 * k + 2, F * 6 + p * 3:F * 6 + p * 3 + 3] = 1
    for kk, (ia, ib, _) in enumerate(anc):
        for i in (ia, ib):
            S[len(rows) * 2 + kk, i * 6:i * 6 + 6] = 1
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


def umeyama3(A, B):
    """B ≈ s·R·A + t 를 푸는 최소 구현(표본 3점 이상). RANSAC 내부용."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    U, D, Vt = np.linalg.svd(Bc.T @ Ac)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float((D * S.diagonal()).sum() / max((Ac ** 2).sum(), 1e-12))
    t = cb - s * R @ ca
    return s, R, t


def sim3_apply_pose(T, s, R, t):
    """카메라→월드 포즈에 sim3 적용. 회전엔 s 를 곱하지 않는다."""
    O = np.eye(4)
    O[:3, :3] = R @ T[:3, :3]
    O[:3, 3] = s * R @ T[:3, 3] + t
    return O


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
    ap.add_argument("--fuse-cos", type=float, default=0.85)
    ap.add_argument("--still-dynamic", action="store_true",
                    help="motion_type 이 동적이라도 이 시퀀스에서 안 움직였으면(moves=False)"
                         " 랜드마크로 쓴다. 접시·소품류가 늘어 관측 밀도가 오른다")
    ap.add_argument("--move-buf", type=int, default=MOVE_BUF,
                    help="이동구간 앞뒤 완충(프레임). every=1 은 경계 오염 관측을 5배 줍므로"
                         " 넓게 잡아야 한다(실측: buf10 에서 ATE 0.170→0.210 역효과)")
    ap.add_argument("--piecewise", action="store_true",
                    help="(c) 이동물체를 이동구간 기준으로 쪼개 **정지구간마다** 별도"
                         " 랜드마크로 쓴다. 관측 사막이 이동물체 때문에 생겼을 때 뚫린다")
    ap.add_argument("--islands", type=int, default=1,
                    help="(b) 씨앗 수. 1=단일(기존). 2+ 는 사막에 막힐 때마다 남은 구간에"
                         " 새 씨앗을 심어 독립 성장 후 RANSAC sim3 로 병합한다")
    ap.add_argument("--merge-cos", type=float, default=0.85, help="아일랜드 병합 임베딩 하한")
    ap.add_argument("--merge-tol", type=float, default=0.4, help="아일랜드 병합 인라이어 허용(m)")
    ap.add_argument("--bridge", type=int, default=0,
                    help="관측 기근으로 PnP 가 멈추면 등록 프레임에서 이 거리(프레임) 이내를"
                         " v1 상대포즈로 건넌다. 브릿지 포즈도 BA 로 다듬어진다. 0=끔")
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

    keep, movers = set(), {}
    for local, m in ids.items():
        rec = gt.get(str(m.get("gt_instance") or m.get("instance_id")))
        if not rec or not rec.get("extent_m") or max(rec["extent_m"]) > args.max_extent:
            continue
        if (rec["motion_type"] == "static" or args.still_dynamic) and not rec.get("moves"):
            keep.add(int(local))
        elif args.piecewise and rec.get("moves"):
            movers[int(local)] = [(mv["start_idx"] - args.move_buf, mv["end_idx"] + args.move_buf)
                                  for mv in rec["moves"]]
    keep |= set(movers)

    def gid(l):
        """가상 id(구간별 이동물체 포함) → gt_instance. 순도 진단용."""
        return ids.get(str(l % PIECE), {}).get("gt_instance")

    # min_fill 기본 0 (= 가림 필터 끔) 은 실측으로 확정한 값이다. 관측이 프레임당
    # 중앙 7개뿐인 기근 상태라 0.65 를 걸면 씨앗 랜드마크 10→3, 등록 36%→8% 로
    # 붕괴한다(2026-08-15). refine_bootstrap 의 0.65 는 관측이 풍부할 때 얘기다.
    obs = load_obs(sd, args.seg, ids, keep, args.every, W, H, args.min_fill)
    if movers:
        # (c) 이동구간(±완충) 관측은 버리고, 정지구간별로 가상 id 를 부여한다.
        n_piece = 0
        for f in list(obs):
            for l in list(obs[f]):
                if l not in movers:
                    continue
                iv = movers[l]
                if any(a <= f <= b for a, b in iv):
                    del obs[f][l]
                else:
                    seg_i = sum(f > b for _, b in iv)
                    obs[f][l + PIECE * (seg_i + 1)] = obs[f].pop(l)
                    n_piece += 1
        print("구간별 이동물체 %d개 → 관측 %d개를 정지구간 랜드마크로" % (len(movers), n_piece))
    frames = sorted(obs)

    emb = None
    if args.fuse_emb:
        zz = np.load(os.path.join(sd, args.fuse_emb))
        E = zz["emb"] / np.linalg.norm(zz["emb"], axis=1, keepdims=True)
        emb = {int(t): E[i] for i, t in enumerate(zz["ids"])}
        for l in {l for f in frames for l in obs[f] if l >= PIECE}:
            if l % PIECE in emb:
                emb[l] = emb[l % PIECE]
    tframes = defaultdict(set)
    for _f in frames:
        for _l in obs[_f]:
            tframes[_l].add(_f)
    print("관측 프레임 %d · 랜드마크 후보 %d · 관측 중앙 %.0f개/프레임"
          % (len(frames), len(keep - set(movers)) + sum(len(v) + 1 for v in movers.values()),
             np.median([len(v) for v in obs.values()])))

    # --- ① 씨앗: 공통 랜드마크가 많으면서 **시차가 충분한** 구간 -----------------
    # ⚠️ 씨앗을 '연속 N 프레임' 으로 잡으면 안 된다. every=1 에서 12프레임은 1.2초라
    # 카메라가 거의 안 움직여 시차가 0 이고, 삼각측량이 **전부 기각**된다(실측: 씨앗
    # 랜드마크 0개, 등록 12/905). 시간 창으로 잡고 그 안에서 성기게 고른다.
    def pick_seed(taken):
        # ⚠️ 점수만 보고 고르면 씨앗이 기존 아일랜드 옆에 뭉친다(실측: 4개가 전부
        # f357~551). 탐사를 강제한다 — 기존 등록 구역에서 SEED_SEP 이상 떨어진
        # 창만 후보로 삼는다.
        SEED_SEP = 50
        tsort = sorted(taken)
        best, bseed = -1, None
        for i in range(len(frames)):
            f0 = frames[i]
            if f0 in taken:
                continue
            if tsort:
                k = bisect.bisect_left(tsort, f0)
                near = min([abs(f0 - tsort[j]) for j in (k - 1, k) if 0 <= j < len(tsort)])
                if near < SEED_SEP:
                    continue
            seg = [f for f in frames[i:] if f - f0 <= args.seed_span and f not in taken]
            if len(seg) < 4:
                continue
            seg = seg[::max(1, len(seg) // args.seed_len)][:args.seed_len]
            C = np.array([init[f][:3, 3] for f in seg])
            if float(np.linalg.norm(C.max(0) - C.min(0))) < MIN_PARALLAX * 2:
                continue
            sets = [set(obs[f]) for f in seg]
            sc = len(set.intersection(*sets)) * 10 \
                + sum(len(a & b) for a, b in zip(sets, sets[1:]))
            if sc > best:
                best, bseed = sc, seg
        return bseed, best

    stats = dict(fuse=0, fuse_ok=0, retry=0, loop=0, bridge=0)

    def grow_island(seed, taken):
        """씨앗 하나에서 ②③④ 로 아일랜드를 키운다. taken 프레임은 건드리지 않는다."""
        poses = {f: init[f].copy() for f in seed}
        pnp_frames = set(seed)
        land, grounded = {}, set()
        anc = (seed[0], seed[-1],
               float(np.linalg.norm(init[seed[0]][:3, 3] - init[seed[-1]][:3, 3])))
        for lid in {l for f in seed for l in obs[f]}:
            views = [(poses[f], obs[f][lid][:2]) for f in seed if lid in obs[f]]
            if len(views) < MIN_TRI_VIEWS:
                continue
            C = np.array([T[:3, 3] for T, _ in views])
            if np.linalg.norm(C.max(0) - C.min(0)) < MIN_PARALLAX:
                continue
            X = triangulate(views, K)
            if accept_point(X, views, K):
                land[lid] = X
        grounded |= set(land)
        local_ba(poses, land, obs, K, anchor=anc)

        def try_fuse(lid, X):
            """융합은 identity 결정 — 브릿지(v1 복사) 포즈 유래의 점은 위치가 드리프트
            만큼 틀어져 오병합을 만든다(실측: 순도 10/10→13/26). PnP 접지끼리만."""
            if emb is None or lid not in emb or lid not in grounded:
                return None
            best_c, bl = -1.0, None
            for l2, X2 in land.items():
                if l2 == lid or l2 not in emb or l2 not in grounded:
                    continue
                if np.linalg.norm(X - X2) > args.fuse_dist:
                    continue
                c = float(emb[lid] @ emb[l2])
                if c > best_c:
                    best_c, bl = c, l2
            if bl is None or best_c < args.fuse_cos or (tframes[lid] & tframes[bl]):
                return None
            return bl

        todo = [f for f in frames if f not in poses and f not in taken]
        tried_at = {}
        n_reg = 0
        closed = set()
        while todo:
            cand = [(len([l for l in obs[f] if l in land]), f) for f in todo
                    if tried_at.get(f, -1) < len(land)]
            cand.sort(reverse=True)
            if cand and cand[0][0] >= 5:
                nvis, f = cand[0]
                stats["retry"] += (f in tried_at)
                tried_at[f] = len(land)
                r = pnp([land[l] for l in obs[f] if l in land],
                        [obs[f][l][:2] for l in obs[f] if l in land], K)
                if r is None:
                    continue
                poses[f], _ = r
                pnp_frames.add(f)
                for lid in obs[f]:
                    if lid in land and lid not in grounded and \
                            sum(g in pnp_frames for g in poses if lid in obs[g]) >= MIN_TRI_VIEWS:
                        grounded.add(lid)
            elif args.bridge:
                # 프론티어 브릿지 — 닭-달걀(삼각측량↔등록)을 v1 한 걸음으로 끊는다.
                # 아일랜드 모드에선 짧게(≤15) 두고, 긴 사막은 새 씨앗이 맡는다.
                regs = sorted(poses)
                best_b = None
                for f2 in todo:
                    k = bisect.bisect_left(regs, f2)
                    for g in regs[max(0, k - 1):k + 1]:
                        gap = abs(g - f2)
                        if gap <= args.bridge and (best_b is None or gap < best_b[0]):
                            best_b = (gap, f2, g)
                if best_b is None:
                    break
                _, f, g = best_b
                poses[f] = poses[g] @ np.linalg.inv(init[g]) @ init[f]
                stats["bridge"] += 1
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
                else:
                    stats["fuse"] += 1
                    stats["fuse_ok"] += (gid(lid) == gid(tgt))
                    for g in tframes[lid]:
                        if lid in obs[g]:
                            obs[g][tgt] = obs[g].pop(lid)
                    tframes[tgt] |= tframes[lid]

            # ④ 루프: 같은 버킷 쌍은 처음 닫힐 때 한 번만 전역 BA (비용 제곱 방지)
            far = [g for g in poses if abs(g - f) > LOOP_GAP and (set(obs[g]) & set(obs[f]))]
            keys = {(min(f, g) // LOOP_GAP, max(f, g) // LOOP_GAP) for g in far} - closed
            if keys:
                closed |= keys
                stats["loop"] += 1
                local_ba(poses, land, obs, K, anchor=anc)
            elif n_reg % (3 * args.ba_every) == 0:
                local_ba(poses, land, obs, K, anchor=anc)
            elif n_reg % args.ba_every == 0:
                recent = sorted(poses)[-3 * args.ba_every:]
                local_ba(poses, land, obs, K, fids=recent)
        return dict(poses=poses, land=land, pnp=pnp_frames, anc=anc,
                    grounded=grounded, seed=seed)

    # --- ⑤ 아일랜드 성장 ---------------------------------------------------------
    islands, taken = [], set()
    for k in range(max(1, args.islands)):
        seed, score = pick_seed(taken)
        if seed is None:
            break
        isl = grow_island(seed, taken)
        taken |= set(isl["poses"])
        r0, r1 = min(isl["poses"]), max(isl["poses"])
        print("아일랜드 %d: 씨앗 f%d~f%d(점수 %d) → 등록 %d (f%d~f%d) · 랜드마크 %d"
              % (k, seed[0], seed[-1], score, len(isl["poses"]), r0, r1, len(isl["land"])))
        islands.append(isl)
        if len(taken) >= 0.95 * len(frames):
            break
    islands.sort(key=lambda I: -len(I["poses"]))

    # --- 아일랜드 병합: 공유 트랙 + 임베딩 후보쌍 위 RANSAC sim3 -------------------
    # 별자리(프레임 단위 3쌍, 18/18 오병합)와 달리 여기는 **접지된 강체 구조 전체**를
    # 놓고 하나의 sim3 를 검증한다. sim3 는 아일랜드별 v1 스케일 차이를 흡수한다.
    A = islands[0]
    n_merged, n_inl_ok, n_inl_all = 0, 0, 0
    unmerged = []
    for B in islands[1:]:
        pairs = [(l, l) for l in set(A["land"]) & set(B["land"])]
        if emb is not None:
            for lb, Xb in B["land"].items():
                if lb not in emb or lb not in B["grounded"]:
                    continue
                for la, Xa in A["land"].items():
                    if la in (p[1] for p in pairs) or la not in emb \
                            or la not in A["grounded"]:
                        continue
                    if float(emb[lb] @ emb[la]) >= args.merge_cos \
                            and not (tframes[lb] & tframes[la]):
                        pairs.append((lb, la))
        if len(pairs) < 3:
            print("병합 실패(후보 %d<3): 아일랜드 f%d~ 는 v1 사전으로 합류"
                  % (len(pairs), min(B["poses"])))
            unmerged.append(B)
            continue
        PB = np.array([B["land"][lb] for lb, _ in pairs])
        PA = np.array([A["land"][la] for _, la in pairs])
        rng = np.random.default_rng(0)
        best_inl, best_T = [], None
        for _ in range(min(2000, len(pairs) ** 3)):
            i3 = rng.choice(len(pairs), 3, replace=False)
            if len({pairs[i][0] for i in i3}) < 3 or len({pairs[i][1] for i in i3}) < 3:
                continue
            s, R, t = umeyama3(PB[i3], PA[i3])
            if not (0.5 <= s <= 2.0):
                continue
            d = np.linalg.norm((s * (R @ PB.T)).T + t - PA, axis=1)
            inl = np.nonzero(d <= args.merge_tol)[0]
            # 인라이어는 일대일이어야 한다
            seen_l, seen_r, uniq = set(), set(), []
            for i in inl[np.argsort(d[inl])]:
                lb, la = pairs[i]
                if lb in seen_l or la in seen_r:
                    continue
                seen_l.add(lb)
                seen_r.add(la)
                uniq.append(i)
            if len(uniq) > len(best_inl):
                best_inl, best_T = uniq, (s, R, t)
        if len(best_inl) < 4:
            print("병합 실패(인라이어 %d<4): 아일랜드 f%d~ 는 v1 사전으로 합류"
                  % (len(best_inl), min(B["poses"])))
            unmerged.append(B)
            continue
        s, R, t = umeyama3(PB[best_inl], PA[best_inl])
        ok = sum(gid(pairs[i][0]) == gid(pairs[i][1]) for i in best_inl)
        n_inl_ok += ok
        n_inl_all += len(best_inl)
        print("병합: f%d~ 아일랜드, 후보 %d → 인라이어 %d (GT 일치 %d) · 스케일 %.3f"
              % (min(B["poses"]), len(pairs), len(best_inl), ok, s))
        for f, T in B["poses"].items():
            A["poses"][f] = sim3_apply_pose(T, s, R, t)
        A["pnp"] |= B["pnp"]
        for lb, Xb in B["land"].items():
            A["land"].setdefault(lb, s * R @ Xb + t)
        A["grounded"] |= B["grounded"]
        for i in best_inl:
            lb, la = pairs[i]
            if lb == la:
                continue
            for g in tframes[lb]:
                if lb in obs[g]:
                    obs[g][la] = obs[g].pop(lb)
            tframes[la] |= tframes[lb]
            A["land"].pop(lb, None)
        n_merged += 1

    poses, land, anc = A["poses"], A["land"], A["anc"]
    # BA 로 움직인 뒤 여전히 설명 안 되는 랜드마크는 지도에서 뺀다
    bad = [l for l, X in land.items()
           if not accept_point(X, [(poses[f], obs[f][l][:2]) for f in poses if l in obs[f]],
                               K, max_reproj=10.0)]
    for l in bad:
        del land[l]
    if bad:
        print("정리: 재투영이 안 맞는 랜드마크 %d개 제거" % len(bad))
    poses, land, rms = local_ba(poses, land, obs, K, max_nfev=200, anchor=anc)
    core = dict(poses)
    # 병합 실패 아일랜드는 **최종 BA 뒤에** 항등으로 합류한다 — 전부 같은 v1 세계좌표에
    # 씨앗-앵커돼 있어 항등이 1차 근사고, BA 에 넣으면 비연결 성분이 게이지를 흔든다.
    for B in unmerged:
        for f, T in B["poses"].items():
            poses.setdefault(f, T)
        A["pnp"] |= B["pnp"]
    med, p9, sc = ate(poses, gtp)
    if len(poses) < 0.3 * len(frames):
        print("\n⚠️ 등록률이 낮아 ATE 가 무의미하다 — 등록된 프레임이 한쪽에 몰리면"
              "\n   sim3 정합이 거의 완벽히 맞아 0.0x m 같은 허상이 나온다.")
    print("\n등록 %d/%d 프레임 (%.0f%%, PnP %d + 브릿지 %d) · 랜드마크 %d · 재시도 %d"
          " · 루프 교정 %d회 · 아일랜드 %d(병합 %d) · RMS %.2f px"
          % (len(poses), len(frames), 100 * len(poses) / len(frames),
             len(A["pnp"]), stats["bridge"], len(land), stats["retry"],
             stats["loop"], len(islands), n_merged, rms or -1))
    if emb is not None:
        print("융합 %d회 (옳음 %d) · 병합 인라이어 %d (옳음 %d)"
              % (stats["fuse"], stats["fuse_ok"], n_inl_all, n_inl_ok))
    print("**ATE 중앙 %.3f m · p90 %.3f · 정합스케일 %.3f**" % (med, p9, sc))
    if unmerged and len(core) > 10:
        m3, p3, _ = ate(core, gtp)
        print("  (병합핵만 %d프레임: ATE 중앙 %.3f · p90 %.3f — 합집합에는 v1 사전"
              " 합류분 %d프레임 포함)" % (len(core), m3, p3, len(poses) - len(core)))
    if stats["bridge"] and len(A["pnp"]) > 10:
        m2, p2, _ = ate({f: poses[f] for f in A["pnp"]}, gtp)
        print("  (PnP 프레임만: ATE 중앙 %.3f · p90 %.3f)" % (m2, p2))

    if args.out:
        arr = np.tile(np.eye(4), (len(gtp), 1, 1))
        for i, T in poses.items():
            arr[i] = T
        np.savetxt(args.out, arr.reshape(len(gtp), 16))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
