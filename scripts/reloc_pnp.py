#!/usr/bin/env python3
"""한 장 조회형 재국소화 — 스캔 구간으로 지도를 한 번 세우고(COLMAP, 무거워도 됨), 라이브 프레임은 **한 장씩 PnP** 로 붙인다.
    python scripts/reloc_pnp.py data/seq/Apartment_release_decoration_seq137_M1292 --scan-end 550 --live-step 10 --work ~/khcache/reloc-adt137
왜: 1fps 구간에 SfM(증분 재구성)을 돌리지 않는다(사용자 결정 2026-09-05). 라이브는 지도의 3D 점에 대한 조회일 뿐이라
프레임 사이 겹침·정지 여부와 무관하고, 후보 프레임에만 돌리면 된다. 여기서는 (1) 실사에서 스캔 지도가 서는가
(2) 한 장 PnP 의 오차 분포 (3) 장당 비용 을 GT(ADT: MPS 포즈) 대비로 잰다 — 시뮬 GT+노이즈 허용의 근거.
입력 규약: <seq>/rgb/%06d.jpg · <seq>/pose/poses.txt(T_world_camera 4x4 행) · <seq>/camera_info.json(fx fy cx cy) — kx/adt/export.py 산출.
지도↔GT 정렬은 스캔 프레임 카메라 중심의 sim3(RANSAC) — 스캔 단계 GT 라 허용. 라이브 오차는 그 변환을 적용해 GT 중심과 비교."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kx.depth.pose_stitch import umeyama  # noqa
import pycolmap  # noqa
ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("--scan-end", type=int, default=550, help="이 인덱스 미만 = 스캔(지도), 이상 = 라이브")
ap.add_argument("--scan-step", type=int, default=1); ap.add_argument("--live-step", type=int, default=1)
ap.add_argument("--work", required=True); ap.add_argument("--features", type=int, default=4096); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--vocab", default=os.path.expanduser("~/khcache/colmap/vocab_tree_faiss_flickr100K_words32K.bin")); ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--min-inliers", type=int, default=12); ap.add_argument("--redo", action="store_true")
a = ap.parse_args(); T0 = time.time()
def log(*x): print("[%5.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)
seq = a.seq.rstrip("/"); os.makedirs(a.work, exist_ok=True)
ci = json.load(open(os.path.join(seq, "camera_info.json"))); W, H = ci["width"], ci["height"]; fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
P = np.loadtxt(os.path.join(seq, "pose", "poses.txt")).reshape(-1, 4, 4)           # T_world_camera
rgbs = sorted(f for f in os.listdir(os.path.join(seq, "rgb")) if f.endswith(".jpg")); n = min(len(rgbs), len(P))
scan = ["rgb/" + rgbs[i] for i in range(0, min(a.scan_end, n), a.scan_step)]; live = ["rgb/" + rgbs[i] for i in range(a.scan_end, n, a.live_step)]
idx = {("rgb/" + rgbs[i]): i for i in range(n)}
log("스캔 %d장 · 라이브 %d장 (전체 %d, 분할 %d)" % (len(scan), len(live), n, a.scan_end))
db = os.path.join(a.work, "db.db"); mk = os.path.join(a.work, ".matched")
if a.redo and os.path.exists(db): os.remove(db); [os.remove(f) for f in (mk,) if os.path.exists(f)]
if not os.path.exists(mk):
    if os.path.exists(db): os.remove(db)
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = "%g,%g,%g,%g" % (fx, fy, cx, cy)
    eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = a.threads; so = pycolmap.SiftExtractionOptions(); so.max_num_features = a.features; eo.sift = so
    pycolmap.extract_features(db, seq, image_names=scan + live, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="PINHOLE", reader_options=ro, extraction_options=eo)
    log("SIFT 추출 %d장" % (len(scan) + len(live)))
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = a.threads
    sq = pycolmap.SequentialPairingOptions(); sq.overlap = 8; sq.loop_detection = False
    pycolmap.match_sequential(db, matching_options=mo, pairing_options=sq); log("순차 매칭")
    vo = pycolmap.VocabTreePairingOptions(); vo.vocab_tree_path = a.vocab; vo.num_images = a.topk; vo.num_threads = a.threads
    pycolmap.match_vocabtree(db, matching_options=mo, pairing_options=vo); log("vocab-tree 매칭(top%d)" % a.topk)
    open(mk, "w").close()
# ── 1) 스캔 지도 (한 번, 무거워도 됨) ──
rec_dir = os.path.join(a.work, "rec_scan"); os.makedirs(rec_dir, exist_ok=True)
subs = [d for d in sorted(os.listdir(rec_dir)) if os.path.isdir(os.path.join(rec_dir, d))]
if subs and not a.redo:
    recs = {d: pycolmap.Reconstruction(os.path.join(rec_dir, d)) for d in subs}
else:
    o = pycolmap.IncrementalPipelineOptions(); o.num_threads = a.threads; o.image_names = scan; o.multiple_models = False
    o.ba_refine_focal_length = False; o.ba_refine_principal_point = False; o.ba_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, seq, rec_dir, o)
rec = max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None
if rec is None: sys.exit("스캔 지도 실패")
log("스캔 지도: 등록 %d/%d · 점 %d · 재투영 %.2fpx" % (rec.num_reg_images(), len(scan), rec.num_points3D(), rec.compute_mean_reprojection_error()))
# 지도 → GT sim3 (스캔 프레임 중심, RANSAC)
X, Y = [], []
for im in rec.images.values():
    if im.has_pose and im.name in idx: X.append(np.asarray(im.projection_center(), float)); Y.append(P[idx[im.name]][:3, 3])
X, Y = np.array(X), np.array(Y)
def ransac_sim3(X, Y, th=0.3, iters=500, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(iters):
        i = rng.choice(len(X), 3, replace=False)
        try: s, R, t = umeyama(X[i], Y[i])
        except Exception: continue
        e = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1); inl = e < th
        if best is None or inl.sum() > best.sum(): best = inl
    s, R, t = umeyama(X[best], Y[best]); e = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    return s, R, t, float(best.mean()), float(np.median(e))
s, R, t, inl, med = ransac_sim3(X, Y)
E0 = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
log("지도↔GT sim3: 인라이어(0.3m) %.2f · 오차 분위 25/50/75/90 = %.3f/%.3f/%.3f/%.3fm · <0.1m %.2f <0.3m %.2f <1m %.2f · 스케일 %.3f  ← 실사에서 지도가 접혔는지" % (
    inl, *np.quantile(E0, [.25, .5, .75, .9]), (E0 < .1).mean(), (E0 < .3).mean(), (E0 < 1).mean(), s))
# 어긋난 스캔 프레임이 시간축 어디인지 (드리프트 vs 접힘)
names_reg = [im.name for im in rec.images.values() if im.has_pose and im.name in idx]; fr_idx = np.array([idx[nm_] for nm_ in names_reg])
bad = fr_idx[E0 > 0.3]
if len(bad): log("  >0.3m 프레임 %d개: 인덱스 범위 %d~%d · 중앙 %d (연속 구간이면 드리프트/분절, 흩어져 있으면 접힘)" % (len(bad), bad.min(), bad.max(), int(np.median(bad))))
# ── 2) 라이브 한 장씩 PnP: 이 프레임의 매칭 상대 중 지도에 등록된 프레임에서 2D-3D 대응을 모아 PnP ──
dbh = pycolmap.Database.open(db)                                                   # pycolmap 3.13: 정적 팩토리
name2id = {im.name: im.image_id for im in dbh.read_all_images()}
reg = {im.image_id: im for im in rec.images.values() if im.has_pose}
kp_cache = {}
def kps(iid):
    if iid not in kp_cache: kp_cache[iid] = dbh.read_keypoints(iid)[:, :2]
    return kp_cache[iid]
p3_of = {}
for iid, im in reg.items():
    m = {}
    for k, p2 in enumerate(im.points2D):
        if p2.has_point3D(): m[k] = rec.points3D[p2.point3D_id].xyz
    p3_of[iid] = m
cam = next(iter(rec.cameras.values()))                                             # SINGLE 카메라 모드 — 스캔 재구성의 카메라를 그대로
eo_ = pycolmap.AbsolutePoseEstimationOptions()
try: eo_.ransac.max_error = 6.0
except Exception: pass
rows = []; tpnp = []
for nm in live:
    qid = name2id.get(nm)
    if qid is None: continue
    t1 = time.time(); pts2, pts3 = [], []
    for mid in reg:
        try: tvg = dbh.read_two_view_geometry(qid, mid)
        except Exception: continue
        M = np.asarray(tvg.inlier_matches)
        if M is None or len(M) == 0: continue
        qk, mk_ = M[:, 0], M[:, 1]                                              # pycolmap: 요청 순서(qid, mid)대로 열이 온다
        q2 = kps(qid); mp3 = p3_of[mid]; nq = len(q2)
        if len(qk) and (qk.max() >= nq):                                         # 반대 순서로 왔으면 뒤집는다
            qk, mk_ = mk_, qk
        for a_, b_ in zip(qk, mk_):
            if int(a_) < nq and int(b_) in mp3: pts2.append(q2[int(a_)]); pts3.append(mp3[int(b_)])
    ok = len(pts2) >= a.min_inliers; res = None
    if ok:
        res = pycolmap.estimate_and_refine_absolute_pose(np.asarray(pts2, float), np.asarray(pts3, float), cam, eo_)
        ok = res is not None and res["num_inliers"] >= a.min_inliers
    dt = time.time() - t1; tpnp.append(dt)
    gtc = P[idx[nm]][:3, 3]
    if ok:
        cfw = res["cam_from_world"]; c = -(cfw.rotation.matrix().T @ cfw.translation)     # 카메라 중심(지도 좌표)
        c_al = s * (R @ c) + t; err = float(np.linalg.norm(c_al - gtc))
        # yaw: 광축 z 를 세계로
        fwd = cfw.rotation.matrix().T @ np.array([0, 0, 1.0]); fwd_al = R @ fwd; gtf = P[idx[nm]][:3, :3] @ np.array([0, 0, 1.0])
        ang = float(np.degrees(np.arccos(np.clip(np.dot(fwd_al, gtf) / (np.linalg.norm(fwd_al) * np.linalg.norm(gtf) + 1e-9), -1, 1))))
        rows.append(dict(name=nm, ok=True, n2d3d=len(pts2), inl=int(res["num_inliers"]), err=err, ang=ang, sec=dt))
    else:
        rows.append(dict(name=nm, ok=False, n2d3d=len(pts2), inl=int(res["num_inliers"]) if res else 0, err=None, ang=None, sec=dt))
okr = [r for r in rows if r["ok"]]; E = np.array([r["err"] for r in okr]); A = np.array([r["ang"] for r in okr])
log("라이브 PnP: 등록 %d/%d (%.2f) · 위치 오차 중앙 %.3fm · <0.5m %.2f · <1m %.2f · 방향 오차 중앙 %.1f° · <10° %.2f · PnP 장당 %.0fms(중앙) · 2D-3D 대응 중앙 %d" % (
    len(okr), len(rows), len(okr) / max(len(rows), 1), np.median(E) if len(E) else -1, (E < 0.5).mean() if len(E) else 0, (E < 1).mean() if len(E) else 0,
    np.median(A) if len(A) else -1, (A < 10).mean() if len(A) else 0, 1000 * np.median(tpnp) if tpnp else -1, int(np.median([r["n2d3d"] for r in rows])) if rows else -1))
json.dump(dict(seq=os.path.basename(seq), n_scan=len(scan), map_reg=rec.num_reg_images(), map_pts=rec.num_points3D(), map_inl=inl, map_med=med, scale=s,
               n_live=len(rows), live_reg=len(okr), err_med=float(np.median(E)) if len(E) else None, lt05=float((E < 0.5).mean()) if len(E) else None,
               lt1=float((E < 1).mean()) if len(E) else None, ang_med=float(np.median(A)) if len(A) else None, pnp_ms=1000 * float(np.median(tpnp)) if tpnp else None,
               rows=rows, sec=time.time() - T0), open(os.path.join(a.work, "summary.json"), "w"), indent=1)
log("→ %s/summary.json" % a.work)
