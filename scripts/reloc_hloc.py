#!/usr/bin/env python3
"""hloc-lite 한 장 재국소화 — 임베딩 검색(스캔 상위 K) → SIFT 기술자 직접 매칭(비율검정) → PnP. 1fps 구간에 SfM 없음.
    python scripts/reloc_hloc.py <seq> --scan-end 550 --live-step 10 --work ~/khcache/hloc-adt137 --map sfm|gt
· 스캔 지도: --map sfm = COLMAP 증분(초기 스캔에서만 허용) · --map gt = 스캔 프레임의 **주어진 포즈**(ADT/Nymeria 는 MPS)로 삼각측량만 —
  지도 품질과 PnP 정확도를 가른다(대조군).
· 검색: CLIP/DINOv2 전역 임베딩(1층 카메라방과 같은 임베딩) → live 마다 스캔 상위 K. vocab-tree 는 10 Hz 이웃 라이브만 골라 live↔스캔 쌍이 없었다(§166-4).
· 매칭: DB 의 SIFT 기술자(uint8 128) 를 읽어 L2 비율검정(0.8) — pycolmap 에 지정 쌍 매칭 함수가 없어 직접.
· 평가: 카메라 중심 오차(GT 대비, 지도는 sim3 로 GT 에 정렬 — --map gt 면 항등) · 방향 오차 · 장당 시간(임베딩·매칭·PnP 분리)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kx.depth.pose_stitch import umeyama  # noqa
import pycolmap  # noqa
ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("--scan-end", type=int, default=550); ap.add_argument("--scan-step", type=int, default=1); ap.add_argument("--live-step", type=int, default=10)
ap.add_argument("--work", required=True); ap.add_argument("--map", default="sfm", choices=["sfm", "gt"]); ap.add_argument("--features", type=int, default=4096); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--vocab", default=os.path.expanduser("~/khcache/colmap/vocab_tree_faiss_flickr100K_words32K.bin")); ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--embed", default="clip", choices=["clip", "dinov2"]); ap.add_argument("--ratio", type=float, default=0.8); ap.add_argument("--min-inliers", type=int, default=12)
ap.add_argument("--max-error", type=float, default=6.0)
ap.add_argument("--live-list", default=None, help="json {house: [t,…]} — 이 live 프레임(t=라이브 인덱스)만 국소화·특징 추출(후보 프레임만, §160)")
ap.add_argument("--house-name", default=None, help="--live-list 의 키(기본: seq 디렉터리 이름)")
ap.add_argument("--pose-out", default=None, help="eval_online POSE_JSONL 형식 {house,t,apos,yaw,inl,ratio} 로 출력")
ap.add_argument("--hssd-mirror", type=int, default=0, help="hssd_to_seq_reloc 가 x 를 뒤집어 만든 seq 이면 1 — apos·yaw 를 gt 프레임으로 되돌린다")
ap.add_argument("--min-ratio", type=float, default=0.0, help="--pose-out 게이트: 인라이어/2D-3D 비율 하한(반복 구조 장면용)")
a = ap.parse_args(); T0 = time.time()
def log(*x): print("[%5.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)
seq = a.seq.rstrip("/"); os.makedirs(a.work, exist_ok=True)
ci = json.load(open(os.path.join(seq, "camera_info.json"))); W, H = ci["width"], ci["height"]; fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
P = np.loadtxt(os.path.join(seq, "pose", "poses.txt")).reshape(-1, 4, 4)
rgbs = sorted(f for f in os.listdir(os.path.join(seq, "rgb")) if f.endswith(".jpg")); n = min(len(rgbs), len(P))
scan = ["rgb/" + rgbs[i] for i in range(0, min(a.scan_end, n), a.scan_step)]; live = ["rgb/" + rgbs[i] for i in range(a.scan_end, n, a.live_step)]
HN = a.house_name or os.path.basename(seq)
if a.live_list:
    _ql = set(int(t) for t in json.load(open(a.live_list)).get(HN, []))          # t = 라이브 프레임 번호 → seq 인덱스 = scan_end + t
    live = ["rgb/" + rgbs[a.scan_end + t] for t in sorted(_ql) if a.scan_end + t < n]
idx = {("rgb/" + rgbs[i]): i for i in range(n)}
log("스캔 %d장 · 라이브 %d장(%d장마다) · 지도 %s · 검색 %s top%d" % (len(scan), len(live), a.live_step, a.map, a.embed, a.topk))
# ── DB: 특징 전부 + 스캔끼리 매칭(지도용) ──
db = os.path.join(a.work, "db.db"); mk = os.path.join(a.work, ".matched")
if not os.path.exists(mk):
    if os.path.exists(db): os.remove(db)
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = "%g,%g,%g,%g" % (fx, fy, cx, cy)
    eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = a.threads; so = pycolmap.SiftExtractionOptions(); so.max_num_features = a.features; eo.sift = so
    allnames = ["rgb/" + f for f in rgbs[:n]] if not a.live_list else scan + live        # 후보만이면 추출도 스캔+후보만
    pycolmap.extract_features(db, seq, image_names=allnames, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="PINHOLE", reader_options=ro, extraction_options=eo)
    log("SIFT 추출 %d장" % len(allnames))
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = a.threads
    sq = pycolmap.SequentialPairingOptions(); sq.overlap = 8; sq.loop_detection = False; pycolmap.match_sequential(db, matching_options=mo, pairing_options=sq)
    vo = pycolmap.VocabTreePairingOptions(); vo.vocab_tree_path = a.vocab; vo.num_images = 20; vo.num_threads = a.threads; pycolmap.match_vocabtree(db, matching_options=mo, pairing_options=vo)
    log("스캔 매칭(순차+vocab)"); open(mk, "w").close()
dbh = pycolmap.Database.open(db); name2id = {im.name: im.image_id for im in dbh.read_all_images()}
# ── 스캔 지도 ──
def cfw_from_Twc(T):
    Rwc, twc = T[:3, :3], T[:3, 3]; Rcw = Rwc.T; tcw = -Rcw @ twc
    return pycolmap.Rigid3d(pycolmap.Rotation3d(Rcw), tcw)
if a.map == "gt":
    # 스캔 프레임의 **주어진 포즈**로 DB 키포인트에서 직접 재구성을 만들고 3D 점만 삼각측량 — SfM 이 안 서는 지도(시뮬 회전 지도)에서도 된다.
    gt_dir = os.path.join(a.work, "rec_gt")
    if os.path.isdir(gt_dir) and os.listdir(gt_dir): rec = pycolmap.Reconstruction(gt_dir)
    else:
        cam0 = dbh.read_all_cameras()[0]; rec0 = pycolmap.Reconstruction(); rec0.add_camera(cam0)
        rig = pycolmap.Rig(); rig.rig_id = 1; rig.add_ref_sensor(cam0.sensor_id); rec0.add_rig(rig)
        dbimgs = {im.name: im for im in dbh.read_all_images()}
        for nm in scan:
            im0 = dbimgs.get(nm)
            if im0 is None or nm not in idx: continue
            kp = dbh.read_keypoints(im0.image_id)[:, :2]
            fr = pycolmap.Frame(); fr.frame_id = im0.image_id; fr.rig_id = 1; fr.rig_from_world = cfw_from_Twc(P[idx[nm]])
            im = pycolmap.Image(name=nm, points2D=pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in kp]), camera_id=cam0.camera_id, image_id=im0.image_id)
            fr.add_data_id(im.data_id); rec0.add_frame(fr); im.frame_id = fr.frame_id; rec0.add_image(im); rec0.register_frame(fr.frame_id)
        os.makedirs(gt_dir, exist_ok=True)
        rec = pycolmap.triangulate_points(rec0, db, seq, gt_dir, clear_points=True)
    log("GT 포즈 삼각측량 지도: 등록 %d/%d · 점 %d · 재투영 %.2fpx  (점이 적거나 재투영이 크면 포즈 규약 오류)" % (rec.num_reg_images(), len(scan), rec.num_points3D(), rec.compute_mean_reprojection_error()))
    s_, R_, t_ = 1.0, np.eye(3), np.zeros(3)
else:
    rec_dir = os.path.join(a.work, "rec_scan"); os.makedirs(rec_dir, exist_ok=True)
    subs = [d for d in sorted(os.listdir(rec_dir)) if os.path.isdir(os.path.join(rec_dir, d))]
    if subs: recs = {d: pycolmap.Reconstruction(os.path.join(rec_dir, d)) for d in subs}
    else:
        o = pycolmap.IncrementalPipelineOptions(); o.num_threads = a.threads; o.image_names = scan; o.multiple_models = False
        o.ba_refine_focal_length = False; o.ba_refine_principal_point = False; o.ba_refine_extra_params = False
        recs = pycolmap.incremental_mapping(db, seq, rec_dir, o)
    rec = max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None
    if rec is None: log("SfM 스캔 지도 실패 — 스캔끼리 매칭이 없거나 초기화 불가"); sys.exit(2)
    log("SfM 스캔 지도: 등록 %d/%d · 점 %d · 재투영 %.2fpx" % (rec.num_reg_images(), len(scan), rec.num_points3D(), rec.compute_mean_reprojection_error()))
    X, Y = [], []
    for im in rec.images.values():
        if im.has_pose and im.name in idx: X.append(np.asarray(im.projection_center(), float)); Y.append(P[idx[im.name]][:3, 3])
    X, Y = np.array(X), np.array(Y); rng = np.random.default_rng(0); best = None
    for _ in range(500):
        i = rng.choice(len(X), 3, replace=False)
        try: s0, R0, t0 = umeyama(X[i], Y[i])
        except Exception: continue
        e = np.linalg.norm((s0 * (R0 @ X.T)).T + t0 - Y, axis=1); inl = e < 0.3
        if best is None or inl.sum() > best.sum(): best = inl
    s_, R_, t_ = umeyama(X[best], Y[best]); E0 = np.linalg.norm((s_ * (R_ @ X.T)).T + t_ - Y, axis=1)
    log("SfM 지도↔GT sim3: <0.1m %.2f <0.3m %.2f <1m %.2f · 분위 50/90 = %.3f/%.3fm · 스케일 %.3f" % ((E0 < .1).mean(), (E0 < .3).mean(), (E0 < 1).mean(), *np.quantile(E0, [.5, .9]), s_))
reg = {im.image_id: im for im in rec.images.values() if im.has_pose}
p3_of = {}
for iid, im in reg.items():
    m = {}
    for k, p2 in enumerate(im.points2D):
        if p2.has_point3D(): m[k] = rec.points3D[p2.point3D_id].xyz
    p3_of[iid] = m
scan_ids = [name2id[nm] for nm in scan if name2id.get(nm) in reg]
# ── 임베딩 검색 ──
import torch
torch.set_num_threads(1)
from PIL import Image
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if a.embed == "clip":
    from transformers import CLIPModel, CLIPProcessor
    pr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16"); md = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
    def _emb(ims): return md.get_image_features(**pr(images=ims, return_tensors="pt").to(dev))
else:
    from transformers import AutoModel, AutoImageProcessor
    pr = AutoImageProcessor.from_pretrained("facebook/dinov2-base"); md = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    def _emb(ims): return md(**pr(images=ims, return_tensors="pt").to(dev)).pooler_output
def emb(names, tag):
    cf = os.path.join(a.work, "emb_%s_%s.npy" % (a.embed, tag))
    if os.path.exists(cf):
        e = np.load(cf)
        if len(e) == len(names): return e
    out = []
    for i in range(0, len(names), 16):
        ims = [Image.open(os.path.join(seq, nm)).convert("RGB") for nm in names[i:i + 16]]
        with torch.no_grad(): e = _emb(ims)
        out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    E = np.concatenate(out); np.save(cf, E); return E
t1 = time.time(); SE = emb([rec.images[i].name for i in scan_ids], "scan"); LE = emb(live, "live"); t_emb = (time.time() - t1) / max(len(live), 1)
sim = LE @ SE.T; top = np.argsort(-sim, axis=1)[:, :a.topk]
# ── 기술자 직접 매칭 + PnP ──
desc_cache = {}; kp_cache = {}
def desc(iid):
    # numpy(BLAS)로 — torch CPU 행렬곱은 pycolmap 의 OpenMP 와 충돌해 pthread_mutex_init 크래시(OMP #179)
    if iid not in desc_cache:
        d = dbh.read_descriptors(iid).astype(np.float32).reshape(-1, 128); desc_cache[iid] = (d, (d * d).sum(1))
    return desc_cache[iid]
def kps(iid):
    if iid not in kp_cache: kp_cache[iid] = dbh.read_keypoints(iid).reshape(-1, 6)[:, :2] if dbh.read_keypoints(iid).size else np.zeros((0, 2))
    return kp_cache[iid]
def match(dq, ds):
    (q, qn), (sd, sn) = dq, ds
    if len(q) < 2 or len(sd) < 2: return np.zeros(0, int), np.zeros(0, int)      # 빈 벽 프레임: 키포인트 0~1개
    d2 = qn[:, None] + sn[None, :] - 2.0 * (q @ sd.T)
    i2 = np.argpartition(d2, 1, axis=1)[:, :2]                                    # 최근접 2개(순서 무관)
    v2 = np.take_along_axis(d2, i2, axis=1); order = np.argsort(v2, axis=1)
    v2 = np.take_along_axis(v2, order, axis=1); i2 = np.take_along_axis(i2, order, axis=1)
    ok = v2[:, 0] < (a.ratio ** 2) * np.maximum(v2[:, 1], 1e-9)
    return np.nonzero(ok)[0], i2[ok, 0]
cam = next(iter(rec.cameras.values())); eo_ = pycolmap.AbsolutePoseEstimationOptions()
try: eo_.ransac.max_error = a.max_error
except Exception: pass
rows = []
for li, nm in enumerate(live):
    qid = name2id[nm]; t2 = time.time(); pts2, pts3 = [], []; dq = desc(qid); q2 = kps(qid)
    for j in top[li]:
        mid = scan_ids[j]; qi, si = match(dq, desc(mid)); mp3 = p3_of[mid]
        for a_, b_ in zip(qi, si):
            if int(b_) in mp3: pts2.append(q2[int(a_)]); pts3.append(mp3[int(b_)])
    t_match = time.time() - t2; t3 = time.time(); res = None; ok = len(pts2) >= a.min_inliers
    if ok:
        res = pycolmap.estimate_and_refine_absolute_pose(np.asarray(pts2, float), np.asarray(pts3, float), cam, eo_); ok = res is not None and res["num_inliers"] >= a.min_inliers
    t_pnp = time.time() - t3; gtT = P[idx[nm]]
    if ok:
        cfw = res["cam_from_world"]; Rc = cfw.rotation.matrix(); c = -(Rc.T @ cfw.translation); c_al = s_ * (R_ @ c) + t_
        fwd = R_ @ (Rc.T @ np.array([0, 0, 1.0])); gtf = gtT[:3, :3] @ np.array([0, 0, 1.0])
        ang = float(np.degrees(np.arccos(np.clip(fwd @ gtf / (np.linalg.norm(fwd) * np.linalg.norm(gtf) + 1e-9), -1, 1))))
        rows.append(dict(name=nm, ok=True, n2d3d=len(pts2), inl=int(res["num_inliers"]), err=float(np.linalg.norm(c_al - gtT[:3, 3])), ang=ang, t_match=t_match, t_pnp=t_pnp, c_al=[float(v) for v in c_al], fwd_al=[float(v) for v in fwd]))
    else: rows.append(dict(name=nm, ok=False, n2d3d=len(pts2), inl=int(res["num_inliers"]) if res else 0, err=None, ang=None, t_match=t_match, t_pnp=t_pnp))
okr = [r for r in rows if r["ok"]]; E = np.array([r["err"] for r in okr]); A = np.array([r["ang"] for r in okr])
log("라이브 PnP(%s 지도): 등록 %d/%d (%.2f) · 위치 오차 중앙 %.3fm · <0.5m %.2f · <1m %.2f · 방향 중앙 %.1f° <10° %.2f · 장당 임베딩 %.0fms 매칭 %.0fms PnP %.0fms · 2D-3D 중앙 %d" % (
    a.map, len(okr), len(rows), len(okr) / max(len(rows), 1), np.median(E) if len(E) else -1, (E < .5).mean() if len(E) else 0, (E < 1).mean() if len(E) else 0,
    np.median(A) if len(A) else -1, (A < 10).mean() if len(A) else 0, 1000 * t_emb, 1000 * np.median([r["t_match"] for r in rows]), 1000 * np.median([r["t_pnp"] for r in rows]), int(np.median([r["n2d3d"] for r in rows]))))
if a.pose_out:
    with open(a.pose_out, "w") as fo:
        for r in rows:
            if not r["ok"] or r["inl"] < a.min_inliers or (r["n2d3d"] and r["inl"] / r["n2d3d"] < a.min_ratio): continue
            c, f = r["c_al"], r["fwd_al"]
            if a.hssd_mirror: c = [-c[0], c[1], c[2]]; f = [-f[0], f[1], f[2]]
            t = idx[r["name"]] - a.scan_end
            fo.write(json.dumps(dict(house=HN, t=int(t), apos=[round(float(c[0]), 3), round(float(c[2]), 3)], yaw=round(float(np.degrees(np.arctan2(f[0], f[2])) % 360), 1), inl=r["inl"], ratio=round(r["inl"] / max(r["n2d3d"], 1), 3))) + "\n")
    log("→ POSE_JSONL %s (%d/%d 프레임, 게이트 inl≥%d ratio≥%.2f)" % (a.pose_out, sum(1 for r in rows if r["ok"] and r["inl"] >= a.min_inliers and (not r["n2d3d"] or r["inl"] / r["n2d3d"] >= a.min_ratio)), len(rows), a.min_inliers, a.min_ratio))
json.dump(dict(seq=os.path.basename(seq), map=a.map, embed=a.embed, topk=a.topk, n_scan=len(scan), map_reg=rec.num_reg_images(), map_pts=rec.num_points3D(), n_live=len(rows), live_reg=len(okr),
               err_med=float(np.median(E)) if len(E) else None, lt05=float((E < .5).mean()) if len(E) else None, lt1=float((E < 1).mean()) if len(E) else None, ang_med=float(np.median(A)) if len(A) else None,
               t_emb_ms=1000 * t_emb, t_match_ms=1000 * float(np.median([r["t_match"] for r in rows])), t_pnp_ms=1000 * float(np.median([r["t_pnp"] for r in rows])), rows=rows, sec=time.time() - T0),
          open(os.path.join(a.work, "summary_%s_%s_step%d.json" % (a.map, a.embed, a.live_step)), "w"), indent=1)
