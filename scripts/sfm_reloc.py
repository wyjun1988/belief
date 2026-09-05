#!/usr/bin/env python3
"""SfM 재국소화 — 매핑워크(map/)를 pycolmap 으로 **한 번** 재구성하고, live 1fps 프레임을 **프레임별로** 지도에
등록(relocalization)한다. 프레임 사이 겹침이 필요 없으므로 1fps·급회전에서도 사슬이 끊기지 않는다(§139 의 실패는
live 스트림 자체를 추적하려던 것).

    python scripts/sfm_reloc.py data/hssd20S2/house_0000 --vocab ~/khcache/colmap/vocab_tree_faiss_flickr100K_words32K.bin \
        --out ~/khcache/sfm/pose_house_0000.jsonl

단계: SIFT(CPU) → vocab-tree 검색 매칭(+순차 매칭) → map 만 증분 매핑 → map 고정(fix_existing_frames) 상태로
live 등록 → sim3 정렬 → {house,t,apos,yaw} jsonl (eval_online POSE_JSONL).
정렬(sim3)은 **GT 매핑워크 포즈**로 푼다 — 단안 SfM 은 척도·전역 좌표계가 자유이고, 평면도(GT)가 그 좌표계에 있으므로
좌표계 정의용이다. 사다리에는 '위치:SfM(정렬:GT맵포즈)' 로 남긴다. 미러 규약(x 반전) 가능성은 두 정렬 중 잔차 작은 쪽.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kx.depth.pose_stitch import umeyama  # noqa: E402
import pycolmap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("house")
ap.add_argument("--work", default=None, help="DB·재구성 디렉터리 (기본 ~/khcache/sfm/<house>)")
ap.add_argument("--vocab", default=os.path.expanduser("~/khcache/colmap/vocab_tree_faiss_flickr100K_words32K.bin"))
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--from-poses", default=None, help="COLMAP 을 건너뛰고 외부 포즈(jsonl: {name, c:[x,y,z], f:[x,y,z]}, **메트릭**)로 정렬·평가만 — VGGT 등 다른 재구성기 결과를 같은 표로 재려고")
ap.add_argument("--live-step", type=int, default=1, help="live 프레임 N장마다 1장만 SfM 에 넣는다(긴 에피소드용). 등록 안 된 프레임은 평가에서 기하 기권")
ap.add_argument("--topk", type=int, default=20, help="vocab-tree 검색 이미지 수")
ap.add_argument("--features", type=int, default=2048, help="프레임당 SIFT 상한. 질감이 많은 장면(v3c: 상한 없이 5,400개)에서 매칭이 제곱으로 느려진다")
ap.add_argument("--out", default=None)
ap.add_argument("--redo", action="store_true")
ap.add_argument("--vmax", type=float, default=2.5, help="속도 필터: ±3초 이웃 등록 프레임과의 중앙 거리(m)가 이보다 크면 기권(1fps 보행 ≤1.5m/s). 0=끔")
ap.add_argument("--scale", default="gt", choices=["gt", "da"], help="척도 출처: gt=sim3 에서 GT 맵포즈로 · da=단안 메트릭 깊이(DA-V2)×데이터셋 상수(--da-k), 정렬은 회전·병진만 GT 맵포즈")
ap.add_argument("--da-k", type=float, default=0.468, help="GT/DA 척도 상수 (HSSD 렌더 4채 중앙 0.468, 집별 ±5%%). 새 렌더러(OG)는 1채로 재보정")
ap.add_argument("--da-n", type=int, default=40)
ap.add_argument("--reject-outside", action="store_true", help="[기본 OFF — 4채 벤치 0.829→0.805] 정렬 뒤 어느 방 폴리곤에도 들어가지 않는 live 프레임을 기권 처리 — 유령 복제(잘못 등록된 사본)를 GT 없이 거른다. 평면도는 사용자 입력")
ap.add_argument("--align", default="gt", choices=["gt", "sites"], help="회전·병진 출처: gt=GT 맵포즈 sim3 · sites=등록 때 붙인 지점 라벨이 평면도 폴리곤 안에 들어가게(GT 좌표 불사용; 척도는 --scale da 필수)")
ap.add_argument("--fast", action="store_true", help="전역 BA 를 덜 자주(1.1→1.3배)·반복 절반 — live 등록 시간 단축(정확도는 4채에서 대조할 것)")
ap.add_argument("--redo-map", action="store_true", help="DB(특징·매칭)는 두고 재구성만 다시 — 매퍼 노브 실험용")
ap.add_argument("--minmatch", type=int, default=0, help="매퍼 min_num_matches (기본 15). 반복 구조에서 지도가 접히면 30~40 으로")
ap.add_argument("--strict", action="store_true", help="PnP 등록 문턱 강화(abs_pose_min_num_inliers 60·inlier_ratio 0.4) — 반복 구조 유령 등록 억제")
ap.add_argument("--map-only", action="store_true", help="정렬·포즈를 지도 단독 재구성(rec_map)에서만 — live 보행이 지도를 오염시키는지 가르는 진단")
ap.add_argument("--seq-overlap", type=int, default=20, help="matcher=seq 의 순차 창(프레임). 지도가 0.2m 간격이면 20 ≈ 4m")
ap.add_argument("--matcher", default="vocab", choices=["vocab", "exhaustive", "seq"], help="vocab: 순차+vocab-tree 검색(faiss 판 트리 필요) · exhaustive: 전수")
ap.add_argument("--gpu", action="store_true", help="CUDA 박스(RTX)에서 SIFT·매칭을 GPU 로")
a = ap.parse_args()

hd = a.house.rstrip("/"); hn = os.path.basename(hd)
work = a.work or os.path.expanduser("~/khcache/sfm/%s" % hn); os.makedirs(work, exist_ok=True)
g = json.load(open(os.path.join(hd, "gt.json")))
maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg"))
lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg"))
_nlive0 = len(lives)
if a.live_step > 1: lives = lives[::a.live_step]
names_map = ["map/" + f for f in maps]; names_live = ["live/" + f for f in lives]
from PIL import Image
W, H = Image.open(os.path.join(hd, "map", maps[0])).size; fx = W / 2.0          # hfov 90° 핀홀
db = os.path.join(work, "db.db"); T0 = time.time()
DEV = pycolmap.Device.cuda if a.gpu else pycolmap.Device.cpu
def log(*x): print("[%5.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)

if a.from_poses:
    P = {}
    for _l in open(a.from_poses):
        _d = json.loads(_l); P[_d["name"]] = (np.asarray(_d["c"], float), np.asarray(_d["f"], float))
    ra = rm = None; S_FIX = 1.0                      # 외부 포즈는 이미 메트릭
    log("외부 포즈 %d (map %d · live %d)" % (len(P), sum(1 for k in P if k.startswith("map/")), sum(1 for k in P if k.startswith("live/"))))
ap_matcher = a.matcher
mk_ext, mk_mat = os.path.join(work, ".extracted"), os.path.join(work, ".matched_" + ap_matcher)
if a.redo and not a.from_poses:
    for f in (db, mk_ext, mk_mat):
        if os.path.exists(f): os.remove(f)
if not a.from_poses and not os.path.exists(mk_ext):
    if os.path.exists(db): os.remove(db)
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = "%g,%g,%g,%g" % (fx, fx, W / 2.0, H / 2.0)
    eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = a.threads; eo.use_gpu = a.gpu
    _so = pycolmap.SiftExtractionOptions(); _so.max_num_features = a.features; eo.sift = _so   # ⚠️ eo.sift 는 복사본을 돌려준다 — 통째로 대입해야 상한이 먹는다
    pycolmap.extract_features(db, hd, image_names=names_map + names_live, camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model="PINHOLE", reader_options=ro, extraction_options=eo, device=DEV)
    open(mk_ext, "w").close(); log("SIFT 추출 map %d + live %d%s" % (len(maps), len(lives), (" (원본 %d 에서 %d장마다)" % (_nlive0, a.live_step)) if a.live_step > 1 else ""))
if not a.from_poses and not os.path.exists(mk_mat):
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = a.threads; mo.use_gpu = a.gpu
    if ap_matcher in ("vocab", "seq"):
        # seq: 순차만 — vocab-tree 검색이 **반복 구조에서 거짓 루프**를 만드는 것이 §164 진단의 결론이라
        # (GT sim3 자유척도에서도 인라이어 0.06~0.17) 검색을 빼고 시간 이웃만 잇는다. 대신 창을 넓힌다.
        so = pycolmap.SequentialPairingOptions(); so.overlap = 8 if ap_matcher == "vocab" else a.seq_overlap
        so.loop_detection = False
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=so, device=DEV)
        log("순차 매칭(overlap %d)" % so.overlap)
        if ap_matcher == "vocab":
            vo = pycolmap.VocabTreePairingOptions(); vo.vocab_tree_path = a.vocab; vo.num_images = a.topk; vo.num_threads = a.threads
            pycolmap.match_vocabtree(db, matching_options=mo, pairing_options=vo, device=DEV); log("vocab-tree 매칭(top%d)" % a.topk)
    else:                                   # 검색 없이 전수 — 1.3k 장이면 CPU 로 십수 분, 살아있는 vocab tree 가 없을 때
        xo = pycolmap.ExhaustivePairingOptions(); xo.block_size = 100
        pycolmap.match_exhaustive(db, matching_options=mo, pairing_options=xo, device=DEV); log("전수 매칭")
    open(mk_mat, "w").close()

def mapper_opts(names, fix):
    o = pycolmap.IncrementalPipelineOptions(); o.num_threads = a.threads; o.image_names = names
    o.ba_refine_focal_length = False; o.ba_refine_principal_point = False; o.ba_refine_extra_params = False
    o.multiple_models = False; o.fix_existing_frames = fix
    if a.minmatch: o.min_num_matches = a.minmatch; o.mapper.abs_pose_min_num_inliers = max(30, a.minmatch)
    if a.strict:
        o.mapper.abs_pose_min_num_inliers = 60; o.mapper.abs_pose_min_inlier_ratio = 0.4
    if a.fast:
        o.ba_global_frames_ratio = 1.3; o.ba_global_points_ratio = 1.3
        o.ba_global_max_num_iterations = 30; o.ba_local_max_num_iterations = 15
    return o
def best_rec(recs):
    return max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None

if not a.from_poses:
    rec_map_dir = os.path.join(work, "rec_map"); rec_all_dir = os.path.join(work, "rec_all")
    rm = None
    if not (a.redo or a.redo_map) and os.path.isdir(rec_map_dir) and any(os.path.isdir(os.path.join(rec_map_dir, d)) for d in os.listdir(rec_map_dir)):
        rm = pycolmap.Reconstruction(os.path.join(rec_map_dir, sorted(d for d in os.listdir(rec_map_dir) if os.path.isdir(os.path.join(rec_map_dir, d)))[0]))
    else:
        os.makedirs(rec_map_dir, exist_ok=True)
        if a.redo_map: os.system("rm -rf '%s'/* '%s'/*" % (rec_map_dir, rec_all_dir))
        rm = best_rec(pycolmap.incremental_mapping(db, hd, rec_map_dir, mapper_opts(names_map, False)))
    # 지점 회전만 있는 매핑워크(HSSD 밀집 remap)는 시차가 없어 map 만으로는 초기화가 안 되거나 몇 장만 붙는다 →
    # 그때는 map+live 를 한 번에 재구성(joint). live 보행이 기준선을 준다. (OG 는 SPEC 3-b 이동 프레임으로 map 만으로도 되게)
    map_ok = rm is not None and rm.num_reg_images() >= max(10, 0.3 * len(maps))
    if rm is not None: log("map 재구성: 등록 %d/%d · 점 %d · 재투영 %.2fpx%s" % (rm.num_reg_images(), len(maps), rm.num_points3D(), rm.compute_mean_reprojection_error(), "" if map_ok else " → 빈약, joint 로"))
    else: log("map 재구성 실패 → joint 로")
    sub = [d for d in sorted(os.listdir(rec_map_dir)) if os.path.isdir(os.path.join(rec_map_dir, d))] if os.path.isdir(rec_map_dir) else []
    in_path = os.path.join(rec_map_dir, sub[0]) if (sub and map_ok) else ""
    os.makedirs(rec_all_dir, exist_ok=True)
    _subs = [d for d in sorted(os.listdir(rec_all_dir)) if os.path.isdir(os.path.join(rec_all_dir, d))]
    if not (a.redo or a.redo_map) and _subs:      # live 등록 결과 캐시 (10~17분짜리) — 가장 큰 모델
        ra = best_rec({d: pycolmap.Reconstruction(os.path.join(rec_all_dir, d)) for d in _subs}); log("live 등록 캐시 사용 (%d모델)" % len(_subs))
    elif in_path:
        ra = best_rec(pycolmap.incremental_mapping(db, hd, rec_all_dir, mapper_opts([], True), input_path=in_path))
    else:
        ra = best_rec(pycolmap.incremental_mapping(db, hd, rec_all_dir, mapper_opts([], False)))
    if a.map_only and rm is not None:
        # §165 계획 3단계: 정렬을 **지도 단독 재구성**으로만. 종전에는 joint(rec_all) 포즈로 정렬했는데,
        # 새 데이터는 live 1200장 중 800장이 제자리(정지)라 joint 가 지도를 끌고 갈 수 있다.
        # fix_existing_frames 가 실제로 지도를 고정했는지도 이 비교로 드러난다.
        ra = rm; log("정렬 재료 = 지도 단독 재구성(map %d장) — joint 미사용" % rm.num_reg_images())
    if ra is None: sys.exit("live 등록 실패")
    n_live_reg = sum(1 for im in ra.images.values() if im.name.startswith("live/") and im.has_pose)
    log("live 등록: %d/%d" % (n_live_reg, len(lives)))

    # ── 포즈 추출 → sim3 정렬(GT map 포즈, 미러 후보 포함) ──
    def poses(rec):
        out = {}
        for im in rec.images.values():
            if not im.has_pose: continue
            out[im.name] = (np.asarray(im.projection_center(), float), np.asarray(im.viewing_direction(), float))
        return out
    P = poses(ra)
    gm = g["map"]; assert len(gm) == len(maps), "gt.map %d ≠ map 프레임 %d" % (len(gm), len(maps))
    src, dst = [], []
    for nm, m in zip(names_map, gm):
        if nm in P: src.append(P[nm][0]); dst.append([m["apos"][0], 1.5, m["apos"][1]])
    src, dst = np.array(src), np.array(dst)
    def da_scale(rec, n):
        """단안 메트릭 깊이(DA-V2)로 SfM 척도: 프레임의 3D 점 SfM 깊이 z 와 같은 픽셀 DA 깊이 비율 중앙값 (GT 불필요)"""
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from PIL import Image as _Im
        mname = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        pr = AutoImageProcessor.from_pretrained(mname); md = AutoModelForDepthEstimation.from_pretrained(mname).to(dev).eval()
        ims = sorted((im for im in rec.images.values() if im.has_pose and im.name.startswith("live/")), key=lambda im: im.name)
        ims = ims[::max(1, len(ims) // n)][:n]; fr = []
        for im in ims:
            p2 = [q for q in im.points2D if q.has_point3D()]
            if len(p2) < 8: continue
            X = np.array([rec.points3D[q.point3D_id].xyz for q in p2]); xy = np.array([q.xy for q in p2])
            cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world; z = (cfw * X)[:, 2]
            img = _Im.open(os.path.join(hd, im.name)).convert("RGB"); inp = pr(images=img, return_tensors="pt").to(dev)
            with torch.no_grad(): D = md(**inp).predicted_depth
            D = torch.nn.functional.interpolate(D[None], size=img.size[::-1], mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
            u = np.clip(xy[:, 0].round().astype(int), 0, D.shape[1] - 1); v = np.clip(xy[:, 1].round().astype(int), 0, D.shape[0] - 1)
            ok = (z > 0.3) & (D[v, u] > 0.3)
            if ok.sum() >= 8: fr.append(float(np.median(D[v, u][ok] / z[ok])))
        return float(np.median(fr)) if fr else float("nan"), len(fr)
if a.from_poses:                                  # COLMAP 블록을 건너뛰었으니 정렬 입력(gm·src·dst)만 여기서 만든다
    gm = g["map"]; assert len(gm) == len(maps), "gt.map %d ≠ map 프레임 %d" % (len(gm), len(maps))
    src, dst = [], []
    for nm, m in zip(names_map, gm):
        if nm in P: src.append(P[nm][0]); dst.append([m["apos"][0], 1.5, m["apos"][1]])
    src, dst = np.array(src), np.array(dst)
S_FIX = (1.0 if a.scale == "da" else None) if a.from_poses else None   # 외부 포즈: --scale da=어댑터가 준 미터 그대로(rigid) · --scale gt=척도까지 GT sim3 로(진단)
if a.from_poses: map_ok = True; rm = None; ra = None
if a.scale == "da" and not a.from_poses:
    _sda, _nf = da_scale(ra, a.da_n); S_FIX = _sda * a.da_k
    log("DA 척도: SfM→m 비율 %.3f (프레임 %d) × 상수 %.3f = %.3f — 정렬은 회전·병진만" % (_sda, _nf, a.da_k, S_FIX))
def rigid(X, Y):
    cx, cy = X.mean(0), Y.mean(0); H = (X - cx).T @ (Y - cy); U, _, Vt = np.linalg.svd(H)
    Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R_ = Vt.T @ Dg @ U.T
    return R_, cy - R_ @ cx
def fit(X, Y):
    if S_FIX is not None:
        R_, t_ = rigid(X * S_FIX, Y); s_ = S_FIX
    else:
        s_, R_, t_ = umeyama(X, Y)
    return s_, R_, t_, np.linalg.norm((s_ * (R_ @ X.T)).T + t_ - Y, axis=1)
def ransac_sim3(X, Y, th=0.5, iters=300, seed=0):
    """map 프레임 일부가 잘못 등록(드리프트·오병합)돼도 다수가 맞는 sim3 — 3점 표본 → 인라이어 → 재적합"""
    rng = np.random.default_rng(seed); n = len(X); best_in = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try: s_, R_, t_, e = fit(X[idx], Y[idx])
        except Exception: continue
        e = np.linalg.norm((s_ * (R_ @ X.T)).T + t_ - Y, axis=1); inl = e < th
        if best_in is None or inl.sum() > best_in.sum(): best_in = inl
    if best_in is None or best_in.sum() < 4: best_in = np.ones(n, bool)
    s_, R_, t_, e = fit(X[best_in], Y[best_in])
    return s_, R_, t_, float(np.sqrt((e ** 2).mean())), float(best_in.mean())
def align_by_labels():
    """GT 좌표 없이 정렬: (1) 중력 = 카메라 y축 평균(피치 0) → y-up, (2) yaw·x·z 는 "지점 i 는 방 label_i 폴리곤 안" 을
    최대로 만족하는 값(격자 탐색 + 국소 정제). 평면도(폴리곤)는 사용자가 주는 입력, 지점 라벨은 등록 때 사용자가 붙인 것."""
    assert S_FIX is not None, "--align sites 는 --scale da 와 함께"
    polys = (g.get("scene_meta") or {}).get("polys") or {}
    def pip_(pt, poly):
        x, z = pt; ins = False; n = len(poly)
        for i in range(n):
            x1, z1 = poly[i][0], poly[i][-1]; x2, z2 = poly[(i + 1) % n][0], poly[(i + 1) % n][-1]
            if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
        return ins
    downs = []
    for im in (ra.images.values() if ra is not None else []):
        if not im.has_pose: continue
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        downs.append(cfw.rotation.matrix().T @ np.array([0, 1.0, 0]))
    if not downs and a.from_poses:
        # 외부 포즈: 어댑터가 카메라 up 벡터("u")를 내보내면 그것으로, 없으면 카메라 중심 평면의 법선(모두 눈높이)으로
        _ups = [np.asarray(_d["u"], float) for _l in open(a.from_poses) for _d in [json.loads(_l)] if "u" in _d]
        if _ups: downs = [-u_ for u_ in _ups]
        else:
            _Cc = np.array([P[k][0] for k in P]); _Cc = _Cc - _Cc.mean(0)
            _n = np.linalg.svd(_Cc, full_matrices=False)[2][-1]
            if _n[1] < 0: _n = -_n                      # 부호 추정: 원시 프레임의 +y 쪽 (어댑터가 u 를 주면 불필요)
            downs = [-_n]; log("⚠️ 외부 포즈에 u 가 없어 중력을 카메라 평면 법선으로 추정(부호는 가정) — 어댑터를 갱신하라")
    up = -np.mean(downs, 0); up /= np.linalg.norm(up)
    # up → +y 회전
    v = np.cross(up, [0, 1.0, 0]); c = float(np.dot(up, [0, 1.0, 0])); vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rg = np.eye(3) + vx + vx @ vx / (1 + c) if c > -0.999 else np.diag([1, -1, -1.0])
    sites = [(S_FIX * (Rg @ P[nm][0]), m["room"]) for nm, m in zip(names_map, gm) if nm in P and m["room"] in polys]
    cams = [S_FIX * (Rg @ P[nm][0]) for nm in P if nm.startswith("live/")]
    cams = np.array(cams)[:: max(1, len(cams) // 200)]
    pc = {r: np.mean(np.array(pl)[:, [0, -1]], 0) for r, pl in polys.items()}
    # 정렬 탐색: 중심 맞춤 ±1m 격자는 방 크기가 제각각인 집에서 참값을 놓친다(20채 중 14채 실패, yaw 60~130°).
    # → **대응 1개 가설**: (지점 s, 그 지점 라벨 방의 중심) 하나가 변환을 정하고, 나머지 지점이 제 방에 들어가는 수로 채점.
    def _score(sp2, tt):
        return sum(pip_((q[0] + tt[0], q[1] + tt[1]), polys[r]) for q, (_, r) in zip(sp2, sites))
    best = None
    _sub = list(range(0, len(sites), max(1, len(sites) // 40)))       # 가설은 최대 40지점만 (비용 절감)
    for mirror in (False, True):
        Mm = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        for yaw in np.arange(0, 360, 3.0):
            y = np.radians(yaw); Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]) @ Mm
            sp = np.array([Ry @ p for p, _ in sites])[:, [0, 2]]
            for k in _sub:
                tt = pc[sites[k][1]] - sp[k]
                sc = _score(sp, tt)
                if best is None or sc > best[0]: best = (sc, yaw, mirror, Ry, tt)
    sc, yaw, mirror, Ry, tt = best
    # 국소 정제 (yaw ±3° / 0.5°, 이동 ±0.6m / 0.15m) — 동점이면 live 카메라가 폴리곤 안에 드는 수로 가른다
    for _ in range(2):
        cand = []
        Mm = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        for dy in np.arange(-3, 3.01, 0.5):
            y = np.radians(yaw + dy); Ry2 = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]) @ Mm
            sp = np.array([Ry2 @ p for p, _ in sites])[:, [0, 2]]
        cp = np.array([Ry2 @ q for q in cams])[:, [0, 2]] if len(cams) else np.zeros((0, 2))   # --map-only: live 0장
            for dx in np.arange(-0.6, 0.61, 0.15):
                for dz in np.arange(-0.6, 0.61, 0.15):
                    t2 = tt + [dx, dz]
                    k2 = (_score(sp, t2), sum(any(pip_((q[0] + t2[0], q[1] + t2[1]), pl) for pl in polys.values()) for q in cp))
                    cand.append((k2, yaw + dy, Ry2, t2))
        k, yaw, Ry, tt = max(cand, key=lambda c: c[0])
    R_ = Ry @ Rg; t_ = np.array([tt[0], 1.5 - S_FIX * float(np.mean([(R_ @ P[nm][0])[1] for nm in P])), tt[1]])
    log("라벨 정렬: 지점 %d/%d 이 제 방 폴리곤 안 · live 카메라 폴리곤 안 %d/%d · yaw %.1f° · 미러 %s" % (k[0], len(sites), k[1], len(cams), yaw, mirror))
    return R_, t_, mirror
best = None
if a.align == "sites":
    R3, T3, mirror = align_by_labels(); S3 = S_FIX; M = np.eye(3)   # 미러는 R3 안에 포함
    al = (S3 * (R3 @ src.T)).T + T3; rms = float(np.sqrt(((al - dst) ** 2).sum(1).mean())); INL = float((np.linalg.norm(al - dst, axis=1) < 0.5).mean())
    log("(대조) GT 맵포즈 대비: rms %.3fm · 0.5m 이내 %.2f" % (rms, INL)); s = S3
else:
    for mirror in (False, True):
        M = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        s, R, t, rms, inl = ransac_sim3(src @ M.T, dst)
        if best is None or (inl, -rms) > (best[6], -best[0]): best = (rms, mirror, M, s, R, t, inl)
    rms, mirror, M, S3, R3, T3, INL = best; s = S3
log("%s 정렬(map %d프레임, RANSAC 0.5m): 인라이어 %.2f · 인라이어 rms %.3fm · 스케일 %.3f · 미러 %s" % ("rigid(척도 DA)" if S_FIX else "sim3", len(src), INL, rms, s, mirror))

def to_ours(c, v):
    c2 = S3 * (R3 @ (M @ c)) + T3; v2 = R3 @ (M @ v)
    return [round(float(c2[0]), 3), round(float(c2[2]), 3)], round(float(np.degrees(np.arctan2(v2[0], v2[2])) % 360), 1)

# ── 평가: live ATE·yaw·카메라방 적중 ──
live = {m["t"]: m for m in g["live"]}
rooms = (g.get("scene_meta") or {}).get("polys") or g.get("rooms") or {}   # HSSD: scene_meta.polys {room: [[x,z],...]}
def poly_of(r):
    if isinstance(r, list) and len(r) >= 3 and isinstance(r[0], (list, tuple)): return r
    if isinstance(r, dict):
        for k in ("poly", "polygon", "pts"):
            if k in r: return r[k]
    return None
def pip(pt, poly):
    x, z = pt; ins = False; n = len(poly)
    for i in range(n):
        x1, z1 = poly[i][0], poly[i][-1]; x2, z2 = poly[(i + 1) % n][0], poly[(i + 1) % n][-1]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
    return ins
try:
    polys = {rid: poly_of(r) for rid, r in (rooms.items() if isinstance(rooms, dict) else enumerate(rooms))}
    polys = {k: v for k, v in polys.items() if v}
except Exception as e:
    print("rooms 폴리곤 해석 실패(%s) — 카메라방 적중 생략" % e); polys = {}
est = {}
for f in lives:
    t = int(f[:-4]); nm = "live/" + f
    if nm in P and t in live: est[t] = to_ours(*P[nm])
n_before = len(est)
if a.vmax > 0:                                   # GT 없이 되는 일관성 검사: 유령 등록(반복 구조)은 이웃과 수 m 튄다
    drop = set()
    for t, (ap_, _) in est.items():
        nb = [np.hypot(ap_[0] - est[u][0][0], ap_[1] - est[u][0][1]) for u in range(t - 3, t + 4) if u != t and u in est]
        if nb and np.median(nb) > a.vmax: drop.add(t)
    for t in drop: est.pop(t)
    log("속도 필터(vmax %.1fm): 기권 %d/%d" % (a.vmax, len(drop), n_before))
rows, ate, yerr, hit, nhit = [], [], [], 0, 0
for t in sorted(est):
    m = live[t]; apos, yaw = est[t]; rows.append(dict(house=hn, t=t, apos=apos, yaw=yaw))
    ate.append(np.hypot(apos[0] - m["apos"][0], apos[1] - m["apos"][1]))
    yerr.append(abs((yaw - m["yaw"] + 180) % 360 - 180))
    if polys and m.get("room") in polys:
        nhit += 1; hit += pip(apos, polys[m["room"]])
    m["room_gt"] = m.get("room")
if a.reject_outside and polys:
    def _inany(pt):
        for pl in polys.values():
            if pip(pt, pl): return True
        return False
    keep = [i for i, r in enumerate(rows) if _inany(r["apos"])]
    n0 = len(rows)
    if len(keep) >= 0.2 * n0:                      # 전부 밖이면(정렬 실패) 필터를 적용하지 않는다
        rows = [rows[i] for i in keep]; ate = [ate[i] for i in keep]; yerr = [yerr[i] for i in keep]
        log("평면도 밖 기권: %d/%d (남은 %d)" % (n0 - len(keep), n0, len(rows)))
    else:
        log("평면도 밖 기권 생략 — 남는 프레임 %d/%d (정렬 실패 의심)" % (len(keep), n0))
    hit = nhit = 0
    for r in rows:
        m = live[r["t"]]
        if m.get("room_gt", m.get("room")) in polys:
            nhit += 1; hit += pip(r["apos"], polys[m.get("room_gt", m.get("room"))])
ate, yerr = np.array(ate), np.array(yerr)
cov = len(rows) / _nlive0
log(("live 표본 등록 %d/%d(%.2f) · " % (len(rows), len(lives), len(rows) / max(len(lives), 1))) if a.live_step > 1 else "")
log("live 커버리지 %.2f · ATE 중앙 %.2fm 평균 %.2fm <0.5m %.2f <1m %.2f · yaw 중앙 %.1f° <10° %.2f · 카메라방 적중 %s" % (
    cov, np.median(ate) if len(ate) else -1, ate.mean() if len(ate) else -1, (ate < 0.5).mean() if len(ate) else 0,
    (ate < 1).mean() if len(ate) else 0, np.median(yerr) if len(yerr) else -1, (yerr < 10).mean() if len(yerr) else 0,
    ("%.2f (n=%d)" % (hit / nhit, nhit)) if nhit else "—"))
# 정렬된 3D 점 내보내기 (평면도용: 벽·가구 = 높이 0.2~2.0m 점) — x, z, h(우리 프레임, y-up)
_P3 = np.array([pt.xyz for pt in ra.points3D.values() if pt.track.length() >= 3]) if (ra is not None and ra.num_points3D()) else np.zeros((0, 3))
if len(_P3):
    _A = (S3 * (R3 @ (M @ _P3.T))).T + T3
    np.savez_compressed(os.path.join(work, "points_%s.npz" % hn), x=_A[:, 0].astype(np.float32), z=_A[:, 2].astype(np.float32), h=_A[:, 1].astype(np.float32))
    log("3D 점 %d (트랙≥3) → points_%s.npz · 높이 중앙 %.2fm" % (len(_A), hn, float(np.median(_A[:, 1]))))
# live 프레임의 2D-3D 대응 (u, v, 메트릭 깊이) — 거리 보정용. GT 앵커 좌표·GT 카메라 위치를 대체한다.
_lt, _lu, _lv, _ld = [], [], [], []
for f in (lives if ra is not None else []):
    t = int(f[:-4]); nm = "live/" + f
    im = next((i for i in ra.images.values() if i.name == nm and i.has_pose), None)
    if im is None: continue
    p2 = [q for q in im.points2D if q.has_point3D()]
    if not p2: continue
    X = np.array([ra.points3D[q.point3D_id].xyz for q in p2]); xy = np.array([q.xy for q in p2])
    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    z = (cfw * X)[:, 2] * S3
    ok = z > 0.1
    _lt.append(np.full(int(ok.sum()), t)); _lu.append(xy[ok, 0]); _lv.append(xy[ok, 1]); _ld.append(z[ok])
if _lt:
    np.savez_compressed(os.path.join(work, "live_points_%s.npz" % hn), t=np.concatenate(_lt).astype(np.int32),
                        u=np.concatenate(_lu).astype(np.float32), v=np.concatenate(_lv).astype(np.float32), d=np.concatenate(_ld).astype(np.float32))
    log("live 2D-3D 대응 %d점 (프레임 %d)" % (sum(len(x) for x in _lt), len(_lt)))
out = a.out or os.path.join(work, "pose_%s.jsonl" % hn)
with open(out, "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
# 맵 포즈 내보내기 + **지점 전파**: 매핑 프로토콜은 지점당 NA 각도(45° 스텝)다. 지점 회전 프레임은 시차가 없어 절반쯤 SfM 에
# 안 붙는데, 같은 지점의 등록 프레임이 하나라도 있으면 위치는 그 평균, yaw 는 45°×(각도 차)로 채운다(프로토콜 지식, GT 아님).
NA = int(os.environ.get("MAP_ANGLES", "8")); STEP = float(os.environ.get("MAP_YAW_STEP", "45"))
if a.from_poses or any(m.get("travel") for m in gm): NA = 10**9   # 외부 포즈·이동 프레임 지도: 지점당 8각 블록 가정이 깨지므로 전파 끔
mpose = {nm: to_ours(*P[nm]) for nm in names_map if nm in P}
_sgn = []
for b0 in range(0, len(names_map), NA):
    ks = [k for k in range(b0, min(b0 + NA, len(names_map))) if names_map[k] in mpose]
    for i1 in range(len(ks)):
        for i2 in range(i1 + 1, len(ks)):
            dy = (mpose[names_map[ks[i2]]][1] - mpose[names_map[ks[i1]]][1] + 180) % 360 - 180
            _sgn.append(np.sign(dy) * (abs(dy) / (STEP * (ks[i2] - ks[i1])) > 0.5))
sgn = 1.0 if not _sgn or np.mean(_sgn) >= 0 else -1.0
n_prop = 0
for b0 in range(0, len(names_map), NA):
    ks = [k for k in range(b0, min(b0 + NA, len(names_map))) if names_map[k] in mpose]
    if not ks: continue
    cpos = np.mean([mpose[names_map[k]][0] for k in ks], 0); kref = ks[0]; yref = mpose[names_map[kref]][1]
    for k in range(b0, min(b0 + NA, len(names_map))):
        if names_map[k] not in mpose:
            mpose[names_map[k]] = ([round(float(cpos[0]), 3), round(float(cpos[1]), 3)], round((yref + sgn * STEP * (k - kref)) % 360, 1)); n_prop += 1
if n_prop: log("지점 전파: 맵 포즈 %d → %d (지점당 %d각·%g° 스텝·방향 %+d)" % (len(mpose) - n_prop, len(mpose), NA, STEP, int(sgn)))
with open(os.path.join(work, "map_pose_%s.jsonl" % hn), "w") as fo:
    for nm, m in zip(names_map, gm):
        if nm in mpose:
            apos, yaw = mpose[nm]; fo.write(json.dumps(dict(house=hn, name=nm, apos=apos, yaw=yaw, prop=(nm not in P), apos_gt=m["apos"], yaw_gt=m["yaw"])) + "\n")
# 맵 프레임의 SfM 3D 점 (픽셀 u,v + 메트릭 깊이) — 초기맵 거리용 (DA 보다 정확, 있는 곳만)
_mu, _mv, _md, _mi, _mn = [], [], [], [], []
for k, nm in enumerate(names_map if ra is not None else []):
    im = next((i for i in ra.images.values() if i.name == nm and i.has_pose), None)
    if im is None: continue
    p2 = [q for q in im.points2D if q.has_point3D()]
    if not p2: continue
    X = np.array([ra.points3D[q.point3D_id].xyz for q in p2]); xy = np.array([q.xy for q in p2])
    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world; z = (cfw * X)[:, 2] * S3
    _mu.append(xy[:, 0]); _mv.append(xy[:, 1]); _md.append(z); _mi.append(np.full(len(z), k)); _mn.append(nm)
if _mu:
    np.savez_compressed(os.path.join(work, "map_points_%s.npz" % hn), u=np.concatenate(_mu).astype(np.float32), v=np.concatenate(_mv).astype(np.float32),
                        d=np.concatenate(_md).astype(np.float32), k=np.concatenate(_mi).astype(np.int32))
    log("맵 프레임 SfM 점 %d (프레임 %d)" % (sum(len(x) for x in _mu), len(_mn)))
json.dump(dict(house=hn, map_reg=(rm.num_reg_images() if rm is not None else sum(1 for k in P if k.startswith("map/"))), n_map=len(maps), map_joint=(not map_ok), live_reg=len(rows), live_reg_raw=n_before, n_live=_nlive0, live_step=a.live_step, cov=cov, vmax=a.vmax, strict=a.strict,
               ate_med=float(np.median(ate)) if len(ate) else None, ate_lt05=float((ate < 0.5).mean()) if len(ate) else None,
               yaw_med=float(np.median(yerr)) if len(yerr) else None, room_hit=(hit / nhit) if nhit else None, n_room=nhit,
               reject_outside=bool(a.reject_outside), sim3_rms=float(rms), sim3_inl=INL, scale=float(s), scale_src=a.scale, align_src=a.align, da_k=a.da_k, mirror=bool(mirror), sec=time.time() - T0), open(os.path.join(work, "summary_%s.json" % hn), "w"), indent=1, default=float)
log("→ %s (%d프레임)" % (out, len(rows)))
