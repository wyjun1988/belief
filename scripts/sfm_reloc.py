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
ap.add_argument("--topk", type=int, default=20, help="vocab-tree 검색 이미지 수")
ap.add_argument("--features", type=int, default=4096)
ap.add_argument("--out", default=None)
ap.add_argument("--redo", action="store_true")
ap.add_argument("--vmax", type=float, default=2.5, help="속도 필터: ±3초 이웃 등록 프레임과의 중앙 거리(m)가 이보다 크면 기권(1fps 보행 ≤1.5m/s). 0=끔")
ap.add_argument("--fast", action="store_true", help="전역 BA 를 덜 자주(1.1→1.3배)·반복 절반 — live 등록 시간 단축(정확도는 4채에서 대조할 것)")
ap.add_argument("--strict", action="store_true", help="PnP 등록 문턱 강화(abs_pose_min_num_inliers 60·inlier_ratio 0.4) — 반복 구조 유령 등록 억제")
ap.add_argument("--matcher", default="vocab", choices=["vocab", "exhaustive"], help="vocab: 순차+vocab-tree 검색(faiss 판 트리 필요) · exhaustive: 전수")
ap.add_argument("--gpu", action="store_true", help="CUDA 박스(RTX)에서 SIFT·매칭을 GPU 로")
a = ap.parse_args()

hd = a.house.rstrip("/"); hn = os.path.basename(hd)
work = a.work or os.path.expanduser("~/khcache/sfm/%s" % hn); os.makedirs(work, exist_ok=True)
g = json.load(open(os.path.join(hd, "gt.json")))
maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg"))
lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg"))
names_map = ["map/" + f for f in maps]; names_live = ["live/" + f for f in lives]
from PIL import Image
W, H = Image.open(os.path.join(hd, "map", maps[0])).size; fx = W / 2.0          # hfov 90° 핀홀
db = os.path.join(work, "db.db"); T0 = time.time()
DEV = pycolmap.Device.cuda if a.gpu else pycolmap.Device.cpu
def log(*x): print("[%5.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)

ap_matcher = a.matcher
mk_ext, mk_mat = os.path.join(work, ".extracted"), os.path.join(work, ".matched_" + ap_matcher)
if a.redo:
    for f in (db, mk_ext, mk_mat):
        if os.path.exists(f): os.remove(f)
if not os.path.exists(mk_ext):
    if os.path.exists(db): os.remove(db)
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = "%g,%g,%g,%g" % (fx, fx, W / 2.0, H / 2.0)
    eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = a.threads; eo.use_gpu = a.gpu; eo.sift.max_num_features = a.features
    pycolmap.extract_features(db, hd, image_names=names_map + names_live, camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model="PINHOLE", reader_options=ro, extraction_options=eo, device=DEV)
    open(mk_ext, "w").close(); log("SIFT 추출 map %d + live %d" % (len(maps), len(lives)))
if not os.path.exists(mk_mat):
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = a.threads; mo.use_gpu = a.gpu
    if ap_matcher == "vocab":
        so = pycolmap.SequentialPairingOptions(); so.overlap = 8; so.loop_detection = False
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=so, device=DEV); log("순차 매칭(overlap 8)")
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
    if a.strict:
        o.mapper.abs_pose_min_num_inliers = 60; o.mapper.abs_pose_min_inlier_ratio = 0.4
    if a.fast:
        o.ba_global_frames_ratio = 1.3; o.ba_global_points_ratio = 1.3
        o.ba_global_max_num_iterations = 30; o.ba_local_max_num_iterations = 15
    return o
def best_rec(recs):
    return max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None

rec_map_dir = os.path.join(work, "rec_map"); rec_all_dir = os.path.join(work, "rec_all")
rm = None
if not a.redo and os.path.isdir(rec_map_dir) and any(os.path.isdir(os.path.join(rec_map_dir, d)) for d in os.listdir(rec_map_dir)):
    rm = pycolmap.Reconstruction(os.path.join(rec_map_dir, sorted(d for d in os.listdir(rec_map_dir) if os.path.isdir(os.path.join(rec_map_dir, d)))[0]))
else:
    os.makedirs(rec_map_dir, exist_ok=True)
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
if not a.redo and _subs:                     # live 등록 결과 캐시 (10~17분짜리) — 가장 큰 모델
    ra = best_rec({d: pycolmap.Reconstruction(os.path.join(rec_all_dir, d)) for d in _subs}); log("live 등록 캐시 사용 (%d모델)" % len(_subs))
elif in_path:
    ra = best_rec(pycolmap.incremental_mapping(db, hd, rec_all_dir, mapper_opts([], True), input_path=in_path))
else:
    ra = best_rec(pycolmap.incremental_mapping(db, hd, rec_all_dir, mapper_opts([], False)))
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
def fit(X, Y):
    s_, R_, t_ = umeyama(X, Y); return s_, R_, t_, np.linalg.norm((s_ * (R_ @ X.T)).T + t_ - Y, axis=1)
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
best = None
for mirror in (False, True):
    M = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
    s, R, t, rms, inl = ransac_sim3(src @ M.T, dst)
    if best is None or (inl, -rms) > (best[6], -best[0]): best = (rms, mirror, M, s, R, t, inl)
rms, mirror, M, S3, R3, T3, INL = best; s = S3
log("sim3 정렬(map %d프레임, RANSAC 0.5m): 인라이어 %.2f · 인라이어 rms %.3fm · 스케일 %.3f · 미러 %s" % (len(src), INL, rms, s, mirror))

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
ate, yerr = np.array(ate), np.array(yerr)
cov = len(rows) / len(lives)
log("live 커버리지 %.2f · ATE 중앙 %.2fm 평균 %.2fm <0.5m %.2f <1m %.2f · yaw 중앙 %.1f° <10° %.2f · 카메라방 적중 %s" % (
    cov, np.median(ate) if len(ate) else -1, ate.mean() if len(ate) else -1, (ate < 0.5).mean() if len(ate) else 0,
    (ate < 1).mean() if len(ate) else 0, np.median(yerr) if len(yerr) else -1, (yerr < 10).mean() if len(yerr) else 0,
    ("%.2f (n=%d)" % (hit / nhit, nhit)) if nhit else "—"))
out = a.out or os.path.join(work, "pose_%s.jsonl" % hn)
with open(out, "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
with open(os.path.join(work, "map_pose_%s.jsonl" % hn), "w") as fo:
    for nm, m in zip(names_map, gm):
        if nm in P:
            apos, yaw = to_ours(*P[nm]); fo.write(json.dumps(dict(house=hn, name=nm, apos=apos, yaw=yaw, apos_gt=m["apos"], yaw_gt=m["yaw"])) + "\n")
json.dump(dict(house=hn, map_reg=(rm.num_reg_images() if rm is not None else 0), n_map=len(maps), map_joint=(not map_ok), live_reg=len(rows), live_reg_raw=n_before, n_live=len(lives), cov=cov, vmax=a.vmax, strict=a.strict,
               ate_med=float(np.median(ate)) if len(ate) else None, ate_lt05=float((ate < 0.5).mean()) if len(ate) else None,
               yaw_med=float(np.median(yerr)) if len(yerr) else None, room_hit=(hit / nhit) if nhit else None, n_room=nhit,
               sim3_rms=float(rms), sim3_inl=INL, scale=float(s), mirror=bool(mirror), sec=time.time() - T0), open(os.path.join(work, "summary_%s.json" % hn), "w"), indent=1, default=float)
log("→ %s (%d프레임)" % (out, len(rows)))
