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
    return o
def best_rec(recs):
    return max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None

rec_map_dir = os.path.join(work, "rec_map"); rec_all_dir = os.path.join(work, "rec_all")
rm = None
if not a.redo and os.path.isdir(rec_map_dir) and any(os.path.isdir(os.path.join(rec_map_dir, d)) for d in os.listdir(rec_map_dir)):
    rm = pycolmap.Reconstruction(os.path.join(rec_map_dir, sorted(os.listdir(rec_map_dir))[0]))
else:
    os.makedirs(rec_map_dir, exist_ok=True)
    rm = best_rec(pycolmap.incremental_mapping(db, hd, rec_map_dir, mapper_opts(names_map, False)))
if rm is None: sys.exit("map 재구성 실패")
log("map 재구성: 등록 %d/%d · 점 %d · 재투영 %.2fpx" % (rm.num_reg_images(), len(maps), rm.num_points3D(), rm.compute_mean_reprojection_error()))
# map 재구성 디렉터리 안의 모델 경로(0/ 같은 하위)
sub = [d for d in sorted(os.listdir(rec_map_dir)) if os.path.isdir(os.path.join(rec_map_dir, d))]
in_path = os.path.join(rec_map_dir, sub[0]) if sub else rec_map_dir
os.makedirs(rec_all_dir, exist_ok=True)
ra = best_rec(pycolmap.incremental_mapping(db, hd, rec_all_dir, mapper_opts([], True), input_path=in_path))
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
best = None
for mirror in (False, True):
    M = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
    s, R, t = umeyama(src @ M.T, dst)
    al = (s * (R @ (src @ M.T).T)).T + t; rms = float(np.sqrt(((al - dst) ** 2).sum(1).mean()))
    if best is None or rms < best[0]: best = (rms, mirror, M, s, R, t)
rms, mirror, M, s, R, t = best
log("sim3 정렬(map %d프레임): rms %.3fm · 스케일 %.3f · 미러 %s" % (len(src), rms, s, mirror))

def to_ours(c, v):
    c2 = s * (R @ (M @ c)) + t; v2 = R @ (M @ v)
    return [round(float(c2[0]), 3), round(float(c2[2]), 3)], round(float(np.degrees(np.arctan2(v2[0], v2[2])) % 360), 1)

# ── 평가: live ATE·yaw·카메라방 적중 ──
live = {m["t"]: m for m in g["live"]}
rooms = g.get("rooms") or {}
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
polys = {rid: poly_of(r) for rid, r in rooms.items()}; polys = {k: v for k, v in polys.items() if v}
rows, ate, yerr, hit, nhit = [], [], [], 0, 0
for f in lives:
    t = int(f[:-4]); nm = "live/" + f; m = live.get(t)
    if nm not in P or m is None: continue
    apos, yaw = to_ours(*P[nm]); rows.append(dict(house=hn, t=t, apos=apos, yaw=yaw))
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
json.dump(dict(house=hn, map_reg=rm.num_reg_images(), n_map=len(maps), live_reg=len(rows), n_live=len(lives), cov=cov,
               ate_med=float(np.median(ate)) if len(ate) else None, ate_lt05=float((ate < 0.5).mean()) if len(ate) else None,
               yaw_med=float(np.median(yerr)) if len(yerr) else None, room_hit=(hit / nhit) if nhit else None, n_room=nhit,
               sim3_rms=rms, scale=s, mirror=mirror, sec=time.time() - T0), open(os.path.join(work, "summary_%s.json" % hn), "w"), indent=1)
log("→ %s (%d프레임)" % (out, len(rows)))
