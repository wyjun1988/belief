#!/usr/bin/env python3
"""공존키 문맥 v2 — 실제 검출(OWLv2 다중 상자) + CLIP 인스턴스 대조로 "옛 자리를 본 프레임" 을 포즈 없이 찾는다 (§166-100, 2026-09-21, RTX 9-53 용).

v1(cokey_ctx.py)은 exemplar 캐시(패치 위치 오차 47px·가시성 정밀도 0.68)만 써서 문맥 정밀도가 0.2~0.3 이었다. v2 는 키 인스턴스의 참조 크롭을 스캔에서 만들고
라이브 앵커마다 OWL 상자 + CLIP 코사인으로 "어느 인스턴스인지" 를 정한 뒤, 키 3개 이상이면 2D 재국소화(x,z,yaw), 2개면 선분 보간으로 자리를 화면에 찍는다.
  키 = 기록 자리 반경 NBR_R 의 정적 물체(scene_meta.static) 중 이동성 < MOB_MAX · 타겟 타입과 단어 안 겹침.
  참조 = 스캔 카메라 포즈(지도 단계 재료)가 키를 향한 스캔 프레임 ≤3장에서 OWL 최고 상자(기대 화면 x 와 가장 가까운 것) → CLIP ViT-B/16 임베딩.
  THOR_ROOT=data/hssd_c3big REC_JSONL=.../rec_sel.jsonl OUT_JSONL=... [HOUSES="house_0001 ..."] [DEVICE=cuda] python scripts/cokey_ctx2.py
출력: anchor_ctx.jsonl 형식(abs_verify_mlx RETR=anchor). 진단(GT 포즈, 선택엔 미사용): 문맥 프레임 중 자리를 향한 비율.
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from owl_compat import owl_post   # 2026-09-22 API 호환
import os, sys, json, glob, math, time, collections, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from owl_alias import alias
ROOT = os.environ.get("THOR_ROOT", "data/hssd_c3big"); REC = os.path.expanduser(os.environ["REC_JSONL"]); OUT = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/cokey_ctx2.jsonl"))
PRIOR = os.environ.get("PRIOR_JSON", "data/hssd_move.json"); HOUSES = set(os.environ.get("HOUSES", "").split())
NBR_R = float(os.environ.get("NBR_R", "4.0")); MOB_MAX = float(os.environ.get("MOB_MAX", "0.1")); OWL_TH = float(os.environ.get("OWL_TH", "0.12")); TAU = float(os.environ.get("CLIP_TAU", "0.80")); TAU_M = float(os.environ.get("CLIP_MARGIN", "0.02"))
RES_MAX = float(os.environ.get("CK_RES", "6.0")); OBJ_M = float(os.environ.get("CK_OBJ_M", "0.35")); PERP_MAX = float(os.environ.get("CK_PERP", "1.5")); MAXF = int(os.environ.get("CK_MAXF", "40")); STRIDE = int(os.environ.get("ANCH_STRIDE", "4"))
FRAME_W = float(os.environ.get("FRAME_W", "768")); FX = float(os.environ.get("FRAME_FX", "0")) or FRAME_W / 2.0; BATCH = int(os.environ.get("BATCH", "8")); KVIEW = int(os.environ.get("K_VIEWS", "3"))
DEV = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
from transformers import Owlv2Processor, Owlv2ForObjectDetection, CLIPModel, CLIPProcessor
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval(); cpp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
mob = json.load(open(PRIOR)).get("mobility", {})
def sp(t): return "a photo of a " + alias(t).replace("_", " ").lower()
def words(t): return set(t.replace("_", " ").lower().split())
def facing(ap, yaw, pt, dmax=4.0, amax=35.0):
    dx, dz = pt[0] - ap[0], pt[1] - ap[1]; d = math.hypot(dx, dz)
    return 0.3 <= d <= dmax and abs((math.degrees(math.atan2(dx, dz)) - yaw + 180) % 360 - 180) <= amax
def bearing_u(ap, yaw, pt):
    b = (math.degrees(math.atan2(pt[0] - ap[0], pt[1] - ap[1])) - yaw + 180) % 360 - 180
    return FRAME_W / 2.0 + math.tan(math.radians(max(-80, min(80, b)))) * FX
@torch.no_grad()
def owl_batch(ims, types, topk=5):
    """이미지 묶음 × 타입 목록 → 각 이미지마다 {type: [(box xyxy px, score), …] 상위 topk}"""
    inp = op(text=[[sp(t) for t in types]] * len(ims), images=ims, return_tensors="pt").to(DEV)
    out = on(**inp); W, H = ims[0].size
    res = owl_post(op, out, threshold=OWL_TH, target_sizes=torch.tensor([[H, W]] * len(ims)).to(DEV))
    R = []
    for r in res:
        by = collections.defaultdict(list)
        for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()): by[types[int(l)]].append((b, float(s)))
        R.append({t: sorted(v, key=lambda x: -x[1])[:topk] for t, v in by.items()})
    return R
def crop(im, b, m=0.15):
    x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0; W, H = im.size
    return im.crop((int(max(0, x0 - m * w)), int(max(0, y0 - m * h)), int(min(W, max(x1 + m * w, x0 + 8))), int(min(H, max(y1 + m * h, y0 + 8)))))
@torch.no_grad()
def clip_emb(crops):
    if not crops: return np.zeros((0, 512), np.float32)
    e = cm.get_image_features(**cpp(images=crops, return_tensors="pt").to(DEV)); e = e / e.norm(dim=-1, keepdim=True)
    return e.float().cpu().numpy()
def resect(U, X):
    """키 화면 x(U) 와 바닥 3D(X, n×2) → (잔차°, 카메라 xz, yaw) 최소 잔차 해. yaw 2° 격자."""
    pb = np.degrees(np.arctan((U - FRAME_W / 2.0) / FX)); best = None
    for yaw in range(0, 360, 2):
        b = np.radians(yaw + pb); d = np.c_[np.sin(b), np.cos(b)]; M = np.zeros((2, 2)); v = np.zeros(2)
        for dk, xk in zip(d, X): Pm = np.eye(2) - np.outer(dk, dk); M += Pm; v += Pm @ xk
        try: c = np.linalg.solve(M, v)
        except Exception: continue
        rel = X - c; dep = (rel * d).sum(1)
        if np.any(dep < 0.3): continue
        res = float(np.mean(np.abs((np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) - np.degrees(b) + 180) % 360 - 180)))
        if best is None or res < best[0]: best = (res, c, float(yaw))
    return best
recs = [json.loads(l) for l in open(REC)]; by_h = collections.defaultdict(list)
for r in recs: by_h[r["house"]].append(r)
fo = open(OUT, "w"); st = collections.Counter(); prec = collections.defaultdict(list); T0 = time.time()
for hn in sorted(by_h):
    if HOUSES and hn not in HOUSES: continue
    hd = os.path.join(ROOT, hn); g = json.load(open(hd + "/gt.json")); stat = (g.get("scene_meta") or {}).get("static") or {}; live = {m["t"]: m for m in g["live"]}; gt0 = g.get("gt0", {})
    mp = g.get("map") or []; mapf = sorted(glob.glob(hd + "/map/*.jpg")); lvf = {int(os.path.basename(p)[:-4]): p for p in glob.glob(hd + "/live/*.jpg")}
    # 1) 타겟별 키 후보 → 집 단위 키 집합
    tkeys = {}
    for r in by_h[hn]:
        rp = r.get("rec_pos"); tw = words(r["type"]); ks = []
        if rp:
            for k, v in stat.items():
                if not v.get("pos") or mob.get(v["type"], 0.0) >= MOB_MAX or (words(v["type"]) & tw): continue
                if math.hypot(v["pos"][0] - rp[0], v["pos"][2] - rp[1]) <= NBR_R: ks.append(k)
        tkeys[r["oid"]] = ks
    allk = sorted(set(k for ks in tkeys.values() for k in ks))
    if not allk:
        for r in by_h[hn]: fo.write(json.dumps(dict(house=hn, oid=r["oid"], type=r["type"], anchor_id="cokey2", anchor_room=r.get("record"), rel=[0.5, 0.5], n_ctx=0, n_keys=0, frames=[])) + "\n")
        st["집: 키 없음"] += 1; continue
    # 2) 키 참조 크롭: 스캔 포즈가 키를 향한 프레임 ≤KVIEW 장에서 OWL 최고 상자(기대 x 최근접)
    need = collections.defaultdict(list)     # map idx → [key ids]
    for k in allk:
        pos = (stat[k]["pos"][0], stat[k]["pos"][2]); cand = []
        for i, m in enumerate(mp):
            if i >= len(mapf) or not m.get("apos") or m.get("yaw") is None: continue
            if facing(m["apos"], m["yaw"], pos, dmax=4.0, amax=30.0): cand.append((math.hypot(pos[0] - m["apos"][0], pos[1] - m["apos"][1]), i))
        for _, i in sorted(cand)[:KVIEW]: need[i].append(k)
    ref = collections.defaultdict(list)      # key → [emb]
    for i in sorted(need):
        im = Image.open(mapf[i]).convert("RGB"); types = sorted(set(stat[k]["type"] for k in need[i]))
        det = owl_batch([im], types)[0]; m = mp[i]; crops = []; owners = []
        for k in need[i]:
            t = stat[k]["type"]; u_exp = bearing_u(m["apos"], m["yaw"], (stat[k]["pos"][0], stat[k]["pos"][2]))
            bs = [(abs((b[0] + b[2]) / 2 - u_exp), b, s) for b, s in det.get(t, [])]
            if not bs: continue
            du, b, s = min(bs)
            if du > 120: continue
            crops.append(crop(im, b)); owners.append(k)
        for k, e in zip(owners, clip_emb(crops)): ref[k].append(e)
    keys = [k for k in allk if ref.get(k)]
    st["키 후보"] += len(allk); st["참조 확보 키"] += len(keys)
    if len(keys) < 2:
        for r in by_h[hn]: fo.write(json.dumps(dict(house=hn, oid=r["oid"], type=r["type"], anchor_id="cokey2", anchor_room=r.get("record"), rel=[0.5, 0.5], n_ctx=0, n_keys=len([k for k in tkeys[r["oid"]] if k in ref]), frames=[])) + "\n")
        continue
    E = {k: np.stack(ref[k]) for k in keys}; ktypes = sorted(set(stat[k]["type"] for k in keys)); bytype = collections.defaultdict(list)
    for k in keys: bytype[stat[k]["type"]].append(k)
    # 3) 라이브 앵커: OWL(집 키 타입) → 크롭 CLIP → 인스턴스 배정
    ts = sorted(t for t in lvf if t % STRIDE == 0); visk = {}     # t → {key: (u, v, box, sim)}
    for bi in range(0, len(ts), BATCH):
        tb = ts[bi:bi + BATCH]; ims = [Image.open(lvf[t]).convert("RGB") for t in tb]; dets = owl_batch(ims, ktypes)
        crops = []; meta = []
        for im, t, det in zip(ims, tb, dets):
            for typ, lst in det.items():
                for b, s in lst: crops.append(crop(im, b)); meta.append((t, typ, b, s))
        embs = clip_emb(crops); per = collections.defaultdict(dict)
        for (t, typ, b, s), e in zip(meta, embs):
            cands = bytype.get(typ, [])
            sims = sorted(((float((E[k] @ e).max()), k) for k in cands), reverse=True)
            if not sims or sims[0][0] < TAU: continue
            if len(sims) >= 2 and sims[0][0] - sims[1][0] < TAU_M: continue
            k = sims[0][1]; u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if k not in per[t] or per[t][k][3] < sims[0][0]: per[t][k] = (u, v, b, sims[0][0])
        for t in tb: visk[t] = per.get(t, {})
    # 4) 타겟별 문맥 프레임
    for r in by_h[hn]:
        oid, typ, rp = r["oid"], r["type"], r.get("rec_pos"); ks = [k for k in tkeys[oid] if k in E]
        row = dict(house=hn, oid=oid, type=typ, anchor_id="cokey2", anchor_room=r.get("record"), rel=[0.5, 0.5], n_keys=len(ks), frames=[])
        if rp is None or len(ks) < 2: row["n_ctx"] = 0; fo.write(json.dumps(row) + "\n"); st["타겟: 키 <2"] += 1; continue
        spot = np.array(rp, float); frames = []
        for t in ts:
            vis = [(k, visk[t][k]) for k in ks if k in visk[t]]
            if len(vis) < 2: continue
            U = np.array([x[1][0] for x in vis]); V = np.array([x[1][1] for x in vis]); X = np.array([[stat[k]["pos"][0], stat[k]["pos"][2]] for k, _ in vis])
            if len(vis) >= 3:
                best = resect(U, X)
                if best is None or best[0] > RES_MAX: st["프레임: 재국소화 실패"] += 1; continue
                _, c, yaw = best
                if not facing(c, yaw, spot): st["프레임: 포즈상 자리 안 향함"] += 1; continue
                u = bearing_u(c, yaw, spot); dist = math.hypot(spot[0] - c[0], spot[1] - c[1]); scale = FX / max(dist, 0.3); nk = len(vis)
            else:
                (k1, a1), (k2, a2) = vis; p1 = X[0]; p2 = X[1]; d = p2 - p1; L = float(np.linalg.norm(d))
                if L < 0.3: continue
                lam = float(np.dot(spot - p1, d) / (L * L)); perp = float(abs(np.cross(d, spot - p1)) / L)
                if perp > PERP_MAX or not (-0.5 <= lam <= 1.5): st["프레임: 2키 배치 밖"] += 1; continue
                u = float(U[0] + lam * (U[1] - U[0])); scale = float(abs(U[1] - U[0]) / L); nk = 2
                if scale < 40: st["프레임: 척도 미달"] += 1; continue
            if not (0.06 * FRAME_W <= u <= 0.94 * FRAME_W): st["프레임: 예측점 화면 밖"] += 1; continue
            v = float(np.mean(V)); h2 = max(48.0, OBJ_M * scale); sim = float(np.mean([x[1][3] for x in vis]))
            frames.append([t, u - 2 * h2, v - 2 * h2, u + 2 * h2, v + 2 * h2, sim, nk])
        frames = sorted(sorted(frames, key=lambda f: -(f[6] * 10 + f[5]))[:MAXF], key=lambda f: f[0])
        row["n_ctx"] = len(frames); row["frames"] = [[f[0], round(f[1], 1), round(f[2], 1), round(f[3], 1), round(f[4], 1), round(f[5], 4)] for f in frames]
        fo.write(json.dumps(row) + "\n"); st["타겟: 문맥 ≥2" if len(frames) >= 2 else "타겟: 문맥 <2"] += 1
        gp = (gt0.get(oid) or {}).get("pos")
        if gp and frames:
            fac = [f for f in frames if live.get(f[0], {}).get("apos") and live[f[0]].get("yaw") is not None and facing(live[f[0]]["apos"], live[f[0]]["yaw"], (gp[0], gp[2]))]
            prec["전체"].append(len(fac) / len(frames))
    fo.flush(); print("  %s 완료 · 키 %d/%d · %.0fs" % (hn, len(keys), len(allk), time.time() - T0), flush=True)
fo.close()
print("집계:", dict(st))
if prec["전체"]: print("진단(GT 포즈): 문맥 프레임이 자리를 향한 비율 — 중앙 %.2f · 평균 %.2f (n=%d)" % (np.median(prec["전체"]), np.mean(prec["전체"]), len(prec["전체"])))
print("COKEY2_DONE →", OUT)
