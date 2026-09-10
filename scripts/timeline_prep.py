#!/usr/bin/env python3
"""타임라인 판정 재료 (2026-09-10, 사용자 설계): 타입 유일 타겟마다 (1) 기록 장면 = 자리를 향한 스캔 프레임 + OWL 박스 + 이웃 가구("on the table, next to a magazine")
(2) 목격 = 앵커 프레임 전체에서 OWL 점수 ≥ SIGHT_TH 또는 exemplar 유사도 상위인 프레임 **전부**(오검출 포함, 시간 구간 상한) (3) 부재 증거 = 기록 프레임에서 타겟 박스를
지우고(회색) CLIP 으로 검색한 문맥 프레임 + PnP 포즈로 자리를 향한 프레임. 각 프레임에 t·카메라방·확신도·점수를 붙여 jsonl 로 → timeline_verdict_mlx.py 가 영어 서술로 묶어 VLM 에 한 번 묻는다.
  THOR_ROOT=... A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ ROOM_JSONL=$B/scores/room_embed_clip.jsonl POSE_JSONL=$B/pnp/pose_all_room.jsonl \\
    INITMAP_FILE=initmap_owl_rc.json OUT_JSONL=$B/scores/timeline_prep.jsonl [SIGHT_TH=0.10 MAX_SIGHT=12 MAX_CTX=10 MAX_OBJ=0 HOUSES=house_0001] python scripts/timeline_prep.py"""
import os, sys, json, glob, math, collections, time
import numpy as np, torch
from PIL import Image
ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2b_pilot"); A3P = os.environ["A3_PREFIX"]; QCP = os.environ["QC_PREFIX"]; RJ = os.environ.get("ROOM_JSONL"); PJ = os.environ.get("POSE_JSONL")
IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); OUTJ = os.environ.get("OUT_JSONL", "/tmp/timeline_prep.jsonl"); EMB = os.path.expanduser(os.environ.get("EMB_CACHE", "~/khcache/room_embed"))
SIGHT_TH = float(os.environ.get("SIGHT_TH", "0.10")); MAX_SIGHT = int(os.environ.get("MAX_SIGHT", "12")); MAX_CTX = int(os.environ.get("MAX_CTX", "10")); MAX_OBJ = int(os.environ.get("MAX_OBJ", "0"))
HOUSES = set(os.environ.get("HOUSES", "").split()); CTX_DIST = float(os.environ.get("CTX_DIST", "3.0")); CTX_ANG = float(os.environ.get("CTX_ANG", "35")); ONLY_MOVED = os.environ.get("ONLY_MOVED", "0") == "1"
PHJ = os.environ.get("PHANTOM_JSON", os.path.join(ROOT, "phantom_ids.json")); PH = set()
if PHJ and os.path.exists(PHJ):
    _p = json.load(open(PHJ)); _d = _p.get("phantom", _p) if isinstance(_p, dict) else {}
    PH = {(h, o) for h, v in _d.items() for o in (v if isinstance(v, list) else v.get("ids", []))} if isinstance(_d, dict) else set()
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
from transformers import Owlv2Processor, Owlv2ForObjectDetection, CLIPModel, CLIPProcessor
opr = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); omd = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
cpr = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16"); cmd = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval()
def words(t): return t.replace("_", " ").lower()
def owl(img, types, th=0.08):
    inp = opr(text=[["a photo of a " + words(t) for t in types]], images=img, return_tensors="pt").to(DEV)
    with torch.no_grad(): out = omd(**inp)
    S = max(img.size); res = opr.post_process_object_detection(out, threshold=th, target_sizes=torch.tensor([[S, S]]).to(DEV))[0]
    return [(int(l), float(s), [float(v) for v in b]) for l, s, b in zip(res["labels"].cpu(), res["scores"].cpu(), res["boxes"].cpu())]
def clip_emb(img):
    with torch.no_grad(): e = cmd.get_image_features(**cpr(images=img, return_tensors="pt").to(DEV))
    e = e[0].float().cpu().numpy(); return e / (np.linalg.norm(e) + 1e-9)
rooms = collections.defaultdict(dict)
if RJ:
    for l in open(os.path.expanduser(RJ)):
        r = json.loads(l); rooms[r["house"]][int(r["t"])] = (r["room"], r.get("room_argmax"), float(r.get("sim", 0)))
POSES = collections.defaultdict(dict)
if PJ:
    for l in open(os.path.expanduser(PJ)):
        r = json.loads(l); POSES[r["house"]][int(r["t"])] = r
def facing(ax, az, yaw, spot, dmax, amax):
    dx, dz = spot[0] - ax, spot[1] - az; d = math.hypot(dx, dz)
    return d <= dmax and abs((math.degrees(math.atan2(dx, dz)) - yaw + 180) % 360 - 180) <= amax
done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ):
        try: r = json.loads(l); done.add((r["house"], r["oid"]))
        except Exception: pass
out = open(OUTJ, "a"); n_obj = 0; t0 = time.time()
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd)); hdr = os.path.realpath(hd)
    if HOUSES and hn not in HOUSES: continue
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, ts, vocab, nT = za["s"], [int(t) for t in za["ts"]], list(za["vocab"]), int(za["nT"]); BX = za["bx"] if "bx" in za.files else None
    SI, QT = zq["si"], list(zq["tg"])
    g = json.load(open(hdr + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gf = os.path.join(hdr, "room_groups.json"); gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}
    grp = lambda r: gm.get(r, r) if r else r
    lv = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(hdr, "live", "*.jpg"))}
    try: El = np.load(f"{EMB}/clip_{hn}_live.npy"); El = El / (np.linalg.norm(El, axis=1, keepdims=True) + 1e-9)
    except FileNotFoundError: print(hn, "임베딩 없음", flush=True); continue
    _polys = (g.get("scene_meta") or {}).get("polys") or {}
    def _pip(pt, poly):
        x, z = pt; ins = False; n_ = len(poly)
        for k_ in range(n_):
            x1, z1 = poly[k_][0], poly[k_][-1]; x2, z2 = poly[(k_ + 1) % n_][0], poly[(k_ + 1) % n_][-1]
            if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
        return ins
    def _room_at(pt):
        for r_, pl in _polys.items():
            pls = pl if (pl and isinstance(pl[0][0], (list, tuple))) else [pl]
            if any(_pip(pt, q) for q in pls): return grp(r_)
        return None
    im = json.load(open(os.path.join(hdr, IMF))); inst = collections.defaultdict(list)
    for it in im: inst[it["type"]].append((it["w"], grp(it.get("room")), it.get("pos")))
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if (hn, k) not in PH); moved = {m["oid"] for m in g["moves"]}
    def rinfo(t):
        r = rooms[hn].get(int(t)); 
        if not r: return grp(live.get(int(t), {}).get("room")), "unknown"
        return grp(r[0]), ("confident" if r[0] == r[1] else "not confident")
    W0 = H0 = None
    for j, oid in enumerate(QT):
        v0 = g["gt0"].get(oid)
        if not v0 or (hn, oid) in PH or cnt[v0["type"]] > 1 or v0["type"] not in inst or v0["type"] not in vocab: continue
        if ONLY_MOVED and oid not in moved: continue
        if (hn, oid) in done: continue
        w, record, spot = inst[v0["type"]][0]
        if spot is None: continue
        ti = vocab.index(v0["type"]); a = words(v0["type"])
        # ── (1) 기록 장면: 자리를 향한 스캔 프레임 ≤5 장에 OWL → 타겟 점수 최대 프레임·박스·이웃
        fac = sorted([(math.hypot(spot[0]-m["apos"][0], spot[1]-m["apos"][1]), k) for k, m in enumerate(g["map"])
                      if 0.3 <= math.hypot(spot[0]-m["apos"][0], spot[1]-m["apos"][1]) <= CTX_DIST and abs((math.degrees(math.atan2(spot[0]-m["apos"][0], spot[1]-m["apos"][1])) - m["yaw"] + 180) % 360 - 180) <= CTX_ANG])
        if not fac: fac = sorted([(math.hypot(spot[0]-m["apos"][0], spot[1]-m["apos"][1]), k) for k, m in enumerate(g["map"])])[:4]
        best = None
        for d_, k in fac[:5]:
            p = os.path.join(hdr, "map", "%04d.jpg" % k)
            if not os.path.exists(p): continue
            img = Image.open(p).convert("RGB"); W0, H0 = img.size
            dets = owl(img, vocab, 0.06); tg = [x for x in dets if x[0] == ti]
            if tg:
                sc, bx = max(tg, key=lambda x: x[1] * (1 + min(1.0, (x[2][2]-x[2][0]) * (x[2][3]-x[2][1]) / (0.02 * W0 * H0))))[1:]   # 점수 × 크기(작은 박스 불리)
                if best is None or sc > best[1]: best = (k, sc, bx, dets)
        rec = dict(house=hn, oid=oid, type=v0["type"], record=record, spot=[round(spot[0], 2), round(spot[1], 2)], W=W0, H=H0)
        if best:
            k, sc, bx, dets = best; rec.update(record_k=k, record_score=round(sc, 3), record_box=([round(v) for v in bx] if sc >= 0.10 and (bx[2]-bx[0]) >= 16 and (bx[3]-bx[1]) >= 16 else None))
            x0, y0, x1, y1 = bx; tw, th_ = x1 - x0, y1 - y0; nb = []
            STRUCT = {"wall", "floor", "ceiling", "door", "doorframe", "window", "curtain", "ceiling lamp", "wall lamp"}
            for l, s, b in dets:
                if l == ti or s < 0.12 or words(vocab[l]) in STRUCT: continue
                ox = max(0, min(x1, b[2]) - max(x0, b[0])); gapx = max(0, max(x0, b[0]) - min(x1, b[2])); gapy = max(0, max(y0, b[1]) - min(y1, b[3]))
                on = ox > 0.3 * tw and b[1] <= y1 <= b[3] + 0.15 * (b[3] - b[1]) and (b[2] - b[0]) > tw
                near = gapx < 0.6 * max(tw, th_) and gapy < 0.6 * max(tw, th_)
                if on: nb.append(("on", words(vocab[l]), round(s, 2)))
                elif near: nb.append(("next to", words(vocab[l]), round(s, 2)))
            seen = set(); nbs = []
            for rel, t_, s_ in sorted(nb, key=lambda x: (x[0] != "on", -x[2])):
                if t_ in seen or t_ == a: continue
                seen.add(t_); nbs.append([rel, t_, s_])
            rec["neighbors"] = nbs[:3]
        else:
            rec.update(record_k=(fac[0][1] if fac else None), record_score=0.0, record_box=None, neighbors=[])
        # ── (2) 목격: OWL 점수 ≥ SIGHT_TH 또는 exemplar 상위 5% (전부, 시간 구간 상한)
        q95 = float(np.quantile(SI[:, j], 0.95)); sights = []
        for i, t in enumerate(ts):
            s_ = float(S[i, ti]); si_ = float(SI[i, j])
            if s_ >= SIGHT_TH or (si_ >= q95 and s_ >= 0.05):
                b = None
                if BX is not None and ti < BX.shape[1]:
                    cx, cy, bw, bh = [float(x) * max(W0 or 1280, H0 or 960) for x in BX[i, ti]]; b = [round(cx - bw/2), round(cy - bh/2), round(cx + bw/2), round(cy + bh/2)]
                r_, c_ = rinfo(t); sights.append(dict(t=int(t), score=round(s_, 3), sim_ex=round(si_, 3), room=r_, room_conf=c_, box=b))
        if len(sights) > MAX_SIGHT:                        # 시간 구간별 최고 점수만
            bins = collections.defaultdict(list); T = max(ts) + 1
            for s_ in sights: bins[int(MAX_SIGHT * s_["t"] / T)].append(s_)
            sights = sorted([max(v, key=lambda x: x["score"]) for v in bins.values()], key=lambda x: x["t"])[:MAX_SIGHT]
        # ── (3) 부재 증거: 타겟을 지운 기록 프레임의 CLIP 문맥 검색(앵커 프레임) ∪ PnP 포즈 자리 향함
        ctx = {}
        if best:
            img = Image.open(os.path.join(hdr, "map", "%04d.jpg" % best[0])).convert("RGB").copy(); x0, y0, x1, y1 = [int(v) for v in best[2]]
            from PIL import ImageDraw; ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], fill=(128, 128, 128)); e = clip_emb(img)
            sims = El[[t for t in ts]] @ e
            for i in np.argsort(-sims)[:8]:
                t = ts[int(i)]; r_, c_ = rinfo(t); ctx[t] = dict(t=int(t), sim_ctx=round(float(sims[int(i)]), 3), geo=0, room=r_, room_conf=c_)
        geo = [t for t in ts if t in POSES[hn] and facing(POSES[hn][t]["apos"][0], POSES[hn][t]["apos"][1], POSES[hn][t]["yaw"], spot, 4.0, 35.0)]
        # LOS (2026-09-10 파일럿 진단): "자리 방향 4 m·35°" 만으로는 다른 방에서 벽 너머로 본 프레임이 들어와 마커가 벽·문 위에 찍힌다 → 카메라 포즈의 방(평면도 폴리곤) == 기록 방일 때만 자리 증거
        if os.environ.get("LOS", "1") == "1" and _polys:
            geo = [t for t in geo if _room_at(POSES[hn][t]["apos"]) == record]
        # 자리의 높이·크기: 기록 프레임의 박스 + 기록 카메라 포즈(GT 스캔 포즈 허용)로 추정 → 라이브 포즈 프레임에 "자리 예상 박스" 투영 (hfov 90° 가정: fx = W/2)
        FX = float(os.environ.get("FRAME_FX", "0")) or (W0 or 768) / 2.0; CX = (W0 or 768) / 2.0; CY = (H0 or 768) / 2.0; hgt = None; osz = None
        if best and rec.get("record_box") and rec.get("record_k") is not None:
            m0 = g["map"][rec["record_k"]]; d0 = math.hypot(spot[0]-m0["apos"][0], spot[1]-m0["apos"][1]); bx0 = rec["record_box"]
            v = (bx0[1] + bx0[3]) / 2.0; el = -math.atan((v - CY) / FX) + math.radians(m0.get("pitch") or 0.0); hgt = d0 * math.tan(el)
            osz = max(bx0[2]-bx0[0], bx0[3]-bx0[1]) * d0 / FX                    # 물체 크기(m, 근사)
        def spot_box(t):
            if hgt is None: return None
            P_ = POSES[hn][t]; dx, dz = spot[0]-P_["apos"][0], spot[1]-P_["apos"][1]; d = math.hypot(dx, dz)
            if d < 0.3: return None
            b = math.degrees(math.atan2(dx, dz)) - P_["yaw"]; b = (b + 180) % 360 - 180
            if abs(b) > 44: return None
            u = CX + FX * math.tan(math.radians(b)); vv = CY - FX * (hgt / d); half = max(24, FX * osz / d * 0.75)
            return [round(u - half), round(vv - half), round(u + half), round(vv + half)]
        for t in sorted(geo)[-6:]:
            r_, c_ = rinfo(t); d = ctx.setdefault(t, dict(t=int(t), sim_ctx=None, geo=1, room=r_, room_conf=c_)); d["geo"] = 1; d["spot_box"] = spot_box(t)
        ctxl = sorted(ctx.values(), key=lambda x: x["t"])
        if len(ctxl) > MAX_CTX: ctxl = sorted(ctxl, key=lambda x: (x["geo"], x["sim_ctx"] or 0), reverse=True)[:MAX_CTX]; ctxl.sort(key=lambda x: x["t"])
        for c in ctxl: c["det_score"] = round(float(S[ts.index(c["t"]), ti]), 3)   # 그 프레임에서의 타겟 검출 점수(없음의 근거)
        rec.update(sightings=sights, context=ctxl, n_sight_all=int(sum(1 for i in range(len(ts)) if S[i, ti] >= SIGHT_TH)))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush(); n_obj += 1
        if n_obj % 10 == 0: print("  %d 타겟 · %.0fs" % (n_obj, time.time() - t0), flush=True)
        if MAX_OBJ and n_obj >= MAX_OBJ: break
    if MAX_OBJ and n_obj >= MAX_OBJ: break
print("TIMELINE_PREP_DONE %d 타겟 · %.0fs" % (n_obj, time.time() - t0))
