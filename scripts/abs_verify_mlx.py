#!/usr/bin/env python3
"""③ 부재 검증 — 스캔 문맥 검색 + VLM 크롭 질문 (mlx, Apple Silicon). GT 는 쓰지 않는다(라벨은 평가기에서).

    THOR_ROOT=data/hssd90_c4e2 A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ ROOM_JSONL=$B/scores/room_embed_clip.jsonl \\
      OUT_JSONL=$B/scores/abs_verify.jsonl ~/mlx-venv/bin/python scripts/abs_verify_mlx.py

물체(타입 단일 타겟)마다: 초기맵 기록(방·자리) → 자리를 향한 **지도 프레임**(스캔 포즈)의 CLIP 임베딩 = 문맥 →
라이브 앵커 중 기록 방(임베딩 카메라방)에서 문맥과 가장 닮은 top-N = "자리를 보는 프레임" → 가장 최근 K_LATE 장과 가장 이른 K_EARLY 장을
OWL 박스(그 타입 argmax) 크롭 + 넓은 크롭으로 잘라 검증기에 "(A) 타입 (B) 대안 (C) 둘 다 아님" 을 묻는다. s_ac = A − max(B, C).
산출: {house, oid, type, record, n_map_facing, late:[[t, s_ac_box, s_ac_wide],…], early:[…]} — 판정은 평가기(ABS_VERIFY_JSONL)에서.
"""
import glob, json, os, math, collections
import numpy as np
from PIL import Image
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
ROOT = os.environ.get("THOR_ROOT", "data/hssd20"); A3P = os.environ.get("A3_PREFIX", "/tmp/hs_a_"); QCP = os.environ.get("QC_PREFIX", "/tmp/hs_q_")
RJ = os.environ.get("ROOM_JSONL"); EMB = os.path.expanduser(os.environ.get("EMB_CACHE", "~/khcache/room_embed"))
OUTJ = os.environ.get("OUT_JSONL", "/tmp/abs_verify.jsonl")
TOPN = int(os.environ.get("TOPN", "10")); K_LATE = int(os.environ.get("K_LATE", "3")); K_EARLY = int(os.environ.get("K_EARLY", "3"))
CTX_DIST = float(os.environ.get("CTX_DIST", "3.0")); CTX_ANG = float(os.environ.get("CTX_ANG", "35")); ONLY_MOVED = os.environ.get("ONLY_MOVED", "0") == "1"
TMP = "/tmp/_absc_%d.jpg" % os.getpid()

model, processor = load(MODEL); cfg = model.config; tok = processor.tokenizer
IDS = {t: tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B", "C")}
def logits(img, q):
    prompt = apply_chat_template(processor, cfg, q, num_images=1)
    inp = prepare_inputs(processor, images=[img], prompts=[prompt], image_token_index=getattr(cfg, "image_token_index", None))
    out = model(inp["input_ids"], inp["pixel_values"], mask=inp.get("attention_mask"), **{k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask")})
    lg = out.logits[0, -1]; mx.eval(lg); return lg
def words(t): return t.replace("_", " ").lower()
def s_ac(cp, a, b):
    lg = logits(cp, "Which is in this image: (A) %s, (B) %s, or (C) neither? Answer only A, B, or C." % (a, b))
    return float(lg[IDS["A"]] - max(float(lg[IDS["B"]]), float(lg[IDS["C"]])))

RETR = os.environ.get("RETR", "clip")          # clip: 문맥 top-N · pose: PnP 포즈 기하(자리 향함) · both: 합집합(기하 우선, 부족분 clip) · nbr: 자리 주변 정적 개체(exemplar)가 같이 보이는 프레임 · posenbr: pose ∪ nbr (카메라방 게이트 없음)
AXP = os.environ.get("AX_PREFIX"); NBR_R = float(os.environ.get("NBR_R", "2.0")); NBR_TH = float(os.environ.get("NBR_TH", "0.03")); NBR_MIN = int(os.environ.get("NBR_MIN", "2"))
POSES = collections.defaultdict(dict)
if os.environ.get("POSE_JSONL"):
    for l in open(os.path.expanduser(os.environ["POSE_JSONL"])):
        r = json.loads(l); POSES[r["house"]][int(r["t"])] = r
def facing(ax, az, yaw, spot, dmax=4.0, amax=35.0):
    dx, dz = spot[0] - ax, spot[1] - az; d = math.hypot(dx, dz)
    return 0.3 <= d <= dmax and abs((math.degrees(math.atan2(dx, dz)) - yaw + 180) % 360 - 180) <= amax
rooms = collections.defaultdict(dict)
if RJ:
    for l in open(RJ):
        r = json.loads(l); rooms[r["house"]][r["t"]] = r["room"]
ACTX = {}                                    # RETR=anchor: anchor_ctx.py 의 문맥 프레임(앵커 인스턴스 박스) — 포즈 없이 "옛 자리" 프레임을 찾는다 (§166-29)
if RETR == "anchor":
    for _l in open(os.path.expanduser(os.environ["ANCHOR_CTX_JSONL"])):
        _r = json.loads(_l); ACTX[(_r["house"], _r["oid"])] = _r
done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ):
        try: r = json.loads(l); done.add((r["house"], r["oid"]))
        except Exception: pass
out = open(OUTJ, "a")
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd)); fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]; vocab, nT = list(za["vocab"]), int(za["nT"])
    BXa = za["bx"] if "bx" in za.files else None; QT = list(zq["tg"])
    hdr = os.path.realpath(hd); g = json.load(open(hdr + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gf = os.path.join(hdr, "room_groups.json"); gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}
    grp = lambda r: gm.get(r, r) if r else r
    try: El = np.load(f"{EMB}/clip_{hn}_live.npy"); Em = np.load(f"{EMB}/clip_{hn}_map.npy")
    except FileNotFoundError: print(hn, "임베딩 없음 → 건너뜀", flush=True); continue
    if len(El) != len(g["live"]) or len(Em) != len(g["map"]): print(hn, "임베딩 불일치 → 건너뜀", flush=True); continue
    im = json.load(open(os.path.join(hdr, os.environ.get("INITMAP_FILE", "initmap_owl.json")))); inst = collections.defaultdict(list)
    for it in im: inst[it["type"]].append((it["w"], grp(it.get("room")), it.get("pos")))
    arm = np.array([grp(rooms[hn].get(int(t), live[int(t)].get("room"))) for t in ts])
    _XSc = _XA = None
    if RETR in ("nbr", "posenbr") and AXP and os.path.exists(AXP + hn + ".npz"):
        _zx = np.load(AXP + hn + ".npz", allow_pickle=True); _XA = list(_zx["anch"]); _XSc = _zx["s"] - np.median(_zx["s"], axis=0, keepdims=True)
        _stat = g.get("scene_meta", {}).get("static", {})
    lv = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(hdr, "live", "*.jpg"))}
    cnt = collections.Counter(v["type"] for v in g["gt0"].values()); moved = {m["oid"] for m in g["moves"]}
    n_obj = 0
    for j, oid in enumerate(QT):
        v0 = g["gt0"].get(oid)
        if not v0 or not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab or v0["type"] not in inst: continue
        if ONLY_MOVED and oid not in moved: continue
        if (hn, oid) in done: continue
        ti = vocab.index(v0["type"]); w, record, spot = (max(inst[v0["type"]]) if os.environ.get("INST_PICK", "first") == "maxw" else inst[v0["type"]][0])   # first: 파일 순서(재군집 순위) · maxw: 점수합 최대
        if spot is None: continue
        fac = [k for k, m in enumerate(g["map"]) if 0.3 <= math.hypot(spot[0]-m["apos"][0], spot[1]-m["apos"][1]) <= CTX_DIST
               and abs((math.degrees(math.atan2(spot[0]-m["apos"][0], spot[1]-m["apos"][1])) - m["yaw"] + 180) % 360 - 180) <= CTX_ANG]
        rec = dict(house=hn, oid=oid, type=v0["type"], record=record, n_map_facing=len(fac), late=[], early=[])
        inroom = [i for i in range(len(ts)) if arm[i] == record]
        allf = list(range(len(ts)))
        if RETR == "anchor":
            ac = ACTX.get((hn, oid)); tidx = {int(t_): i_ for i_, t_ in enumerate(ts)}
            fr_ = sorted([f for f in (ac["frames"] if ac else []) if int(f[0]) in tidx and int(f[0]) in lv], key=lambda f: f[0])
            rec.update(anchor_id=(ac or {}).get("anchor_id"), anchor_room=(ac or {}).get("anchor_room"), n_ctx=len(fr_))
            picks = [("late", f) for f in fr_[-K_LATE:]] + [("early", f) for f in fr_[:-K_LATE][:K_EARLY]]
            for role, f in picks:
                t = int(f[0]); i = tidx[t]; img = Image.open(lv[t]).convert("RGB"); W, H = img.size
                x0, y0, x1, y1 = f[1:5]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h2 = max(48, int(max(x1 - x0, y1 - y0) * 0.6))   # 앵커 박스 크기의 크롭(그 위·옆) + 넓은 크롭
                a = words(v0["type"]); order = np.argsort(-S[i, :nT]); b = words(vocab[int(order[1] if order[0] == ti else order[0])])
                sc = []
                for hh in (h2, max(h2 * 2, W // 4)):
                    img.crop((max(0, int(cx)-hh), max(0, int(cy)-hh), min(W, int(cx)+hh), min(H, int(cy)+hh))).resize((336, 336)).save(TMP, quality=92)
                    sc.append(round(s_ac(TMP, a, b), 3))
                rec[role].append([t, sc[0], sc[1], round(float(f[5]), 3), 1])
            # 전반 증거가 모자라면 **스캔 프레임**(타겟이 그 앵커 옆에 있던 기록 장면)을 전반 행으로 — 기록 자체가 "거기 있었다"는 증거다. t 는 음수로 표시.
            for f in ((ac or {}).get("scan_frames") or [])[:max(0, K_EARLY - len(rec["early"]))]:
                mp_ = os.path.join(hdr, "map", "%04d.jpg" % int(f[0]))
                if not os.path.exists(mp_): continue
                img = Image.open(mp_).convert("RGB"); W, H = img.size; x0, y0, x1, y1 = f[1:5]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h2 = max(48, int(max(x1 - x0, y1 - y0) * 0.65))
                a = words(v0["type"]); b = words(vocab[int(np.argsort(-S[0, :nT])[1])]); sc = []
                for hh in (h2, max(h2 * 2, W // 4)):
                    img.crop((max(0, int(cx)-hh), max(0, int(cy)-hh), min(W, int(cx)+hh), min(H, int(cy)+hh))).resize((336, 336)).save(TMP, quality=92)
                    sc.append(round(s_ac(TMP, a, b), 3))
                rec["early"].append([-1000 - int(f[0]), sc[0], sc[1], 1.0, 1])
            out.write(json.dumps(rec) + "\n"); out.flush(); n_obj += 1
            continue
        if RETR in ("nbr", "posenbr"):
            # 카메라방 게이트 없이: 자리 NBR_R m 안의 정적 개체(스캔) 중 NBR_MIN 개 이상이 exemplar 로 같이 보이는 프레임 (사용자 제안 2026-09-07 — 카메라 위치가 아니라 주변 물체 조합)
            nbr_c = [_XA.index(k) for k, v in _stat.items() if _XA and k in _XA and v.get("pos") and math.hypot(v["pos"][0] - spot[0], v["pos"][2] - spot[1]) <= NBR_R] if _XSc is not None else []
            nbrf = [i for i in allf if nbr_c and int(np.sum(_XSc[i, nbr_c] >= NBR_TH)) >= NBR_MIN]
            geo = [i for i in allf if int(ts[i]) in POSES[hn] and facing(POSES[hn][int(ts[i])]["apos"][0], POSES[hn][int(ts[i])]["apos"][1], POSES[hn][int(ts[i])]["yaw"], spot)] if RETR == "posenbr" else []
            rec["n_nbr_static"] = len(nbr_c); rec["n_nbrf"] = len(nbrf)
        else:
            geo = [i for i in inroom if int(ts[i]) in POSES[hn] and facing(POSES[hn][int(ts[i])]["apos"][0], POSES[hn][int(ts[i])]["apos"][1], POSES[hn][int(ts[i])]["yaw"], spot)] if RETR in ("pose", "both") else []
            nbrf = []
        rec["n_geo"] = len(geo)
        if RETR in ("nbr", "posenbr"):
            cand = sorted(set(geo) | set(nbrf), key=lambda i: ts[i]); sim = np.zeros(len(ts))
            picks = [("late", i) for i in cand[-K_LATE:]] + [("early", i) for i in cand[:-K_LATE][:K_EARLY]] if len(cand) >= 1 else []
            for role, i in picks:
                t = int(ts[i])
                if t not in lv: continue
                img = Image.open(lv[t]).convert("RGB"); W, H = img.size
                if BXa is not None:
                    bcx, bcy, bw, bh = [float(x) * max(W, H) for x in BXa[i, ti]]; h2 = max(48, int(max(bw, bh) * 0.65)); cx, cy = bcx, bcy
                else:
                    cx = (P[i, ti] % pw + .5) / pw * W; cy = (P[i, ti] // pw + .5) / ph * H; h2 = max(64, W // 6)
                a = words(v0["type"]); order = np.argsort(-S[i, :nT]); b = words(vocab[int(order[1] if order[0] == ti else order[0])])
                sc = []
                for hh in (h2, max(h2 * 2, W // 4)):
                    img.crop((max(0, int(cx)-hh), max(0, int(cy)-hh), min(W, int(cx)+hh), min(H, int(cy)+hh))).resize((336, 336)).save(TMP, quality=92)
                    sc.append(round(s_ac(TMP, a, b), 3))
                rec[role].append([t, sc[0], sc[1], 0.0, int(i in geo), int(i in nbrf)])
            out.write(json.dumps(rec) + "\n"); out.flush(); n_obj += 1
            continue
        if fac or (RETR == "pose" and geo):
            sim = (El[ts.astype(int)] @ Em[fac].T).max(1) if fac else np.zeros(len(ts))
            if RETR == "pose": cand = geo
            elif RETR == "both": cand = list(dict.fromkeys(geo + sorted(inroom, key=lambda i: -sim[i])[:TOPN]))[:max(TOPN, len(geo))]
            else: cand = sorted(inroom, key=lambda i: -sim[i])[:TOPN]
            cand.sort(key=lambda i: ts[i])
            picks = [("late", i) for i in cand[-K_LATE:]] + [("early", i) for i in cand[:-K_LATE][:K_EARLY]]
            for role, i in picks:
                t = int(ts[i])
                if t not in lv: continue
                img = Image.open(lv[t]).convert("RGB"); W, H = img.size
                if BXa is not None:
                    bcx, bcy, bw, bh = [float(x) * max(W, H) for x in BXa[i, ti]]; h2 = max(48, int(max(bw, bh) * 0.65)); cx, cy = bcx, bcy
                else:
                    cx = (P[i, ti] % pw + .5) / pw * W; cy = (P[i, ti] // pw + .5) / ph * H; h2 = max(64, W // 6)
                a = words(v0["type"]); order = np.argsort(-S[i, :nT]); b = words(vocab[int(order[1] if order[0] == ti else order[0])])
                sc = []
                for hh in (h2, max(h2 * 2, W // 4)):
                    img.crop((max(0, int(cx)-hh), max(0, int(cy)-hh), min(W, int(cx)+hh), min(H, int(cy)+hh))).resize((336, 336)).save(TMP, quality=92)
                    sc.append(round(s_ac(TMP, a, b), 3))
                rec[role].append([t, sc[0], sc[1], round(float(sim[i]), 3), int(i in geo)])
        out.write(json.dumps(rec) + "\n"); out.flush(); n_obj += 1
    print(hn, "완료 · 물체", n_obj, flush=True)
out.close(); print("→", OUTJ)
