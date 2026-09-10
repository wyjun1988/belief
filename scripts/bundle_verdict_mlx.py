#!/usr/bin/env python3
"""②③ 묶음 판정 (2026-09-10, 사용자 제안): 규칙 게이트 대신 **후보를 다 모아 질문과 함께 VLM 한 번**에 묻는다.
타입 단일 타겟마다 이미지 묶음 = [기록 장면(자리를 향한 지도 프레임)] + [자리를 보는 라이브 프레임 K장(부재 검증기 late/early 목록, 방 게이트 없이)]
+ [이미지 질의(exemplar) 상위 M장 — 어느 방이든]. 각 프레임의 임베딩 카메라방을 글로 붙여 준다.
질문: 어느 프레임이 기록 자리를 실제로 보는가 / 거기에 물체가 있는가 / 다른 프레임에 물체가 보이는가(몇 번). JSON 답 → 평가기(BUNDLE_JSONL).
  THOR_ROOT=... QC_PREFIX=$B/cache/hs2_q_ ROOM_JSONL=$B/scores/room_embed_clip.jsonl ABS_VERIFY_JSONL=$B/scores/abs_verify_gate0.jsonl \\
    INITMAP_FILE=initmap_owl_rc.json OUT_JSONL=$B/scores/bundle.jsonl [K_SPOT=4 M_ELSE=4 IMG_W=448 ONLY_MOVED=0] ~/mlx-venv/bin/python scripts/bundle_verdict_mlx.py
행: {house, oid, type, record, imgs:[[role, t, room]], spot_seen:[img ids], at_spot:'yes|no|unsure', else_img, else_t, else_room, raw}"""
import glob, json, os, math, collections, re, time
import numpy as np
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2b_pilot"); QCP = os.environ.get("QC_PREFIX"); RJ = os.environ.get("ROOM_JSONL"); AVJ = os.environ.get("ABS_VERIFY_JSONL")
IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); OUTJ = os.environ.get("OUT_JSONL", "/tmp/bundle.jsonl")
K_SPOT = int(os.environ.get("K_SPOT", "4")); M_ELSE = int(os.environ.get("M_ELSE", "4")); IMG_W = int(os.environ.get("IMG_W", "448")); ONLY_MOVED = os.environ.get("ONLY_MOVED", "0") == "1"
MAX_OBJ = int(os.environ.get("MAX_OBJ", "0")); CTX_DIST = float(os.environ.get("CTX_DIST", "3.0")); CTX_ANG = float(os.environ.get("CTX_ANG", "35")); MAXTOK = int(os.environ.get("MAXTOK", "160"))
PHJ = os.environ.get("PHANTOM_JSON", os.path.join(ROOT, "phantom_ids.json")); PH = set()
if PHJ and os.path.exists(PHJ):
    _p = json.load(open(PHJ)); PH = {(h, o) for h, v in (_p.get("phantom") or _p).items() for o in (v if isinstance(v, list) else v.get("ids", []))} if isinstance(_p, dict) else set()
def words(t): return t.replace("_", " ").lower()
rooms = collections.defaultdict(dict)
if RJ:
    for l in open(os.path.expanduser(RJ)):
        r = json.loads(l); rooms[r["house"]][int(r["t"])] = r["room"]
ABSV = {}
if AVJ:
    for l in open(os.path.expanduser(AVJ)):
        r = json.loads(l); ABSV[(r["house"], r["oid"])] = r
model, processor = load(MODEL); cfg = model.config
done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ):
        try: done.add(tuple(json.loads(l)[k] for k in ("house", "oid")))
        except Exception: pass
out = open(OUTJ, "a"); TMPD = "/tmp/_bundle_%d" % os.getpid(); os.makedirs(TMPD, exist_ok=True)
def prep(path, i):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = IMG_W / float(w); im = im.resize((IMG_W, max(8, int(h * s)))); p = os.path.join(TMPD, "%02d.jpg" % i); im.save(p, quality=90); return p
def ask(paths, q):
    prompt = apply_chat_template(processor, cfg, q, num_images=len(paths))
    r = generate(model, processor, prompt, paths, max_tokens=MAXTOK, verbose=False, temperature=0.0)
    return r if isinstance(r, str) else getattr(r, "text", str(r))
def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        try: return json.loads(re.sub(r"(\w+):", r'"\1":', m.group(0)).replace("'", '"'))
        except Exception: return None
n_obj = 0; t0 = time.time()
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd)); hdr = os.path.realpath(hd); fq = QCP + hn + ".npz"
    if not os.path.exists(fq): continue
    zq = np.load(fq, allow_pickle=True); SI, ts, QT = zq["si"], [int(t) for t in zq["ts"]], list(zq["tg"])
    g = json.load(open(hdr + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gf = os.path.join(hdr, "room_groups.json"); gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}
    grp = lambda r: gm.get(r, r) if r else r
    lv = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(hdr, "live", "*.jpg"))}
    im = json.load(open(os.path.join(hdr, IMF))); inst = collections.defaultdict(list)
    for it in im: inst[it["type"]].append((it["w"], grp(it.get("room")), it.get("pos")))
    moved = {m["oid"] for m in g["moves"]}
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if (hn, k) not in PH)
    rm = lambda t: grp(rooms[hn].get(int(t), live.get(int(t), {}).get("room")))
    for j, oid in enumerate(QT):
        v0 = g["gt0"].get(oid)
        if not v0 or (hn, oid) in PH or cnt[v0["type"]] > 1 or v0["type"] not in inst: continue
        if ONLY_MOVED and oid not in moved: continue
        if (hn, oid) in done: continue
        w, record, spot = inst[v0["type"]][0]
        if spot is None: continue
        av = ABSV.get((hn, oid)); spot_t = []
        if av:
            late = sorted([int(x[0]) for x in av.get("late", []) if int(x[0]) >= 0], reverse=True); early = sorted([int(x[0]) for x in av.get("early", []) if int(x[0]) >= 0])
            spot_t = (late[:K_SPOT] + early[:max(0, K_SPOT - len(late))])[:K_SPOT]
        # 이미지 질의 상위 M — 자리 프레임 제외, 같은 방 2장 상한(방 다양성)
        order = np.argsort(-SI[:, j]); els = []; per = collections.Counter()
        for i in order:
            t = ts[i]
            if t in spot_t or t not in lv: continue
            r_ = rm(t) or "?"
            if per[r_] >= 2: continue
            per[r_] += 1; els.append((t, float(SI[i, j])))
            if len(els) >= M_ELSE: break
        # 기록 장면: 자리를 향한 지도 프레임 중 가장 가까운 것
        fac = [(math.hypot(spot[0]-m["apos"][0], spot[1]-m["apos"][1]), k) for k, m in enumerate(g["map"])
               if 0.3 <= math.hypot(spot[0]-m["apos"][0], spot[1]-m["apos"][1]) <= CTX_DIST and abs((math.degrees(math.atan2(spot[0]-m["apos"][0], spot[1]-m["apos"][1])) - m["yaw"] + 180) % 360 - 180) <= CTX_ANG]
        recf = os.path.join(hdr, "map", "%04d.jpg" % min(fac)[1]) if fac else None
        imgs = []; paths = []
        if recf and os.path.exists(recf): imgs.append(["record", -1, record]); paths.append(prep(recf, len(paths) + 1))
        for t in spot_t: imgs.append(["spot", t, rm(t)]); paths.append(prep(lv[t], len(paths) + 1))
        for t, s_ in els: imgs.append(["else", t, rm(t)]); paths.append(prep(lv[t], len(paths) + 1))
        if not paths or len(spot_t) + len(els) == 0: continue
        a = words(v0["type"]); lines = []
        sp_ids = [i for i, x in enumerate(imgs, 1) if x[0] == "spot"]; el_ids = [i for i, x in enumerate(imgs, 1) if x[0] == "else"]
        for i, (role, t, r_) in enumerate(imgs, 1):
            if role == "record": lines.append("Image %d (RECORD): the scan view where the %s was recorded, in the %s." % (i, a, r_ or "house"))
            elif role == "spot": lines.append("Image %d (PLACE candidate): a later view that may show that same place; camera was in the %s." % (i, r_ or "unknown room"))
            else: lines.append("Image %d (ELSEWHERE candidate): a later frame where something looks like that %s; camera was in the %s." % (i, a, r_ or "unknown room"))
        q = ("A household robot recorded a %s in the %s (Image 1). Later frames follow, in two groups: PLACE candidates (images %s) that may show the recorded place again, "
             "and ELSEWHERE candidates (images %s) that may show the %s in a different place.\n%s\n"
             "Compare furniture, walls and layout with Image 1. Answer with JSON only: "
             "{\"spot_images\": [PLACE candidate numbers that clearly show the same place as Image 1], "
             "\"object_at_spot\": \"yes\"|\"no\"|\"unsure\" (in those spot images, is the %s still where it was recorded? \"unsure\" if no spot image), "
             "\"object_elsewhere\": ELSEWHERE candidate number where the %s itself is clearly visible in a different place than Image 1, else 0, "
             "\"confidence\": \"high\"|\"low\"}" % (a, record or "house", (("%d-%d" % (sp_ids[0], sp_ids[-1])) if sp_ids else "none"), (("%d-%d" % (el_ids[0], el_ids[-1])) if el_ids else "none"), a, "\n".join(lines), a, a))
        txt = ask(paths, q); js = parse(txt) or {}
        def _ints(v): return [int(x) for x in (v if isinstance(v, list) else [v]) if str(x).strip().lstrip("-").isdigit()]
        sp = [i for i in _ints(js.get("spot_images", [])) if 1 <= i <= len(imgs) and imgs[i-1][0] == "spot"]
        ei = _ints(js.get("object_elsewhere", 0)); ei = ei[0] if ei else 0
        if not (1 <= ei <= len(imgs)) or imgs[ei-1][0] != "else": ei = 0
        rec = dict(house=hn, oid=oid, type=v0["type"], record=record, imgs=imgs, spot_seen=sp, at_spot=str(js.get("object_at_spot", "unsure")).lower(),
                   else_img=ei, else_t=(imgs[ei-1][1] if ei else None), else_room=(imgs[ei-1][2] if ei else None), conf=str(js.get("confidence", "")).lower(), raw=txt[:300])
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush(); n_obj += 1
        if n_obj % 10 == 0: print("  %d 타겟 · %.0fs" % (n_obj, time.time() - t0), flush=True)
        if MAX_OBJ and n_obj >= MAX_OBJ: break
    if MAX_OBJ and n_obj >= MAX_OBJ: break
print("BUNDLE_DONE %d 타겟 · %.0fs" % (n_obj, time.time() - t0))
