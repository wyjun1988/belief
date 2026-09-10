#!/usr/bin/env python3
"""오라클 입력 VLM 프로브 (2026-09-10, 사용자 질문: "오검출 이미지가 섞여서 못 맞추는 건가? GT 프레임만 주면 맞추나?").
검색·검출·박스 없이 **GT 로 고른 프레임만** 준다: 기록 장면 = 타겟이 가장 가까이 보이는 스캔 프레임 · 나중 장면 = GT 포즈가 옛 자리를 향하고(같은 방·≤4 m·≤35°) 이동(있으면) 이후인 라이브 프레임 K장.
질문 하나: "Is the <type> still at its recorded place?" → ① 은 yes 가 정답, ②③(이동 후)은 no 가 정답. 변형 A 박스 없음 / B 기록 장면에만 빨간 박스.
  THOR_ROOT=data/hssd_v2b_pilot OUT_JSONL=/tmp/oracle.jsonl [K=4 VARIANT=A|B IMG_W=448] ~/mlx-venv/bin/python scripts/oracle_vlm_probe.py"""
import os, json, glob, math, re, time, collections, shutil
from PIL import Image, ImageDraw
BACKEND = os.environ.get("BACKEND", "mlx")           # mlx(M2 4B) | hf(RTX, MODEL=Qwen/Qwen3.5-9B 등)
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4" if BACKEND == "mlx" else "Qwen/Qwen3.5-9B"); ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2b_pilot"); OUTJ = os.environ.get("OUT_JSONL", "/tmp/oracle.jsonl")
K = int(os.environ.get("K", "4")); VAR = os.environ.get("VARIANT", "A"); IMG_W = int(os.environ.get("IMG_W", "448")); MAX_OBJ = int(os.environ.get("MAX_OBJ", "0"))
PRE = os.environ.get("PREAMBLE", "0") == "1"       # 목적·결과를 앞에 설명 + 보수적 판정 지시
EVID = os.environ.get("EVID", "0") == "1"          # 나중 장면마다 자리까지 거리·화면 내 방향(포즈 유래) 글로
MOBP = os.environ.get("MOB", "0") == "1"           # 이동성 사전확률 한 줄(hssd_move.json)
COT = os.environ.get("COT", "0") == "1"            # 장면별 한 문장 서술 후 JSON
REFCROP = os.environ.get("REFCROP", "0") == "1"    # 기록 장면의 물체 주변 확대 크롭을 참조 이미지로 추가
SPOTCROP = os.environ.get("SPOTCROP", "0") == "1"  # 나중 장면마다 자리 주변 확대 크롭 추가(GT 중심 또는 포즈 투영)
_MOBT = json.load(open("data/hssd_move.json")).get("mobility", {}) if MOBP else {}
SELECT_OUT = os.environ.get("SELECT_OUT", "")          # M2: 프레임 선택만 해서 <dir>/select.jsonl + 프레임 복사(RTX 로 보낼 묶음). 모델 안 띄움
SELECT_IN = os.environ.get("SELECT_IN", "")            # RTX: select.jsonl 이 있는 묶음 디렉터리에서 실행(gt.json 불필요)
PHJ = os.path.join(ROOT, "phantom_ids.json"); PH = set()
if os.path.exists(PHJ):
    _p = json.load(open(PHJ)); _d = _p.get("phantom", _p) if isinstance(_p, dict) else {}
    PH = {(h, o) for h, v in _d.items() for o in (v if isinstance(v, list) else v.get("ids", []))} if isinstance(_d, dict) else set()
TMPD = "/tmp/_oracle_%d" % os.getpid(); os.makedirs(TMPD, exist_ok=True)
if not SELECT_OUT:
    if BACKEND == "mlx":
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor = load(MODEL); cfg = model.config
    else:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(MODEL); model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
def facing(ax, az, yaw, spot, dmax=4.0, amax=35.0):
    dx, dz = spot[0] - ax, spot[1] - az; d = math.hypot(dx, dz); return d <= dmax and abs((math.degrees(math.atan2(dx, dz)) - yaw + 180) % 360 - 180) <= amax
def crop(path, i, ctr, frac=0.34):
    im = Image.open(path).convert("RGB"); w, h = im.size; r = int(frac * w / 2); x0, y0 = max(0, int(ctr[0]) - r), max(0, int(ctr[1]) - r)
    c = im.crop((x0, y0, min(w, x0 + 2 * r), min(h, y0 + 2 * r))).resize((IMG_W, IMG_W)); p = os.path.join(TMPD, "%02d.jpg" % i); c.save(p, quality=90); return p
def prep(path, i, ctr=None):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if ctr: r = int(0.09 * w); ImageDraw.Draw(im).rectangle([max(0, ctr[0]-r), max(0, ctr[1]-r), min(w-1, ctr[0]+r), min(h-1, ctr[1]+r)], outline=(255, 0, 0), width=max(3, w // 200))
    s = IMG_W / float(w); im = im.resize((IMG_W, max(8, int(h * s)))); p = os.path.join(TMPD, "%02d.jpg" % i); im.save(p, quality=90); return p
def ask(paths, q):
    if BACKEND == "mlx":
        prompt = apply_chat_template(processor, cfg, q, num_images=len(paths)); r = generate(model, processor, prompt, paths, max_tokens=(600 if COT else 200), verbose=False, temperature=0.0)
        return r if isinstance(r, str) else getattr(r, "text", str(r))
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in paths] + [{"type": "text", "text": q}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); ims = [Image.open(p).convert("RGB") for p in paths]
    inp = processor(text=[text], images=ims, return_tensors="pt").to(model.device)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    return processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
def run_one(hn, oid, typ, moved, rec_path, ctr0, later_paths, meta, evid=None):
    """선택된 프레임으로 질문 한 번. 변형 A 박스 없음 / B 기록 장면에 GT 중심 박스. PREAMBLE/EVID/MOB/COT/REFCROP/SPOTCROP 로 프롬프트 사다리."""
    a = typ.replace("_", " ").lower(); an = ("a " if a[:1] not in "aeiou" else "an ") + a
    paths = [prep(rec_path, 1, ctr0 if VAR == "B" else None)]; lines = []
    if REFCROP: paths.append(crop(rec_path, 2, ctr0)); lines.append("Image 2: a zoomed-in crop of Image 1 around the %s, so you can see what it looks like." % a)
    first_later = len(paths) + 1; spot_idx = []
    for i, p in enumerate(later_paths):
        n = len(paths) + 1; paths.append(prep(p, n)); e = (evid or [None] * len(later_paths))[i] or {}
        desc = "Image %d (t=%s): a later view of the recorded place" % (n, e.get("t", "?"))
        if EVID and e: desc += "; the recorded place is about %.1f m from the camera, toward the %s of the frame" % (e.get("dist", 0.0), e.get("side", "center"))
        lines.append(desc + ".")
        if SPOTCROP and e and e.get("ctr"):
            n2 = len(paths) + 1; paths.append(crop(p, n2, e["ctr"])); lines.append("Image %d: a zoomed-in crop of Image %d around the recorded place." % (n2, n)); spot_idx.append(n2)
    later_idx = [i + 1 for i, x in enumerate(paths) if i + 1 >= first_later]
    pre = ("You help a home robot answer \"Where is the %s?\". The robot recorded the %s at a place during a scan (Image 1) and later walked past that place again. "
           "Your job is to decide from the later views whether the %s is still there. Most household objects do not move, so answer \"no\" only when the recorded place is clearly visible and the %s is clearly not there; "
           "if the place is partly hidden, too far, or the object is small and hard to see, answer \"unsure\" rather than \"no\". A wrong \"no\" makes the robot search the whole house for nothing.\n\n" % (a, a, a, a)) if PRE else ""
    mobl = ""
    if MOBP:
        mv_ = _MOBT.get(a, None); mobl = ("Prior knowledge: %s %s.\n" % (an, "almost never moves" if (mv_ is not None and mv_ < 0.1) else "moves occasionally (mobility %.1f)" % mv_ if mv_ is not None else "may or may not move"))
    cot = "First write one short sentence per later image describing what is at the recorded place in it. Then " if COT else ""
    q = (pre + "Image 1 shows %s at its recorded place in a house%s.\n%s\n%s%sIs the %s still at its recorded place in the later images? %sAnswer with JSON%s: "
         "{\"still_there\": \"yes\"|\"no\"|\"unsure\", \"seen_in\": [image numbers where the %s is visible], \"confidence\": 0-100}"
         % (an, " (inside the red box)" if VAR == "B" else "", "\n".join(lines), mobl, "The later views are in time order. ", a, cot, " only" if not COT else " on the last line", a))
    txt = ask(paths, q); js = parse(txt) or {}; truth = "no" if moved else "yes"; ans = str(js.get("still_there", "")).lower()
    row = dict(house=hn, oid=oid, type=typ, moved=moved, truth=truth, ans=ans, conf=js.get("confidence"), seen_in=js.get("seen_in"), variant=VAR, model=MODEL,
               opts=dict(pre=PRE, evid=EVID, mob=MOBP, cot=COT, refcrop=REFCROP, spotcrop=SPOTCROP), raw=txt[:600]); row.update(meta)
    return row, truth, ans
if SELECT_IN:                                              # RTX 경로: 묶음의 select.jsonl 로 바로
    out = open(OUTJ, "a"); n = 0; t0 = time.time(); stat = collections.Counter()
    for l in open(os.path.join(SELECT_IN, "select.jsonl")):
        r = json.loads(l); rec = os.path.join(SELECT_IN, r["rec_file"]); later = [os.path.join(SELECT_IN, f) for f in r["later_files"]]
        row, truth, ans = run_one(r["house"], r["oid"], r["type"], r["moved"], rec, r["ctr0"], later, dict(n_later=r["n_later"], vis_later=r["vis_later"], d0=r["d0"]))
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush(); n += 1; stat[(truth, ans)] += 1
        if MAX_OBJ and n >= MAX_OBJ: break
    print("ORACLE_DONE %s · %d 타겟 · %.0fs · %s" % (VAR, n, time.time() - t0, dict(stat))); raise SystemExit
def parse(txt):
    for st in [i for i, c in enumerate(txt) if c == "{"]:
        d = 0
        for i in range(st, len(txt)):
            if txt[i] == "{": d += 1
            elif txt[i] == "}":
                d -= 1
                if d == 0:
                    try: return json.loads(re.sub(r",\s*([}\]])", r"\1", txt[st:i+1]))
                    except Exception: break
    return None
out = open(OUTJ, "a") if not SELECT_OUT else None; n = 0; t0 = time.time(); stat = collections.Counter()
if SELECT_OUT: os.makedirs(SELECT_OUT, exist_ok=True); open(os.path.join(SELECT_OUT, "select.jsonl"), "w").close()
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd)); hdr = os.path.realpath(hd); g = json.load(open(hdr + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gm = json.load(open(hdr + "/room_groups.json"))["groups"] if os.path.exists(hdr + "/room_groups.json") else {}; grp = lambda r: gm.get(r, r) if r else r
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if (hn, k) not in PH); mv = {m["oid"]: m for m in g["moves"]}
    for oid, v0 in g["gt0"].items():
        if (hn, oid) in PH or cnt[v0["type"]] > 1: continue
        a = v0["type"].replace("_", " ").lower(); spot = [v0["pos"][0], v0["pos"][2]]; room = grp(v0["room"])
        # 기록 장면: 타겟이 보이는 스캔 프레임 중 가장 가까운 것
        cands = [(m["dist"][oid], k) for k, m in enumerate(g["map"]) if oid in (m.get("ctr") or {}) and oid in (m.get("dist") or {})]
        if not cands: stat["no_record"] += 1; continue
        d0, k0 = min(cands); ctr0 = g["map"][k0]["ctr"][oid]
        t_after = (mv[oid]["t"] if oid in mv else len(live) // 2)
        later = [t for t, m in live.items() if t > t_after and m.get("apos") and m.get("yaw") is not None and grp(m["room"]) == room and facing(m["apos"][0], m["apos"][1], m["yaw"], spot)]
        if len(later) < 2: stat["few_later"] += 1; continue
        later.sort(); step = max(1, len(later) // K); pick = later[::step][:K]
        rec_path = os.path.join(hdr, "map", "%04d.jpg" % k0); later_paths = [os.path.join(hdr, "live", "%06d.jpg" % t) for t in pick]
        meta = dict(n_later=len(later), pick=pick, k0=k0, d0=round(d0, 2), vis_later=sum(1 for t in pick if oid in (live[t].get("vis") or [])))
        evid = []
        for t in pick:                                     # 포즈 유래 증거: 자리까지 거리·화면 내 방향·자리 중심(GT 가시면 GT 중심, 아니면 방위 투영)
            m = live[t]; dx, dz = spot[0] - m["apos"][0], spot[1] - m["apos"][1]; d = math.hypot(dx, dz); b = (math.degrees(math.atan2(dx, dz)) - m["yaw"] + 180) % 360 - 180
            W_ = 768; u = W_ / 2 + (W_ / 2) * math.tan(math.radians(b)); side = "left" if b < -12 else "right" if b > 12 else "center"
            c = (m.get("ctr") or {}).get(oid) or [max(0, min(W_ - 1, u)), W_ / 2]
            evid.append(dict(t=t, dist=round(d, 2), side=side, ctr=[float(c[0]), float(c[1])]))
        if SELECT_OUT:                                     # 프레임 선택만 저장(+복사) → RTX
            os.makedirs(os.path.join(SELECT_OUT, hn), exist_ok=True); rf = "%s/map_%04d.jpg" % (hn, k0); lfs = ["%s/live_%06d.jpg" % (hn, t) for t in pick]
            for src, dst in [(rec_path, rf)] + list(zip(later_paths, lfs)):
                if not os.path.exists(os.path.join(SELECT_OUT, dst)): shutil.copy(src, os.path.join(SELECT_OUT, dst))
            open(os.path.join(SELECT_OUT, "select.jsonl"), "a").write(json.dumps(dict(house=hn, oid=oid, type=v0["type"], moved=oid in mv, rec_file=rf, ctr0=ctr0, later_files=lfs, **meta)) + "\n"); n += 1; continue
        row, truth, ans = run_one(hn, oid, v0["type"], oid in mv, rec_path, ctr0, later_paths, meta, evid)
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush(); n += 1; stat[(truth, ans)] += 1
        if MAX_OBJ and n >= MAX_OBJ: break
    if MAX_OBJ and n >= MAX_OBJ: break
print("ORACLE_DONE %s · %d 타겟 · %.0fs · %s" % (VAR, n, time.time() - t0, dict(stat)))
