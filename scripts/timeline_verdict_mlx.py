#!/usr/bin/env python3
"""타임라인 판정 (2026-09-10, 사용자 설계): timeline_prep.jsonl 의 재료를 **시간순 영어 서술 + 전체 프레임(박스 표시)** 로 묶어 VLM 에 한 번 묻는다.
기록 장면(t=0, 어디에 무엇 옆에) → 목격(검출 점수·exemplar·방·확신도, 오검출 포함) → 부재 증거(타겟을 지운 문맥 검색·포즈 자리 향함, 그 프레임의 검출 점수) 를 한 줄씩.
답: 마지막 자리 확인 t · 사라진 뒤 t · 자리를 보여주는 부재 프레임 · 오검출 목록 · 다른 곳 진짜 목격 · 현재 위치 · 부재 확신도 0~100 → 평가기 BUNDLE_JSONL (absent_conf·BUNDLE_TH).
  THOR_ROOT=... PREP_JSONL=$B/scores/timeline_prep.jsonl OUT_JSONL=$B/scores/timeline.jsonl [IMG_W=448 MAX_IMG=16 MAX_OBJ=0] ~/mlx-venv/bin/python scripts/timeline_verdict_mlx.py
  RTX(HF): BACKEND=hf MODEL=Qwen/Qwen3.5-9B THOR_ROOT=<pack> PREP_JSONL=<pack>/timeline_prep_los.jsonl OUT_JSONL=... python scripts/timeline_verdict_mlx.py  (재료는 timeline_pack.py 로 묶어 보냄)"""
import os, json, re, time, collections
from PIL import Image, ImageDraw
BACKEND = os.environ.get("BACKEND", "mlx")            # mlx: Apple Silicon(Qwen3.5-4B mxfp4) · hf: transformers(RTX, MODEL=Qwen/Qwen3.5-9B 등) — 프롬프트·후처리는 동일
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4" if BACKEND == "mlx" else "Qwen/Qwen3.5-9B"); ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2b_pilot")
PREP = os.environ["PREP_JSONL"]; OUTJ = os.environ.get("OUT_JSONL", "/tmp/timeline.jsonl"); IMG_W = int(os.environ.get("IMG_W", "448")); MAX_IMG = int(os.environ.get("MAX_IMG", "16"))
MAX_OBJ = int(os.environ.get("MAX_OBJ", "0")); MAXTOK = int(os.environ.get("MAXTOK", "1000")); HOUSES = set(os.environ.get("HOUSES", "").split())
MODE = os.environ.get("MODE", "timeline")           # timeline: 전체 서술(§166-44/46) · simple: 오라클 형식(§166-48) — 기록 장면(박스)+포즈 검증 자리 프레임 ≤K장만, 목적 서문, "still there?" 하나
K_SPOT = int(os.environ.get("K_SPOT", "4")); REFCROP = os.environ.get("REFCROP", "0") == "1"
if BACKEND == "mlx":
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(MODEL); cfg = model.config
else:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(MODEL); model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ):
        try: done.add(tuple(json.loads(l)[k] for k in ("house", "oid")))
        except Exception: pass
out = open(OUTJ, "a"); TMPD = "/tmp/_tl_%d" % os.getpid(); os.makedirs(TMPD, exist_ok=True)
def prep_img(path, i, box=None, color=(255, 0, 0)):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if box:
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w - 1, box[2]), min(h - 1, box[3])
        if x1 - x0 >= 8 and y1 - y0 >= 8: ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=color, width=max(3, w // 200))
        else: box = None
    s = IMG_W / float(w); im = im.resize((IMG_W, max(8, int(h * s)))); p = os.path.join(TMPD, "%02d.jpg" % i); im.save(p, quality=90); return p
def ask(paths, q):
    if BACKEND == "mlx":
        prompt = apply_chat_template(processor, cfg, q, num_images=len(paths))
        r = generate(model, processor, prompt, paths, max_tokens=MAXTOK, verbose=False, temperature=0.0)
        return r if isinstance(r, str) else getattr(r, "text", str(r))
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in paths] + [{"type": "text", "text": q}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ims = [Image.open(p).convert("RGB") for p in paths]
    inp = processor(text=[text], images=ims, return_tensors="pt").to(model.device)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=MAXTOK, do_sample=False)
    return processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
def parse(txt):
    """마지막으로 닫힌 최상위 {…} 블록(중괄호 짝 맞춤) → json; 후행 쉼표 허용."""
    starts = [i for i, c in enumerate(txt) if c == "{"]
    for st in starts:
        depth = 0
        for i in range(st, len(txt)):
            if txt[i] == "{": depth += 1
            elif txt[i] == "}":
                depth -= 1
                if depth == 0:
                    blk = txt[st:i + 1]
                    try: return json.loads(blk)
                    except Exception:
                        try: return json.loads(re.sub(r",\s*([}\]])", r"\1", blk))
                        except Exception: break
    return None
def article(t): return ("an " if t[:1] in "aeiou" else "a ") + t
def crop_img(path, i, box):
    im = Image.open(path).convert("RGB"); w, h = im.size; cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2; r = max(int(0.17 * w), int(max(box[2] - box[0], box[3] - box[1]) * 0.9))
    x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r); c = im.crop((x0, y0, min(w, x0 + 2 * r), min(h, y0 + 2 * r))).resize((IMG_W, IMG_W)); p = os.path.join(TMPD, "%02d.jpg" % i); c.save(p, quality=90); return p
def simple_one(r, hdr, rec_room, a):
    """오라클 형식(§166-48)을 파이프라인 프레임으로: 기록 장면(OWL 박스) [+확대] + 포즈 검증(LOS·마커) 자리 프레임 중 마커가 큰 순 K장(시간순) · 목적 서문 · still_there 하나."""
    if r.get("record_k") is None or not r.get("record_box"): return None
    rec_path = os.path.join(hdr, "map", "%04d.jpg" % r["record_k"])
    if not os.path.exists(rec_path): return None
    # 확인된 마지막 목격(기록 방에서 검출 점수 ≥0.30) 이후의 자리 프레임만
    last_true = max([s_["t"] for s_ in r["sightings"] if s_.get("score", 0) >= 0.30 and s_.get("room") == rec_room], default=-1)
    ctx = [c for c in r["context"] if c.get("geo") and c.get("spot_box") and c["t"] > last_true and os.path.exists(os.path.join(hdr, "live", "%06d.jpg" % c["t"]))]
    ctx = sorted(ctx, key=lambda c: -((c["spot_box"][2] - c["spot_box"][0]) * (c["spot_box"][3] - c["spot_box"][1])))[:K_SPOT]; ctx.sort(key=lambda c: c["t"])
    if len(ctx) < 1: return dict(house=r["house"], oid=r["oid"], type=r["type"], record=rec_room, imgs=[], spot_seen=[], absent_conf=0.0, at_spot="unsure", else_t=None, else_room=None, conf="low", n_spot=0, raw="(no pose-verified spot frame)")
    paths = [prep_img(rec_path, 1, r["record_box"])]; imgs = [["record", -1, rec_room, {}]]
    if REFCROP: paths.append(crop_img(rec_path, 2, r["record_box"])); imgs.append(["refcrop", -1, rec_room, {}])
    for c in ctx:
        n = len(paths) + 1; paths.append(prep_img(os.path.join(hdr, "live", "%06d.jpg" % c["t"]), n, c["spot_box"], color=(0, 120, 255))); imgs.append(["spot", c["t"], c.get("room"), dict(geo=1, marker=1)])
    sp = [i + 1 for i, x in enumerate(imgs) if x[0] == "spot"]
    q = ("You help a home robot answer \"Where is the %s?\". The robot recorded the %s at a place during a scan (Image 1, inside the red box) and later walked past that place again. "
         "Your job is to decide from the later views whether the %s is still there. Most household objects do not move, so answer \"no\" only when the recorded place is clearly visible and the %s is clearly not there; "
         "if the place is partly hidden, too far, or the object is small and hard to see, answer \"unsure\" rather than \"no\". A wrong \"no\" makes the robot search the whole house for nothing.\n\n"
         "%sImages %d-%d are later views of that same place in time order; the blue box marks where the %s should appear from that camera position.\n"
         "Is the %s still at its recorded place in the later images? Answer with JSON only (the JSON must be the last line): "
         "{\"still_there\": \"yes\"|\"no\"|\"unsure\", \"seen_in\": [image numbers where the %s is visible], \"confidence\": 0-100}"
         % (a, a, a, a, ("Image 2 is a zoomed-in crop of Image 1 around the %s. " % a) if REFCROP else "", sp[0], sp[-1], a, a, a))
    txt = ask(paths, q); js = parse(txt) or {}; st = str(js.get("still_there", "")).lower()
    try: cf = float(js.get("confidence", 0) or 0)
    except Exception: cf = 0.0
    def _ints(v): return [int(x) for x in (v if isinstance(v, list) else [v]) if str(x).strip().lstrip("-").isdigit()]
    return dict(house=r["house"], oid=r["oid"], type=r["type"], record=rec_room, imgs=imgs, n_spot=len(sp), last_true_t=last_true, seen_in=_ints(js.get("seen_in", [])),
                spot_seen=(sp if st == "no" else []), absent_conf=(cf if st == "no" else 0.0), at_spot=(st if st in ("yes", "no", "unsure") else "unsure"), else_t=None, else_room=None, conf=("high" if cf >= 70 else "low"), raw=txt[:600])
n_obj = 0; t0 = time.time()
for l in open(PREP):
    r = json.loads(l); hn, oid, a = r["house"], r["oid"], r["type"].replace("_", " ").lower()
    if HOUSES and hn not in HOUSES: continue
    if (hn, oid) in done: continue
    hdr = os.path.realpath(os.path.join(ROOT, hn)); rec_room = r.get("record") or "house"
    if MODE == "simple":
        row = simple_one(r, hdr, rec_room, a)
        if row is None: continue
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush(); n_obj += 1
        if n_obj % 10 == 0: print("  %d 타겟 · %.0fs" % (n_obj, time.time() - t0), flush=True)
        if MAX_OBJ and n_obj >= MAX_OBJ: break
        continue
    events = [("sight", s) for s in r["sightings"]] + [("ctx", c) for c in r["context"]]
    events.sort(key=lambda e: e[1]["t"])
    if len(events) > MAX_IMG - 1:                               # 상한: 문맥은 유지, 목격은 점수 상위로
        keep_ctx = [e for e in events if e[0] == "ctx"]; keep_s = sorted([e for e in events if e[0] == "sight"], key=lambda e: -e[1]["score"])[:max(0, MAX_IMG - 1 - len(keep_ctx))]
        events = sorted(keep_ctx + keep_s, key=lambda e: e[1]["t"])
    paths = []; lines = []; imgs = []
    if r.get("record_k") is not None:
        p = os.path.join(hdr, "map", "%04d.jpg" % r["record_k"])
        if os.path.exists(p):
            paths.append(prep_img(p, 1, r.get("record_box"))); imgs.append(["record", -1, rec_room, {}])
            where = ", ".join(("%s the %s" % (rel, t_)) for rel, t_, s_ in (r.get("neighbors") or [])[:3])
            lines.append("Image 1 (scan, t=0): the %s was recorded in the %s%s. %s" % (a, rec_room, (", " + where) if where else "", "It is inside the red box." if r.get("record_box") else "It was detected here (no box available)."))
    if not paths:
        lines.append("(No scan image is available; the %s was recorded in the %s.)" % (a, rec_room))
    for kind, e in events:
        t = e["t"]; p = os.path.join(hdr, "live", "%06d.jpg" % t)
        if not os.path.exists(p): continue
        n = len(paths) + 1; room = e.get("room") or "unknown room"; conf = e.get("room_conf", "unknown")
        if kind == "sight":
            paths.append(prep_img(p, n, e.get("box"))); imgs.append(["sight", t, room, dict(score=e["score"], sim_ex=e.get("sim_ex", 0.0))])
            lines.append("Image %d (t=%d s): the detector reports %s inside the red box (detector score %.2f, appearance similarity to the recorded %s %.2f); the room classifier says %s (%s)."
                         % (n, t, article(a), e["score"], a, e.get("sim_ex", 0.0), room, conf))
        else:
            mb = e.get("spot_box")
            if mb and not (min(mb[2], 10**6) - max(mb[0], 0) >= 8 and min(mb[3], 10**6) - max(mb[1], 0) >= 8 and mb[2] > 0 and mb[3] > 0): mb = None
            paths.append(prep_img(p, n, mb, color=(0, 120, 255))); imgs.append(["ctx", t, room, dict(geo=int(bool(e.get("geo"))), marker=int(bool(mb)), sim_ctx=(e.get("sim_ctx") or 0.0), det=e.get("det_score", 0.0))])
            why = ("the camera pose says it is looking at the recorded spot, and the blue box marks where the spot should appear" if e.get("geo") and mb else
                   "the camera pose says it is looking at the recorded spot" if e.get("geo") else "it resembles the recorded scene with the %s removed (similarity %.2f); no pose is available, so it may not be the spot at all" % (a, e.get("sim_ctx") or 0.0))
            lines.append("Image %d (t=%d s): this frame was retrieved because %s; the detector score for %s here is %.2f; the room classifier says %s (%s)."
                         % (n, t, why, article(a), e.get("det_score", 0.0), room, conf))
    if len(paths) < 2: continue
    q = ("A home robot is asked: \"Where is the %s?\" Below is everything the robot has, in time order. Detector reports can be false alarms (a similar object, or nothing), "
         "the room classifier can be wrong, and a retrieved frame may not actually show the recorded spot. Compare the images with Image 1 (same furniture, walls, layout) and reconstruct what happened to the %s.\n\n%s\n\n"
         "Answer with JSON only:\n{\"last_seen_at_spot_t\": <t of the last image where the %s is clearly at its recorded spot, or null>,\n"
         "\"gone_after_t\": <t after which it is no longer at the recorded spot, or null>,\n"
         "\"absence_frames\": [<image numbers that clearly show the recorded spot with the %s missing>],\n"
         "\"false_sightings\": [<image numbers whose detector report is not the %s>],\n"
         "\"elsewhere\": [{\"image\": <n>, \"t\": <t>, \"room\": \"<room>\"}] (images where the %s itself is clearly seen somewhere other than its recorded spot),\n"
         "\"current_location\": \"at the recorded spot\" | \"<room name>\" | \"unknown\",\n"
         "\"absent_confidence\": <0-100, how sure you are that the %s is no longer at its recorded spot>}\n"
         "First write 2-4 sentences of reasoning (which images really show the recorded spot, which detector reports are real), then the JSON on a single line without line breaks. "
         "Do not mark an image as an absence frame unless you can verify it shows the same furniture and walls as Image 1 (a blue box, when present, marks where the spot should be). "
         "If you cannot verify the spot in any image, set absence_frames to [] and absent_confidence below 30. If the %s is still at its spot in the latest verified image, set gone_after_t to null and absent_confidence low.") % (a, a, "\n".join(lines), a, a, a, a, a, a)
    txt = ask(paths, q); js = parse(txt) or {}
    def _ints(v): return [int(x) for x in (v if isinstance(v, list) else [v]) if str(x).strip().lstrip("-").isdigit()]
    absf = [i for i in _ints(js.get("absence_frames", [])) if 1 <= i <= len(imgs) and imgs[i-1][0] == "ctx" or (1 <= i <= len(imgs) and imgs[i-1][0] == "sight")]
    absf = [i for i in absf if imgs[i-1][0] != "record"]
    try: ac = float(js.get("absent_confidence", 0) or 0)
    except Exception: ac = 0.0
    els = []
    for e in (js.get("elsewhere") or []):
        try:
            n = int(e.get("image")); 
            if 1 <= n <= len(imgs) and imgs[n-1][0] == "sight": els.append(dict(image=n, t=imgs[n-1][1], room=(e.get("room") or imgs[n-1][2])))
        except Exception: pass
    CTX_SIM_MIN = float(os.environ.get("CTX_SIM_MIN", "0.9")); SIGHT_MIN = float(os.environ.get("SIGHT_MIN", "0.15"))
    fs = set(_ints(js.get("false_sightings", [])))
    true_s = [(i + 1, im_) for i, im_ in enumerate(imgs) if im_[0] == "sight" and (i + 1) not in fs and im_[3].get("score", 0) >= SIGHT_MIN]
    at_spot_s = [n for n, im_ in true_s if im_[2] == rec_room]; last_true = max([imgs[n-1][1] for n in at_spot_s], default=-1)
    A = [n for n in absf if imgs[n-1][0] == "ctx" and imgs[n-1][1] > last_true and (imgs[n-1][3].get("marker") or imgs[n-1][3].get("sim_ctx", 0) >= CTX_SIM_MIN)]
    els_t = [e for e in els if imgs[e["image"]-1][2] and imgs[e["image"]-1][2] != rec_room and e["image"] not in fs and imgs[e["image"]-1][1] > last_true]
    last = els_t[-1] if els_t else None
    ac_model = ac; ac = ac if A else 0.0
    row = dict(house=hn, oid=oid, type=r["type"], record=rec_room, n_img=len(paths), imgs=imgs,
               last_seen_t=js.get("last_seen_at_spot_t"), gone_after_t=js.get("gone_after_t"), absent_conf=ac, current=str(js.get("current_location", "")),
               false_sightings=_ints(js.get("false_sightings", [])), elsewhere=els,
               # 평가기 BUNDLE 훅 호환 필드
               spot_seen=A, abs_model=absf, absent_conf_model=ac_model, last_true_t=last_true, at_spot=("no" if ac >= 70 and A else ("yes" if ac < 30 else "unsure")), else_img=(last["image"] if last else 0), else_t=(last["t"] if last else None),
               else_room=(imgs[last["image"]-1][2] if last else None), conf="high", raw=txt)
    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush(); n_obj += 1
    if n_obj % 10 == 0: print("  %d 타겟 · %.0fs" % (n_obj, time.time() - t0), flush=True)
    if MAX_OBJ and n_obj >= MAX_OBJ: break
print("TIMELINE_DONE %d 타겟 · %.0fs" % (n_obj, time.time() - t0))
