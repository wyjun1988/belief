#!/usr/bin/env python3
"""타임라인 판정 (2026-09-10, 사용자 설계): timeline_prep.jsonl 의 재료를 **시간순 영어 서술 + 전체 프레임(박스 표시)** 로 묶어 VLM 에 한 번 묻는다.
기록 장면(t=0, 어디에 무엇 옆에) → 목격(검출 점수·exemplar·방·확신도, 오검출 포함) → 부재 증거(타겟을 지운 문맥 검색·포즈 자리 향함, 그 프레임의 검출 점수) 를 한 줄씩.
답: 마지막 자리 확인 t · 사라진 뒤 t · 자리를 보여주는 부재 프레임 · 오검출 목록 · 다른 곳 진짜 목격 · 현재 위치 · 부재 확신도 0~100 → 평가기 BUNDLE_JSONL (absent_conf·BUNDLE_TH).
  THOR_ROOT=... PREP_JSONL=$B/scores/timeline_prep.jsonl OUT_JSONL=$B/scores/timeline.jsonl [IMG_W=448 MAX_IMG=24 MAX_OBJ=0] ~/mlx-venv/bin/python scripts/timeline_verdict_mlx.py"""
import os, json, re, time, collections
from PIL import Image, ImageDraw
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4"); ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2b_pilot")
PREP = os.environ["PREP_JSONL"]; OUTJ = os.environ.get("OUT_JSONL", "/tmp/timeline.jsonl"); IMG_W = int(os.environ.get("IMG_W", "448")); MAX_IMG = int(os.environ.get("MAX_IMG", "16"))
MAX_OBJ = int(os.environ.get("MAX_OBJ", "0")); MAXTOK = int(os.environ.get("MAXTOK", "1000")); HOUSES = set(os.environ.get("HOUSES", "").split())
model, processor = load(MODEL); cfg = model.config
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
    prompt = apply_chat_template(processor, cfg, q, num_images=len(paths))
    r = generate(model, processor, prompt, paths, max_tokens=MAXTOK, verbose=False, temperature=0.0)
    return r if isinstance(r, str) else getattr(r, "text", str(r))
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
n_obj = 0; t0 = time.time()
for l in open(PREP):
    r = json.loads(l); hn, oid, a = r["house"], r["oid"], r["type"].replace("_", " ").lower()
    if HOUSES and hn not in HOUSES: continue
    if (hn, oid) in done: continue
    hdr = os.path.realpath(os.path.join(ROOT, hn)); rec_room = r.get("record") or "house"
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
