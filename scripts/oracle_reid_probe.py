#!/usr/bin/env python3
"""② 오라클 프로브 — 인스턴스 재식별 (2026-09-10 밤, 사용자 지적: ② 도 LLM 으로 평가해야). 기록 장면(참조, 빨간 박스) + 후보 프레임들(검출 박스):
양성 = 이동 후 새 방에서 GT 로 실제 보이는 프레임(가장 가까운 ≤2) · 음성 = 같은 타입이 OWL 로 잡혔지만 GT 로는 그 물체가 없는 앵커 프레임(오검출, ≤2).
질문 하나: "which images show the same <type> as Image 1?" → 재발견 정답률(양성 잡음) · 오검출 거부율(음성 안 잡음).
  THOR_ROOT=data/hssd_v2b_pilot A3_PREFIX=$B/cache/hs2_a_ OUT_JSONL=/tmp/reid.jsonl [PREAMBLE=1 IMG_W=448] ~/mlx-venv/bin/python scripts/oracle_reid_probe.py"""
import os, json, glob, math, re, time, collections
import numpy as np
from PIL import Image, ImageDraw
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4"); ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2b_pilot"); A3P = os.environ["A3_PREFIX"]; OUTJ = os.environ.get("OUT_JSONL", "/tmp/reid.jsonl")
IMG_W = int(os.environ.get("IMG_W", "448")); PRE = os.environ.get("PREAMBLE", "1") == "1"; NEG_TH = float(os.environ.get("NEG_TH", "0.10")); MAX_OBJ = int(os.environ.get("MAX_OBJ", "0"))
PHJ = os.path.join(ROOT, "phantom_ids.json"); PH = set()
if os.path.exists(PHJ):
    _p = json.load(open(PHJ)); _d = _p.get("phantom", _p) if isinstance(_p, dict) else {}
    PH = {(h, o) for h, v in _d.items() for o in (v if isinstance(v, list) else v.get("ids", []))} if isinstance(_d, dict) else set()
model, processor = load(MODEL); cfg = model.config; TMPD = "/tmp/_reid_%d" % os.getpid(); os.makedirs(TMPD, exist_ok=True)
def prep(path, i, box=None, ctr=None, color=(255, 0, 0)):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if box is None and ctr: r = int(0.09 * w); box = [ctr[0]-r, ctr[1]-r, ctr[0]+r, ctr[1]+r]
    if box:
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w-1, box[2]), min(h-1, box[3])
        if x1 - x0 >= 8 and y1 - y0 >= 8: ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=color, width=max(3, w // 200))
    s = IMG_W / float(w); im = im.resize((IMG_W, max(8, int(h * s)))); p = os.path.join(TMPD, "%02d.jpg" % i); im.save(p, quality=90); return p
def ask(paths, q):
    prompt = apply_chat_template(processor, cfg, q, num_images=len(paths)); r = generate(model, processor, prompt, paths, max_tokens=400, verbose=False, temperature=0.0)
    return r if isinstance(r, str) else getattr(r, "text", str(r))
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
out = open(OUTJ, "a"); n = 0; t0 = time.time(); TP = FN = FP = TN = 0
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd)); hdr = os.path.realpath(hd); fa = A3P + hn + ".npz"
    if not os.path.exists(fa): continue
    za = np.load(fa, allow_pickle=True); S, ts, vocab, nT = za["s"], [int(t) for t in za["ts"]], list(za["vocab"]), int(za["nT"]); BX = za["bx"] if "bx" in za.files else None
    g = json.load(open(hdr + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gm = json.load(open(hdr + "/room_groups.json"))["groups"] if os.path.exists(hdr + "/room_groups.json") else {}; grp = lambda r: gm.get(r, r) if r else r
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if (hn, k) not in PH); mv = {m["oid"]: m for m in g["moves"]}
    for oid, v0 in g["gt0"].items():
        if (hn, oid) in PH or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        a = v0["type"].replace("_", " ").lower(); ti = vocab.index(v0["type"]); W_ = 768
        cands = [(m["dist"][oid], k) for k, m in enumerate(g["map"]) if oid in (m.get("ctr") or {}) and oid in (m.get("dist") or {})]
        if not cands: continue
        d0, k0 = min(cands); ctr0 = g["map"][k0]["ctr"][oid]
        t_mv = mv[oid]["t"] if oid in mv else None
        # 양성: 이동 후 새 방에서 GT 가시(가장 가까운 2)
        pos = []
        if t_mv is not None:
            pos = sorted([(m["dist"].get(oid, 99), t) for t, m in live.items() if t > t_mv and oid in (m.get("vis") or []) and oid in (m.get("ctr") or {})])[:2]
        # 음성: OWL 이 그 타입을 잡았는데(≥NEG_TH) GT 로는 그 물체가 안 보이는 앵커 프레임(점수 상위 2)
        neg = sorted([(float(S[i, ti]), t) for i, t in enumerate(ts) if S[i, ti] >= NEG_TH and oid not in (live[t].get("vis") or []) and (t_mv is None or t > t_mv)], reverse=True)[:2]
        if not pos and not neg: continue
        items = [("pos", t, live[t]["ctr"][oid], None) for _, t in pos] + [("neg", t, None, ([float(x) * W_ for x in BX[ts.index(t), ti]] if BX is not None else None)) for _, t in neg]
        paths = [prep(os.path.join(hdr, "map", "%04d.jpg" % k0), 1, ctr=ctr0)]; lines = []; roles = []
        for role, t, ctr, bx in items:
            i = len(paths) + 1; box = ([bx[0]-bx[2]/2, bx[1]-bx[3]/2, bx[0]+bx[2]/2, bx[1]+bx[3]/2] if bx else None)
            paths.append(prep(os.path.join(hdr, "live", "%06d.jpg" % t), i, box=box, ctr=(ctr if not box else None))); roles.append((i, role, t))
            lines.append("Image %d (t=%d s): a later frame in which a detector reported %s inside the red box; camera in the %s." % (i, t, ("a " if a[:1] not in "aeiou" else "an ") + a, grp(live[t]["room"]) or "house"))
        pre = ("You help a home robot answer \"Where is the %s?\". Image 1 shows the specific %s the robot recorded (inside the red box). Later, a detector flagged %s-like objects in other frames; some are that same %s moved to a new place, others are a different %s or a false alarm. "
               "Compare shape, color, size and parts with Image 1.\n" % (a, a, a, a, a)) if PRE else "Image 1 shows the recorded %s (red box).\n" % a
        q = pre + "\n".join(lines) + ("\nWhich later images show the SAME %s as Image 1 (not just any %s)? Answer with JSON only (the JSON must be the last line): {\"same\": [image numbers], \"confidence\": 0-100}" % (a, a))
        txt = ask(paths, q); js = parse(txt) or {}
        same = set(int(x) for x in (js.get("same") or []) if str(x).strip().isdigit())
        res = []
        for i, role, t in roles:
            hit = i in same; res.append(dict(image=i, role=role, t=t, said_same=hit))
            if role == "pos": TP += hit; FN += (not hit)
            else: FP += hit; TN += (not hit)
        out.write(json.dumps(dict(house=hn, oid=oid, type=v0["type"], moved=t_mv is not None, n_pos=len(pos), n_neg=len(neg), results=res, conf=js.get("confidence"), raw=txt[:400]), ensure_ascii=False) + "\n"); out.flush(); n += 1
        if MAX_OBJ and n >= MAX_OBJ: break
    if MAX_OBJ and n >= MAX_OBJ: break
print("REID_DONE %d 타겟 · %.0fs · 재발견 잡음 %d/%d (%.2f) · 오검출 거부 %d/%d (%.2f)" % (n, time.time() - t0, TP, TP + FN, TP / max(1, TP + FN), TN, TN + FP, TN / max(1, TN + FP)))
