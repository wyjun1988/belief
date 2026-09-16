#!/usr/bin/env python3
"""기록 자리 선택 — **채택 판정기**로 후보를 겨루게 한다 (2026-09-16).

지금까지 기록방을 못 고친 이유: 후보를 가르는 신호가 검출점수·사전확률·방위뿐이고 셋 다 포화다
(§166-76·77, 군집·순위·투영·가중치 쓸기 전부 ≤6행). 안 써본 신호는 **인스턴스 외형 대조**다.
기록 자리 판정기는 "이게 머그인가"(AUC 0.779)를 묻지만, 채택 판정기는 **"이게 저것과 같은 머그인가"**
를 묻고 실측 AUC 0.9155 다. 학습 방향 그대로 쓴다 — 참조=지도 크롭, 후보=라이브 크롭.

각 초기맵 후보의 스캔 뷰 크롭을 참조로 두고, **채택 필터를 통과한 라이브 프레임 상위 K장**과 대조해
"같은 물체" 마진을 합산한다. 진짜 자리의 크롭만 라이브 목격과 일치할 것이라는 가정.

  THOR_ROOT=data/hssd_v2 INITMAP_FILE=initmap_k8.json ADAPTER=~/khcache/adopt4b/lora_adopt_qwen3_5-4b \
  VERIFY_JSONL=~/khcache/bench-v2full/scores/t1_m0.jsonl A3_PREFIX=~/khcache/bench-v2full/cache/hs2_a_ \
  HOUSES="..." K_LIVE=3 OUT_JSONL=~/khcache/record_vote.jsonl python scripts/lora_record_vote.py
산출: {house, type, pos, room, vote} — vote = 라이브 K장과의 마진 합(높을수록 진짜 자리).
"""
import os, json, glob, time, collections
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2"); IMF = os.environ.get("INITMAP_FILE", "initmap_k8.json")
ADAPTER = os.path.expanduser(os.environ.get("ADAPTER", "")); MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-4B")
DEV = os.environ.get("DEVICE", "mps"); OUT = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/record_vote.jsonl"))
VJ = os.path.expanduser(os.environ["VERIFY_JSONL"]); A3P = os.path.expanduser(os.environ["A3_PREFIX"])
VTH, VTH2 = float(os.environ.get("VERIFY_TH", "2.069")), float(os.environ.get("VERIFY_TH2", "0.887"))
K_LIVE = int(os.environ.get("K_LIVE", "3")); IMG_W = int(os.environ.get("IMG_W", "448"))
HOUSES = [h for h in os.environ.get("HOUSES", "").split() if h]
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
tk = getattr(proc, "tokenizer", proc)
YES, NO = tk.encode("yes", add_special_tokens=False)[0], tk.encode("no", add_special_tokens=False)[0]
def art(t): return ("an " if t[:1] in "aeiou" else "a ") + t
PF = '{"same_object": "'
def boxed(src, box):
    if not os.path.exists(src): return None
    im = Image.open(src).convert("RGB"); W, H = im.size
    if box:
        x0, y0, x1, y1 = [max(0, min(W - 1 if i % 2 == 0 else H - 1, int(v))) for i, v in enumerate(box)]
        if x1 > x0 and y1 > y0:
            ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=max(3, W // 150))
    sc = IMG_W / float(W); hh = max(28, int(round(H * sc / 28.0)) * 28)   # 모양 고정(MPS 재컴파일 방지)
    return im.resize((IMG_W, hh))
def ask(ref, cand, typ):
    q = ("Image 1 shows %s at its recorded place in a house (inside the red box). Image 2 is a later view from elsewhere "
         "in the same house, where an object detector proposed %s (inside the red box). The object may have been moved, "
         "so a different room is possible. Judge the object itself, not the room: is the thing in the red box in image 2 "
         'the same %s as in image 1? Answer with JSON only: {"same_object": "yes"|"no", "confidence": 0-100}'
         % (art(typ), art(typ), typ))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": q}]}]
    try: t = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: t = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if "<think>" not in t[-60:]: t = t + ("" if t.endswith("\n") else "\n") + "<think>\n\n</think>\n\n"
    inp = proc(text=[t + PF], images=[ref, cand], return_tensors="pt").to(model.device)
    with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1); return float(lp[YES] - lp[NO])
VSC = {}
for l in open(VJ):
    d = json.loads(l); VSC[(d["house"], d["oid"])] = d.get("scored") or []
hds = sorted(glob.glob(os.path.join(ROOT, "house_*")))
if HOUSES: hds = [h for h in hds if os.path.basename(h) in HOUSES]
n = 0; t0 = time.time(); st = collections.Counter()
with open(OUT, "w") as fo:
    for hd in hds:
        hn = os.path.basename(hd); rp = os.path.join(hd, "anchor_registry.json")
        zp = A3P + hn + ".npz"
        if not (os.path.exists(rp) and os.path.exists(zp)): st["재료 없음"] += 1; continue
        g = json.load(open(hd + "/gt.json")); z = np.load(zp, allow_pickle=True)
        ts = z["ts"]; vocab = [str(x) for x in z["vocab"]]
        cnt = collections.Counter(v["type"] for v in g["gt0"].values())
        reg = json.load(open(rp)); byt = collections.defaultdict(list)
        for it in reg: byt[it["type"]].append(it)
        for oid, v0 in g["gt0"].items():
            ty = v0["type"]
            if cnt[ty] > 1 or ty not in vocab: continue
            cands = byt.get(ty) or []
            if len(cands) < 2: continue
            ti = vocab.index(ty)
            # 라이브 앵커: 채택 필터 통과 프레임 상위 K
            rows = [e for e in VSC.get((hn, oid), []) if e[1] >= VTH and (len(e) < 3 or e[2] >= VTH2)]
            lives = []
            for e in rows[:K_LIVE]:
                i = int(e[0])
                if i >= len(ts): continue
                t = int(ts[i]); p = f"{hd}/live/%06d.jpg" % t
                if not os.path.exists(p): continue
                W = float(Image.open(p).size[0]); bx = [float(x) * W for x in z["bx"][i, ti]]
                bb = [bx[0] - bx[2] / 2, bx[1] - bx[3] / 2, bx[0] + bx[2] / 2, bx[1] + bx[3] / 2] if bx[2] > 1 else None
                im = boxed(p, bb)
                if im is not None: lives.append(im)
            if not lives: st["라이브 없음"] += 1; continue
            for it in cands:
                v = (it.get("views") or [{}])[0]
                k, b = v.get("k"), v.get("box")
                if k is None or not b: st["뷰 없음"] += 1; continue
                ref = boxed(f"{hd}/map/%04d.jpg" % int(k), b)
                if ref is None: continue
                ms = [ask(ref, lv, ty.replace("_", " ").lower()) for lv in lives]
                fo.write(json.dumps(dict(house=hn, oid=oid, type=ty, pos=it.get("pos"), room=it.get("room"),
                                         vote=round(float(np.mean(ms)), 3), n_live=len(ms))) + "\n")
                n += 1
                if n % 100 == 0:
                    print("  %d · %.0fs (%.2fs/건) · %s" % (n, time.time() - t0, (time.time() - t0) / n, dict(st)), flush=True)
                    fo.flush()
                    if DEV == "mps":
                        try: torch.mps.empty_cache()
                        except Exception: pass
print("RECORD_VOTE_DONE %d · %.0fs · %s → %s" % (n, time.time() - t0, dict(st), OUT))
