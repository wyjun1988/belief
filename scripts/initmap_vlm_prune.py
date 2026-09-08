#!/usr/bin/env python3
"""초기맵 인스턴스 VLM 진위 검사(RTX 9-11, HF Qwen): anchor_crops.py 가 저장한 인스턴스 크롭마다 "정말 그 타입인가"(A/B) 를 묻고
통과 인스턴스만 남긴 초기맵 파일을 쓴다. 검출 임계를 내려(TH 0.06) 늘어난 후보의 오검출을 걸러 ② 손실을 되돌리는 것이 목적.
  THOR_ROOT=... INITMAP_FILE=initmap_owl_rc06.json [MODEL=Qwen/Qwen3.5-9B] [KEEP_UNCROPPED=1] python scripts/initmap_vlm_prune.py
산출: <house>/<INITMAP_FILE 이름>_v.json (+ 판정 기록 <이름>_vlm.json). 크롭이 없는 인스턴스(raw 점 없음·박스 없음)는 KEEP_UNCROPPED=1(기본)이면 남긴다."""
import os, json, glob, time, collections, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json"); HOUSES = os.environ.get("HOUSES", "").split()
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B"); KEEP = os.environ.get("KEEP_UNCROPPED", "1") == "1"; Q1_TH = float(os.environ.get("Q1_TH", "0")); REDO = os.environ.get("REDO", "0") == "1"
pr = AutoProcessor.from_pretrained(MODEL); md = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval(); tok = pr.tokenizer
IDA, IDB = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
def ab(img, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    try: text = pr.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError: text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=[img], text=text, return_tensors="pt").to(md.device)
    with torch.no_grad(): lg = md(**inp).logits[0, -1]
    return float(lg[IDA] - lg[IDB])
hds = [os.path.join(ROOT, h) for h in HOUSES] if HOUSES else sorted(glob.glob(os.path.join(ROOT, "house_*")))
stem = IMF[:-5] if IMF.endswith(".json") else IMF
for hd in hds:
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); outp = os.path.join(hdr, stem + "_v.json"); cp_ = os.path.join(hdr, "anchor_cands.json")
    if os.path.exists(outp) and not REDO: print("%s 있음 — 건너뜀" % hn, flush=True); continue
    if not (os.path.exists(os.path.join(hdr, IMF)) and os.path.exists(cp_)): print("%s 초기맵/크롭 없음(anchor_crops.py 먼저)" % hn, flush=True); continue
    im_ = json.load(open(os.path.join(hdr, IMF))); cands = json.load(open(cp_)); T0 = time.time()
    key = lambda it: (it["type"], tuple(round(v, 2) for v in (it.get("pos") or [])))
    verdict = {}
    for c in cands:
        q1 = ab(Image.open(os.path.join(hdr, "anchor_crops", "%03d.jpg" % c["i"])).convert("RGB"), "Is there a %s in this image? (A) yes (B) no. Answer only A or B." % c["type"])
        verdict[key(c)] = q1
    kept = []; n_drop = n_unc = 0; log = []
    for it in im_:
        q1 = verdict.get(key(it))
        if q1 is None: n_unc += 1; (kept.append(it) if KEEP else None); log.append(dict(type=it["type"], room=it.get("room"), w=it["w"], q1=None, keep=KEEP)); continue
        ok = q1 > Q1_TH; n_drop += (not ok)
        if ok: kept.append(it)
        log.append(dict(type=it["type"], room=it.get("room"), w=it["w"], q1=round(q1, 2), keep=ok))
    json.dump(kept, open(outp, "w"), ensure_ascii=False); json.dump(log, open(os.path.join(hdr, stem + "_vlm.json"), "w"), ensure_ascii=False)
    print("%s: 인스턴스 %d → %d (탈락 %d · 크롭 없음 %d) %.0fs" % (hn, len(im_), len(kept), n_drop, n_unc, time.time() - T0), flush=True)
print("INITMAP_VLM_PRUNE_DONE")
