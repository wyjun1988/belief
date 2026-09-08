#!/usr/bin/env python3
"""고정 앵커 명부 2단계(mlx-venv): anchor_crops.py 가 저장한 크롭마다 Qwen3.5-4B 에게 (Q1) 정말 그 타입인가, (Q2) 잘 안 옮기는
고정 가구·가전인가 / 자주 옮기는 물건인가 를 묻고 둘 다 통과한 인스턴스만 앵커로 적는다 (사용자 결정 2026-09-08 17:25, §166-28).
GT 는 안 쓴다. 산출: <house>/anchors_vlm.json = [{type, pos, room, w, k, box, owl, q1, q2, anchor}] — eval_online ANCH_SRC=vlm.
  THOR_ROOT=... HOUSES="house_0014" ~/mlx-venv/bin/python scripts/anchor_vlm_mlx.py"""
import os, json, glob, time, collections
from PIL import Image
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); OUTN = os.environ.get("ANCH_OUT", "anchors_vlm.json"); HOUSES = os.environ.get("HOUSES", "").split(); REDO = os.environ.get("REDO", "0") == "1"
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4")
Q2_TH = float(os.environ.get("Q2_TH", "0.5")); EXCL = set(t.strip() for t in os.environ.get("ANCH_EXCL", "Wall,Doorway,Doorframe,Floor,Window,Painting,Faucet").split(","))   # 방 경계에 걸치는 구조물은 앵커에서 제외
model, processor = load(MODEL); cfg = model.config; tok = processor.tokenizer
IDS = {t: tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B")}
def ab(img, q):
    prompt = apply_chat_template(processor, cfg, q, num_images=1)
    inp = prepare_inputs(processor, images=[img], prompts=[prompt], image_token_index=getattr(cfg, "image_token_index", None))
    out = model(inp["input_ids"], inp["pixel_values"], mask=inp.get("attention_mask"), **{k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask")})
    lg = out.logits[0, -1]; mx.eval(lg); return float(lg[IDS["A"]] - lg[IDS["B"]])
hds = [os.path.join(ROOT, h) for h in HOUSES] if HOUSES else sorted(glob.glob(os.path.join(ROOT, "house_*")))
for hd in hds:
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); outp = os.path.join(hdr, OUTN); cp_ = os.path.join(hdr, "anchor_cands.json")
    if os.path.exists(outp) and not REDO: print("%s 있음 — 건너뜀" % hn, flush=True); continue
    if not os.path.exists(cp_): print("%s 후보 없음(1단계 먼저)" % hn, flush=True); continue
    cands = json.load(open(cp_)); T0 = time.time(); out = []; n_a = 0
    for c in cands:
        cp = Image.open(os.path.join(hdr, "anchor_crops", "%03d.jpg" % c["i"])).convert("RGB")
        q1 = ab(cp, "Is there a %s in this image? (A) yes (B) no. Answer only A or B." % c["type"])
        # 문구 보정(2026-09-08 house_0014 12크롭): P1 "rarely move" 는 소파·테이블을 이동으로 봤다(-1.0). P2 "months in the same place vs carried around"
        # 는 sofa 2.0·table 1.9·ceiling lamp 1.0 / vase -0.9·picture frame -0.9·table lamp -0.4 로 갈린다. 문턱 0.5 (wall lamp 0.6 경계).
        q2 = ab(cp, "In a home, is this %s a large item that stays in the same place for months (A), or a small item that people carry around and put in different places (B)? Answer only A or B." % c["type"]) if q1 > 0 else None
        anchor = bool(q1 > 0 and q2 is not None and q2 > Q2_TH and c["type"] not in EXCL); n_a += anchor
        out.append(dict(c, q1=round(q1, 2), q2=(round(q2, 2) if q2 is not None else None), anchor=anchor))
    json.dump(out, open(outp, "w"), ensure_ascii=False)
    tc = collections.Counter(o["type"] for o in out if o["anchor"])
    print("%s: 후보 %d → 앵커 %d (%.0fs) · 앵커 타입 %s" % (hn, len(out), n_a, time.time() - T0, tc.most_common(8)), flush=True)
print("ANCHOR_VLM_DONE")
