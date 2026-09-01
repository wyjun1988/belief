#!/usr/bin/env python3
"""HSSD 어휘용 시나리오 사전확률 — LLM 이 만든다. (thor_move_llm 의 HSSD 판)

    ~/mlx-venv/bin/python scripts/hssd_move_llm.py --vocab /tmp/hssd_vocab.json \\
      --out data/hssd_move.json

THOR 에서 Qwen 3.5 9B 로 만들었던 3종(방 체류·물체 이동성향·이동 목적지)을
HSSD 어휘(main_category 62종 · region label)로 다시 만든다. 이게 없으면
배회·이동이 **전부 균등 난수**가 되어 (a) 재방문 패턴이 비현실적이고
(b) 목적지가 균등이라 belief 가 원리적으로 못 맞힌다 (§67 과 같은 실패).

산출 스키마 = thor_move.json 과 동일:
  {"dwell": {방유형: 상대체류}, "mobility": {물체: 0~1}, "dest": {물체: {방유형: 확률}}}
"""
import argparse, json, os, re
from mlx_lm import load, generate

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", default="/tmp/hssd_vocab.json")
ap.add_argument("--out", default="data/hssd_move.json")
ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
args = ap.parse_args()

V = json.load(open(args.vocab))
OBJ = V["objects"]
# 방 라벨 정규화: bedroom.001 → bedroom, toilet → bathroom 계열 유지
ROOMS = sorted({re.sub(r"\.\d+$", "", r) for r in V["rooms"]})
print("물체 %d · 방유형 %d" % (len(OBJ), len(ROOMS)), flush=True)

model, tok = load(args.model)

def ask(prompt, maxtok=2200):
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True)
    r = generate(model, tok, prompt=text, max_tokens=maxtok, verbose=False)
    m = re.search(r"\{.*\}", r, re.S)
    if not m: raise SystemExit("JSON 못 찾음:\n" + r[:400])
    return json.loads(m.group(0))

# ① 방 체류 (상대 시간) — 실제 가정에서 사람이 머무는 시간 비율
dwell = ask(
    "You model where a person spends time at home during a normal day.\n"
    "Rooms: %s\n"
    "Output ONLY a JSON object mapping each room to a relative dwell weight "
    "(0.02~1.0, living room highest, closets/garage lowest). No prose."
    % ", ".join(ROOMS))
print("dwell OK (%d)" % len(dwell), flush=True)

# ② 물체별 이동성향 — 하루 중 사람이 옮길 확률
mobility = ask(
    "You model which household objects get moved by people during a day.\n"
    "Objects: %s\n"
    "Output ONLY a JSON object mapping each object to a mobility score 0.0~1.0 "
    "(phone/book/drinkware high; bed/bathtub/counter ~0.0). No prose."
    % ", ".join(OBJ[:80]), maxtok=2600)
print("mobility OK (%d)" % len(mobility), flush=True)

# ③ 이동 목적지 — **배치와 다르다**: 머그컵은 부엌에 놓이지만 거실에서 발견된다
dest = {}
CH = 20
for i in range(0, min(len(OBJ), 60), CH):
    part = OBJ[i:i+CH]
    d = ask(
        "When a person carries a household object to another room and leaves it "
        "there, where does it end up? This is DIFFERENT from where it is stored.\n"
        "Objects: %s\nRooms: %s\n"
        "Output ONLY a JSON object: {object: {room: probability}} with 2-4 rooms "
        "each, probabilities summing to ~1. No prose."
        % (", ".join(part), ", ".join(ROOMS)), maxtok=2600)
    dest.update(d)
    print("  dest %d/%d" % (len(dest), min(len(OBJ), 60)), flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(dict(dwell=dwell, mobility=mobility, dest=dest), open(args.out, "w"),
          ensure_ascii=False, indent=1)
print("→", args.out)
