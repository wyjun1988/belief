#!/usr/bin/env python3
"""LoRA — 자리 존재 판정 (RTX, transformers + peft) (2026-09-11). lora_presence_data.py 산출(train/val.jsonl + images/) 로 Qwen3.5-VL 을 학습한다.
형식은 §166-48 B 판과 동일(참조 박스 + 나중 프레임 ≤4, 서문 없음, 답 JSON) — 학습 뒤 oracle_vlm_probe.py(BACKEND=hf, ADAPTER=)로 같은 잣대로 잰다.
  pip install peft   # transformers 는 이미 있음
  python scripts/lora_presence_train.py --data ~/khcache/lora_presence --model Qwen/Qwen3.5-4B --out ~/khcache/lora_presence_4b [--epochs 2 --lr 1e-4 --r 16 --max-steps 0 --eval-every 200]
  --eval-only --adapter <dir> 로 val 만 채점. 드라이런: --max-steps 20."""
import os, json, argparse, random, time, re, collections
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--eval-every", type=int, default=200); ap.add_argument("--grad-accum", type=int, default=8); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--adapter", default="")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val-max", type=int, default=300); a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
processor = AutoProcessor.from_pretrained(a.model); model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
from peft import LoraConfig, get_peft_model, PeftModel
if a.adapter: model = PeftModel.from_pretrained(model, a.adapter, is_trainable=not a.eval_only)
elif not a.eval_only:
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg); model.print_trainable_parameters()
def load(split): return [json.loads(l) for l in open(os.path.join(a.data, split + ".jsonl"))]
def article(t): return ("an " if t[:1] in "aeiou" else "a ") + t
def prompt_of(r):
    n = len(r["cands"]) + 1
    return ("Image 1 shows %s at its recorded place in a house (inside the red box). Images 2-%d are later views of that same place, in time order. "
            "Look carefully: is the %s still at its recorded place in the later images? Answer with JSON only: "
            "{\"still_there\": \"yes\"|\"no\"|\"unsure\", \"seen_in\": [image numbers (2-%d) where the %s is visible], \"confidence\": 0-100}" % (article(r["type"]), n, r["type"], n, r["type"]))
def answer_of(r): return json.dumps(dict(still_there=r["label"], seen_in=r.get("seen_in", []), confidence=(90 if r["label"] in ("yes", "no") else 50)))
def images_of(r): return [Image.open(os.path.join(a.data, "images", f)).convert("RGB") for f in [r["ref"]] + r["cands"]]
def chat(r, with_answer):
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(len(r["cands"]) + 1)] + [{"type": "text", "text": prompt_of(r)}]}]
    if with_answer: msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer_of(r)}]})
    try: t = processor.apply_chat_template(msgs, add_generation_prompt=not with_answer, tokenize=False, enable_thinking=False)
    except TypeError: t = processor.apply_chat_template(msgs, add_generation_prompt=not with_answer, tokenize=False)
    if not with_answer and "<think>" not in t[-60:]: t = t + ("" if t.endswith("\n") else "\n") + "<think>\n\n</think>\n\n"
    return t
def encode(r):
    """전체(프롬프트+답) 토큰과 프롬프트 길이 → labels 는 답 부분만."""
    ims = images_of(r); full = processor(text=[chat(r, True)], images=ims, return_tensors="pt"); pl = processor(text=[chat(r, False)], images=ims, return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone(); labels[:, :pl] = -100; full["labels"] = labels; return full
def parse(txt):
    m = re.search(r"\{.*?\}", txt, re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}
@torch.no_grad()
def evaluate(rows, tag):
    model.eval(); st = collections.Counter(); pf = '{"still_there": "'
    for r in rows[:a.val_max]:
        ims = images_of(r); text = chat(r, False) + pf; inp = processor(text=[text], images=ims, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=80, do_sample=False); txt = pf + processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ans = str(parse(txt).get("still_there", "")).lower(); st[(r["label"], ans)] += 1
    y = sum(v for (l, ans), v in st.items() if l == "yes"); yy = st[("yes", "yes")]; n = sum(v for (l, ans), v in st.items() if l == "no"); nn = st[("no", "no")]; u = sum(v for (l, ans), v in st.items() if l == "unsure"); uu = st[("unsure", "unsure")] + st[("unsure", "yes")]
    print("EVAL[%s] yes→yes %d/%d (%.2f) · no→no %d/%d (%.2f) · unsure→unsure|yes %d/%d (%.2f) · %s" % (tag, yy, y, yy / max(1, y), nn, n, nn / max(1, n), uu, u, uu / max(1, u), dict(st)), flush=True); model.train(); return yy / max(1, y), nn / max(1, n)
train, val = load("train"), load("val"); print("train %d · val %d" % (len(train), len(val)), flush=True)
if a.eval_only: evaluate(val, "val"); raise SystemExit
model.train(); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0); step = 0; t0 = time.time(); best = -1
for ep in range(a.epochs):
    random.shuffle(train); acc = 0.0
    for i, r in enumerate(train):
        try: enc = encode(r).to(model.device)
        except Exception as e: print("skip", r["house"], r["oid"], e); continue
        loss = model(**enc).loss / a.grad_accum; loss.backward(); acc += float(loss)
        if (i + 1) % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); step += 1
            if step % 10 == 0: print("ep %d step %d loss %.4f · %.0fs" % (ep, step, acc, time.time() - t0), flush=True)
            acc = 0.0
            if a.eval_every and step % a.eval_every == 0:
                y, n = evaluate(val, "step%d" % step)
                if y + n > best: best = y + n; model.save_pretrained(a.out); print("  saved", a.out, flush=True)
            if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
y, n = evaluate(val, "final")
if y + n >= best: model.save_pretrained(a.out); print("  saved", a.out, flush=True)
print("LORA_TRAIN_DONE steps %d · %.0fs · best yes %.2f no %.2f" % (step, time.time() - t0, y, n))
