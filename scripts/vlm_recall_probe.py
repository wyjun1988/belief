#!/usr/bin/env python3
"""작은 물체 재현율 탐침 — OWL 이 놓친 물체(지도 프레임에서 GT 로 보이는데 투영점이 GT 1 m 안에 없음)를 VLM(Qwen3.5-4B mlx) 전체 프레임 질문으로 잡을 수 있나,
thinking 을 켜면 나아지나. 같은 수의 음성 프레임(그 물체가 안 보이는 프레임)으로 거짓 양성도 잰다.
    THOR_ROOTS="data/hssd90_c4e2 data/hssd40_c3" MAXPOS=60 OUT_JSONL=~/khcache/vlm_recall_probe.jsonl ~/mlx-venv/bin/python scripts/vlm_recall_probe.py"""
import glob, json, os, math, time, random, collections, re
import numpy as np
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4"); ROOTS = os.environ.get("THOR_ROOTS", "data/hssd90_c4e2").split()
MAXPOS = int(os.environ.get("MAXPOS", "60")); OUTJ = os.path.expanduser(os.environ.get("OUT_JSONL", "~/khcache/vlm_recall_probe.jsonl")); THINK_TOK = int(os.environ.get("THINK_TOK", "600"))
random.seed(0)
model, processor = load(MODEL); cfg = model.config
def ask(img, tp, think):
    q = "Is there a %s visible in this image? Answer yes or no." % tp.replace("_", " ")
    if think: q += " Think carefully about every surface and corner first, then give the final answer as 'Answer: yes' or 'Answer: no'."
    prompt = apply_chat_template(processor, cfg, q, num_images=1, enable_thinking=think)
    t0 = time.time(); r = generate(model, processor, prompt, image=[img], max_tokens=THINK_TOK if think else 8, temperature=0.0, verbose=False)
    txt = r.text if hasattr(r, "text") else str(r); tail = re.split(r"</think>", txt)[-1]
    m = re.search(r"answer\s*:\s*(yes|no)", tail, re.I) or re.search(r"\b(yes|no)\b", tail, re.I)
    return (m.group(1).lower() if m else "?"), round(time.time() - t0, 1), len(txt)
pos, neg = [], []
for root in ROOTS:
    for hd in sorted(glob.glob(root + "/house_*")):
        if not os.path.exists(hd + "/initmap_raw.json"): continue
        raw = json.load(open(hd + "/initmap_raw.json")); g = json.load(open(hd + "/gt.json"))
        cnt = collections.Counter(v["type"] for v in g["gt0"].values()); mv = {m["oid"] for m in g["moves"]}
        for oid, v0 in g["gt0"].items():
            if oid in mv or cnt[v0["type"]] > 1 or not v0["room"]: continue
            gs = (v0["pos"][0], v0["pos"][2]); pts = raw.get(v0["type"], [])
            if any(math.hypot(p[0]-gs[0], p[1]-gs[1]) <= 1.0 for p in pts): continue           # OWL 이 (어느 정도) 잡은 물체는 제외
            vis = [k for k, m in enumerate(g["map"]) if oid in (m.get("box") or {})]
            if not vis: continue
            k = vis[len(vis) // 2]; pos.append((hd, oid, v0["type"], k))
            far = [k2 for k2, m in enumerate(g["map"]) if oid not in (m.get("box") or {}) and m.get("room") != v0["room"]]
            if far: neg.append((hd, oid, v0["type"], random.choice(far)))
random.shuffle(pos); pos = pos[:MAXPOS]; neg = neg[:len(pos)]
print("양성(OWL 놓침·GT 가시) %d · 음성 %d · 모델 %s" % (len(pos), len(neg), MODEL), flush=True)
out = open(OUTJ, "w"); stat = collections.defaultdict(list)
for lab, items in (("pos", pos), ("neg", neg)):
    for hd, oid, tp, k in items:
        img = "%s/map/%04d.jpg" % (hd, k); rec = dict(house=os.path.basename(hd), oid=oid, type=tp, k=k, label=lab)
        for think in (False, True):
            a, dt, n = ask(img, tp, think); rec["think" if think else "plain"] = a; rec["t_" + ("think" if think else "plain")] = dt; rec["n_" + ("think" if think else "plain")] = n
            stat[(lab, think)].append((a == "yes", dt))
        out.write(json.dumps(rec) + "\n"); out.flush()
        print("  %s %-13s %s plain=%s think=%s (%.0fs)" % (lab, tp[:13], os.path.basename(hd)[-4:], rec["plain"], rec["think"], rec["t_think"]), flush=True)
for lab in ("pos", "neg"):
    for think in (False, True):
        v = stat[(lab, think)]
        if v: print("%s %-8s: yes 비율 %.2f (n=%d) · 평균 %.1fs" % (lab, "thinking" if think else "plain", np.mean([x[0] for x in v]), len(v), np.mean([x[1] for x in v])), flush=True)
print("VLM_RECALL_PROBE_DONE →", OUTJ)
