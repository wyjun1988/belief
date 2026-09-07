#!/usr/bin/env python3
"""카메라방 판정 탐침 — VLM(Qwen3.5-4B mlx)에 방 목록(+초기맵의 방별 물체 유형, 무GT)과 현재 프레임을 주고 방을 고르게 한다. 임베딩 카메라방(ROOM_JSONL)과 같은 프레임에서 GT 대조.
    THOR_ROOT=data/hssd90_c4e2 ROOM_JSONL=$B/scores/room_embed_clip.jsonl INITMAP_FILE=initmap_owl_rc.json N=120 ~/mlx-venv/bin/python scripts/vlm_room_probe.py"""
import glob, json, os, random, re, time, collections, numpy as np
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
MODEL = os.environ.get("MODEL", "RepublicOfKorokke/Qwen3.5-4B-mlx-vlm-mxfp4"); ROOT = os.environ.get("THOR_ROOT", "data/hssd90_c4e2"); RJ = os.environ["ROOM_JSONL"]; IMF = os.environ.get("INITMAP_FILE", "initmap_owl_rc.json")
N = int(os.environ.get("N", "120")); MAXH = int(os.environ.get("MAXH", "10")); OUTJ = os.path.expanduser(os.environ.get("OUT_JSONL", "~/khcache/vlm_room_probe.jsonl")); random.seed(0)
model, processor = load(MODEL); cfg = model.config
emb = collections.defaultdict(dict)
for l in open(RJ):
    r = json.loads(l); emb[r["house"]][r["t"]] = r["room"]
rows = []; houses = sorted(glob.glob(ROOT + "/house_*"))[:MAXH]
for hd in houses:
    hn = os.path.basename(hd); g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gf = hd + "/room_groups.json"; gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}; grp = lambda r: gm.get(r, r) if r else r
    im = json.load(open(hd + "/" + IMF)); objs = collections.defaultdict(collections.Counter)
    for it in im: objs[grp(it.get("room"))][it["type"]] += it.get("w", 1.0)
    rooms = sorted({grp(r) for r in g["room_types"]}); opts = "\n".join("- %s: %s" % (r, ", ".join(t for t, _ in objs[r].most_common(6)) or "(no objects recorded)") for r in rooms)
    ts = sorted(live); random.shuffle(ts); ts = ts[: max(1, N // len(houses))]
    BATCH = int(os.environ.get("BATCH", "1"))     # 한 프롬프트에 프레임 여러 장(사용자 제안 2026-09-07): "각각 어느 방인가"
    for b0 in range(0, len(ts), BATCH):
        tb = ts[b0:b0 + BATCH]; imgs = ["%s/live/%06d.jpg" % (hd, t) for t in tb]; t0 = time.time()
        if BATCH == 1:
            q = "You are inside a house. Rooms in this house and the objects recorded in each room:\n%s\nWhich room is this photo taken in? Answer with exactly one room name from the list." % opts
        else:
            q = "You are inside a house. Rooms in this house and the objects recorded in each room:\n%s\nYou are given %d photos taken in this house. For each photo, in order, answer which room it was taken in. Reply with %d lines, each 'photo k: <room name from the list>'." % (opts, len(tb), len(tb))
        prompt = apply_chat_template(processor, cfg, q, num_images=len(imgs), enable_thinking=False)
        out = generate(model, processor, prompt, image=imgs, max_tokens=24 * len(tb), temperature=0.0, verbose=False); txt = (out.text if hasattr(out, "text") else str(out)).strip().lower()
        lines = [l for l in txt.split("\n") if l.strip()] if BATCH > 1 else [txt]
        for k, t in enumerate(tb):
            seg = lines[k] if k < len(lines) else ""
            pick = next((r for r in sorted(rooms, key=len, reverse=True) if r.lower() in seg), None)
            rows.append(dict(house=hn, t=t, gt=grp(live[t].get("room")), emb=grp(emb[hn].get(t)), vlm=pick, raw=seg[:60], dt=round((time.time() - t0) / len(tb), 1)))
    print("  %s: VLM %.2f · 임베딩 %.2f (n=%d)" % (hn, np.mean([r["vlm"] == r["gt"] for r in rows if r["house"] == hn]), np.mean([r["emb"] == r["gt"] for r in rows if r["house"] == hn]), sum(r["house"] == hn for r in rows)), flush=True)
with open(OUTJ, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
v = np.mean([r["vlm"] == r["gt"] for r in rows]); e = np.mean([r["emb"] == r["gt"] for r in rows]); both = np.mean([(r["vlm"] == r["gt"]) or (r["emb"] == r["gt"]) for r in rows])
print("전체 n=%d: VLM(방 목록+물체) %.2f · 임베딩 Viterbi %.2f · 둘 중 하나 맞음 %.2f · VLM 무응답 %.2f · 장당 %.1fs" % (len(rows), v, e, both, np.mean([r["vlm"] is None for r in rows]), np.mean([r["dt"] for r in rows])))
print("VLM_ROOM_PROBE_DONE →", OUTJ)
