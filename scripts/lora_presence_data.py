#!/usr/bin/env python3
"""LoRA 학습셋 생성 — "자리 존재 판정"(§166-48 B 판 형식: 참조 장면 + 빨간 박스, 나중 자리 프레임 ≤K, 서문 없음, 답 JSON) (2026-09-11).
시뮬 GT 로 라벨이 공짜: 타입 유일 타겟마다 참조 = 물체가 가장 가까이 보이는 스캔 프레임(GT 중심 → 박스), 후보 = GT 포즈가 자리를 향한(같은 방·≤4 m·≤35°) 라이브 프레임 K장.
라벨: 이동 전이면 "yes"(물체가 GT 로 K장 중 ≥1 보이면) / 이동 후면 "no" / 안 옮겼는데 K장 모두 GT 로 안 보이면 "unsure"(가려짐 — 4B 의 "없다" 편향을 학습시키지 않기 위한 클래스).
같은 타겟에서 시간 구간을 달리해 여러 표본(이동 전·후 각각). 집 단위 분할(train/val) 은 --val-houses.
  python scripts/lora_presence_data.py data/hssd_v2 out/lora_presence --k 4 --per-target 3 --val-houses 25   # → out/.../{train,val}.jsonl + images/ (HF 학습 스크립트 lora_presence_train.py 입력)"""
import os, sys, json, glob, math, random, shutil, argparse, collections
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--per-target", type=int, default=3)
ap.add_argument("--val-houses", type=int, default=25); ap.add_argument("--img-w", type=int, default=448); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-houses", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed)
def facing(ax, az, yaw, spot, dmax=4.0, amax=35.0):
    dx, dz = spot[0] - ax, spot[1] - az; d = math.hypot(dx, dz); return d <= dmax and abs((math.degrees(math.atan2(dx, dz)) - yaw + 180) % 360 - 180) <= amax
def save(src, dst, ctr=None):
    if os.path.exists(dst): return
    im = Image.open(src).convert("RGB"); w, h = im.size
    if ctr: r = int(0.09 * w); ImageDraw.Draw(im).rectangle([max(0, ctr[0]-r), max(0, ctr[1]-r), min(w-1, ctr[0]+r), min(h-1, ctr[1]+r)], outline=(255, 0, 0), width=max(3, w // 200))
    s = a.img_w / float(w); im.resize((a.img_w, max(8, int(h * s)))).save(dst, quality=90)
houses = sorted(glob.glob(os.path.join(a.root, "house_*")))
if a.max_houses: houses = houses[:a.max_houses]
rng.shuffle(houses); val = set(houses[:a.val_houses]); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
fo = {"train": open(os.path.join(a.out, "train.jsonl"), "w"), "val": open(os.path.join(a.out, "val.jsonl"), "w")}; stat = collections.Counter()
for hd in houses:
    hn = os.path.basename(hd); split = "val" if hd in val else "train"; g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    gm = json.load(open(hd + "/room_groups.json"))["groups"] if os.path.exists(hd + "/room_groups.json") else {}; grp = lambda r: gm.get(r, r) if r else r
    ph = set()
    if os.path.exists(hd + "/phantom_ids.json"):
        _p = json.load(open(hd + "/phantom_ids.json")); _d = _p.get("phantom", _p) if isinstance(_p, dict) else {}
        ph = set(sum([v if isinstance(v, list) else v.get("ids", []) for v in _d.values()], [])) if isinstance(_d, dict) else set()
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if k not in ph); mv = {m["oid"]: m for m in g["moves"]}
    for oid, v0 in g["gt0"].items():
        if oid in ph or cnt[v0["type"]] > 1: continue
        typ = v0["type"].replace("_", " ").lower(); spot = [v0["pos"][0], v0["pos"][2]]; room = grp(v0["room"])
        cands = [(m["dist"][oid], k) for k, m in enumerate(g["map"]) if oid in (m.get("ctr") or {}) and oid in (m.get("dist") or {})]
        if not cands: stat["no_ref"] += 1; continue
        d0, k0 = min(cands); ctr0 = g["map"][k0]["ctr"][oid]; ref = "%s_map%04d_%s.jpg" % (hn, k0, oid.replace("|", "_").replace(" ", "_")); save(os.path.join(hd, "map", "%04d.jpg" % k0), os.path.join(a.out, "images", ref), ctr0)
        fac = sorted(t for t, m in live.items() if m.get("apos") and m.get("yaw") is not None and grp(m["room"]) == room and facing(m["apos"][0], m["apos"][1], m["yaw"], spot))
        t_mv = mv[oid]["t"] if oid in mv else None
        segs = []                                          # (프레임 후보, 라벨 규칙)
        pre = [t for t in fac if t_mv is None or t <= t_mv]; post = [t for t in fac if t_mv is not None and t > t_mv]
        if len(pre) >= 2: segs.append((pre, "pre"))
        if len(post) >= 2: segs.append((post, "post"))
        for frames, kind in segs:
            for _ in range(a.per_target if kind == "pre" else a.per_target + 1):   # 이동 후(부재) 표본이 적으니 조금 더
                pick = sorted(rng.sample(frames, min(a.k, len(frames))))
                vis = sum(1 for t in pick if oid in (live[t].get("vis") or []))
                label = "no" if kind == "post" else ("yes" if vis >= 1 else "unsure")
                files = []
                for t in pick:
                    f = "%s_live%06d.jpg" % (hn, t); save(os.path.join(hd, "live", "%06d.jpg" % t), os.path.join(a.out, "images", f)); files.append(f)
                seen = [i + 2 for i, t in enumerate(pick) if oid in (live[t].get("vis") or [])]
                fo[split].write(json.dumps(dict(house=hn, oid=oid, type=typ, ref=ref, cands=files, label=label, seen_in=seen, kind=kind, t_move=t_mv)) + "\n"); stat[(split, label)] += 1
for f in fo.values(): f.close()
print("LORA_DATA_DONE", dict(stat), "→", a.out)
