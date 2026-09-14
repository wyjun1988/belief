#!/usr/bin/env python3
"""LoRA 학습셋 — "기록 자리 판정"(§166-68: ① 오류 194건 중 175건이 순위·선택 문제) (2026-09-14).

초기맵은 타입마다 자리 후보를 최대 8개 갖는다. 질문은 **"이 빨간 박스 안의 것이 정말 <타입>인가"** 다.
후보의 스캔 뷰 크롭(anchor_registry.json 의 박스)을 한 장 보여주고 yes/no 를 묻는다. 참조 이미지는 없다 —
기록 자체를 만드는 단계라 대조할 기준이 없기 때문이다. 채택 판정(lora_adopt_data.py)과 다른 점이 이것이다.

라벨은 GT 로 공짜: 그 후보의 방(열린공간 병합)이 GT 물체의 방과 같으면 yes, 아니면 no.
  THOR_ROOT 은 인자로. anchor_registry.py 를 INITMAP_FILE=<초기맵> 으로 먼저 돌려둘 것.
  python scripts/lora_place_data.py data/hssd_v2 ~/khcache/lora_place --views 1 --val-houses 25
"""
import argparse, json, glob, os, random, collections
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--views", type=int, default=1)          # 인스턴스당 쓸 뷰 수(최대 3)
ap.add_argument("--val-houses", type=int, default=25); ap.add_argument("--img-w", type=int, default=448)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-houses", type=int, default=0)
ap.add_argument("--pad", type=float, default=0.25)       # 박스 주변 여백(문맥) 비율
a = ap.parse_args(); rng = random.Random(a.seed)
def save(src, dst, box, pad):
    if os.path.exists(dst): return True
    if not os.path.exists(src): return False
    im = Image.open(src).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx0 = max(0, x0 - bw * pad); cy0 = max(0, y0 - bh * pad)
    cx1 = min(W, x1 + bw * pad); cy1 = min(H, y1 + bh * pad)
    if cx1 - cx0 < 24 or cy1 - cy0 < 24: return False
    d = ImageDraw.Draw(im)
    d.rectangle([max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)], outline=(255, 0, 0), width=max(3, W // 150))
    im = im.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    w2, h2 = im.size; s = a.img_w / float(w2)
    im.resize((a.img_w, max(8, int(h2 * s)))).save(dst, quality=90); return True
houses = sorted(glob.glob(os.path.join(a.root, "house_*")))
if a.max_houses: houses = houses[:a.max_houses]
rng.shuffle(houses); val = set(houses[:a.val_houses]); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
fo = {"train": open(os.path.join(a.out, "train.jsonl"), "w"), "val": open(os.path.join(a.out, "val.jsonl"), "w")}
st = collections.Counter()
for hd in houses:
    hn = os.path.basename(hd); split = "val" if hd in val else "train"
    rp = os.path.join(hd, "anchor_registry.json")
    if not os.path.exists(rp): st["등록부 없음"] += 1; continue
    g = json.load(open(hd + "/gt.json")); reg = json.load(open(rp))
    gm = json.load(open(hd + "/room_groups.json"))["groups"] if os.path.exists(hd + "/room_groups.json") else {}
    grp = lambda r: gm.get(r, r) if r else r
    ph = set()
    if os.path.exists(hd + "/phantom_ids.json"):
        _p = json.load(open(hd + "/phantom_ids.json")); _d = _p.get("phantom", _p) if isinstance(_p, dict) else {}
        ph = set(sum([v if isinstance(v, list) else v.get("ids", []) for v in _d.values()], [])) if isinstance(_d, dict) else set()
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if k not in ph)
    gt_room = {}
    for oid, v0 in g["gt0"].items():
        if oid in ph or cnt[v0["type"]] > 1: continue
        gt_room[v0["type"]] = grp(v0["room"])
    for it in reg:
        t = it["type"]
        if t not in gt_room: continue                      # 타입 유일 타겟만 (질의 대상과 같은 조건)
        lab = "yes" if grp(it.get("room")) == gt_room[t] else "no"
        for v in (it.get("views") or [])[:a.views]:
            k = v.get("k"); box = v.get("box")
            if k is None or not box: continue
            f = "%s_%s_v%d.jpg" % (hn, it["id"].replace("#", "_").replace(" ", "_"), k)
            if not save(os.path.join(hd, "map", "%04d.jpg" % int(k)), os.path.join(a.out, "images", f), box, a.pad):
                st["크롭 실패"] += 1; continue
            fo[split].write(json.dumps(dict(house=hn, iid=it["id"], type=t.replace("_", " ").lower(),
                                            cands=[f], ref=None, label=lab, room=it.get("room"),
                                            owl=v.get("owl"), w=it.get("w"))) + "\n")
            st[(split, lab)] += 1
for f in fo.values(): f.close()
print("LORA_PLACE_DATA_DONE", dict(st), "→", a.out)
