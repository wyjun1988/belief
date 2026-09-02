#!/usr/bin/env python3
"""HSSD 방 구조 심사 — 벤치 장면 목록 생성 (SIM_SCREENING §HSSD 재현).

    python scripts/hssd_screen.py --root ~/hssd-hab --strict docs/bench/hssd20_scenes.txt \\
        --n 40 --out docs/bench/hssd40_scenes.txt

region 주석(semantics/scenes/*.json)으로 방 수를 세고, scene_instance 가 있는 장면만.
엄격: 방 4~8 · 비화장실 2~6 / 완화: 방 3~10 · 비화장실 2~8. 기존 목록을 앞에 두고
나머지를 장면 id 순으로 채운다(결정론).
"""
import argparse, glob, json, os, re
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/hssd-hab"))
ap.add_argument("--strict", default="docs/bench/hssd20_scenes.txt")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="docs/bench/hssd40_scenes.txt")
ap.add_argument("--min-cands", type=int, default=0, help="타입 단일·이동가능·비부착 물체 수 최소 (시뮬 없이 scene_instance+csv 로 계산)")
a = ap.parse_args()
OUTDOOR = ("outdoor", "porch", "balcony", "yard", "garden", "patio")
BATH = ("bathroom", "toilet")
inst = {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(a.root, "scenes*", "*.scene_instance.json"))}
MOVABLE = ("book", "cushion", "plate", "bowl", "cup", "mug", "lamp", "clock", "vase",
           "basket", "kitchenutensil", "sponge", "toy", "phone", "laptop", "can",
           "box", "picture frame", "plant", "shoe", "bottle", "handbag", "drinkware",
           "toiletry", "candle", "clothing", "tray", "kettle", "remote", "bag", "hat")
MOUNTED = ("ceiling", "wall lamp", "wall clock", "curtain", "chandelier", "sconce")
import csv
from collections import Counter
HASH = {}
for mf in glob.glob(os.path.join(a.root, "metadata", "fpmodels*.csv")):
    for row in csv.DictReader(open(mf, newline="")):
        c = (row.get("main_category") or "").strip()
        if row.get("id") and c: HASH[row["id"]] = c.replace("_", " ").lower()
def n_cands(sc):
    ps = glob.glob(os.path.join(a.root, "scenes*", sc + ".scene_instance.json"))
    if not ps: return 0
    labs = [HASH.get(oi["template_name"].split("/")[-1]) for oi in json.load(open(ps[0])).get("object_instances", [])]
    cnt = Counter(l for l in labs if l)
    return sum(1 for l, c in cnt.items() if c == 1 and any(m in l for m in MOVABLE) and not any(m in l for m in MOUNTED))
rows = []
for p in sorted(glob.glob(os.path.join(a.root, "semantics", "scenes", "*.json"))):
    sc = os.path.basename(p).split(".")[0]
    if sc not in inst: continue
    try: regs = json.load(open(p)).get("region_annotations", [])
    except Exception: continue
    names = [re.sub(r"\.\d+$", "", r.get("name", "")).lower() for r in regs]
    rooms = [n for n in names if n and not any(k in n for k in OUTDOOR)]
    nonbath = [n for n in rooms if not any(k in n for k in BATH)]
    rows.append((sc, len(rooms), len(nonbath), n_cands(sc) if a.min_cands else 0))
rows = [r for r in rows if r[3] >= a.min_cands]
strict = [r for r in rows if 4 <= r[1] <= 8 and 2 <= r[2] <= 6]
relaxed = [r for r in rows if 3 <= r[1] <= 10 and 2 <= r[2] <= 8]
if a.min_cands: print("후보≥%d 장면 %d · 후보 수 분포: %s" % (a.min_cands, len(rows), sorted(Counter(r[3] for r in rows).items())))
print("장면 %d · 엄격 %d · 완화 %d" % (len(rows), len(strict), len(relaxed)))
have = [l.strip() for l in open(a.strict) if l.strip()] if os.path.exists(a.strict) and not a.min_cands else []
print("기존 목록 %d · 그중 엄격 충족 %d · 완화 충족 %d" % (len(have),
      sum(1 for s in have if any(r[0] == s for r in strict)), sum(1 for s in have if any(r[0] == s for r in relaxed))))
out = list(have)
for r in sorted(strict, key=lambda r: (-r[3], r[0])) + sorted(relaxed, key=lambda r: (-r[3], r[0])):   # 후보 많은 순
    if r[0] not in out: out.append(r[0])
    if len(out) >= a.n: break
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(out) + "\n")
print("→ %s (%d채)" % (a.out, len(out)))
