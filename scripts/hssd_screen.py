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
a = ap.parse_args()
OUTDOOR = ("outdoor", "porch", "balcony", "yard", "garden", "patio")
BATH = ("bathroom", "toilet")
inst = {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(a.root, "scenes*", "*.scene_instance.json"))}
rows = []
for p in sorted(glob.glob(os.path.join(a.root, "semantics", "scenes", "*.json"))):
    sc = os.path.basename(p).split(".")[0]
    if sc not in inst: continue
    try: regs = json.load(open(p)).get("region_annotations", [])
    except Exception: continue
    names = [re.sub(r"\.\d+$", "", r.get("name", "")).lower() for r in regs]
    rooms = [n for n in names if n and not any(k in n for k in OUTDOOR)]
    nonbath = [n for n in rooms if not any(k in n for k in BATH)]
    rows.append((sc, len(rooms), len(nonbath)))
strict = [r for r in rows if 4 <= r[1] <= 8 and 2 <= r[2] <= 6]
relaxed = [r for r in rows if 3 <= r[1] <= 10 and 2 <= r[2] <= 8]
print("장면 %d · 엄격 %d · 완화 %d" % (len(rows), len(strict), len(relaxed)))
have = [l.strip() for l in open(a.strict) if l.strip()] if os.path.exists(a.strict) else []
print("기존 목록 %d · 그중 엄격 충족 %d · 완화 충족 %d" % (len(have),
      sum(1 for s in have if any(r[0] == s for r in strict)), sum(1 for s in have if any(r[0] == s for r in relaxed))))
out = list(have)
for r in sorted(strict) + sorted(relaxed):
    if r[0] not in out: out.append(r[0])
    if len(out) >= a.n: break
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write("\n".join(out) + "\n")
print("→ %s (%d채)" % (a.out, len(out)))
