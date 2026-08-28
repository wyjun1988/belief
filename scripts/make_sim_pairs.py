#!/usr/bin/env python3
"""시뮬 검증기 쌍 생성 — 해상도 실험용. (GT vis+ctr 기반)

    THOR_ROOT=data/res768 A3_PREFIX=... QC_PREFIX=... OUT=/tmp/pairs768 CROP=256 \
      python scripts/make_sim_pairs.py

양성 = 그 물체가 실제로 보이는 프레임의 ctr 크롭.
음성 = 같은 프레임의 **다른 물체** 크롭에 이 물체 라벨.
exp_vlm_verify3 과 같은 meta.jsonl 스키마 → 로짓 채점 후 로컬 스윕.
"""
import glob, json, os
import numpy as np
from PIL import Image
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/res384")
OUT = os.environ.get("OUT", "/tmp/pairs")
CROP = int(os.environ.get("CROP", "0"))   # 0 = 프레임 폭의 1/3 자동 (기하 통일)
N = int(os.environ.get("N", "400"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
meta = []; k = 0
def words(t): return "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
HDS = sorted(glob.glob(ROOT + "/house_*"))
HB = max(4, -(-N // max(len(HDS), 1)))   # 채당 상한 — 앞쪽 채 편중 방지
for hd in HDS:
    if k >= N: break
    hk = 0
    g = json.load(open(os.path.join(hd, "gt.json")))
    typ = {o: v["type"] for o, v in g["gt0"].items()}
    lv = {int(os.path.basename(p)[:-4]): p
          for p in glob.glob(os.path.join(hd, "live", "*.jpg"))}
    ms = [m for m in g["live"] if m["t"] in lv and len(m.get("ctr") or {}) >= 2]
    rng.shuffle(ms)
    for m in ms:
        if k >= N or hk >= HB: break
        oids = [o for o, c in (m.get("ctr") or {}).items() if c and o in typ]
        if len(oids) < 2: continue
        o1, o2 = rng.choice(oids, 2, replace=False)
        if typ[o1] == typ[o2]: continue
        im = Image.open(lv[m["t"]]).convert("RGB")
        W = im.width; h = (CROP if CROP else W // 3) // 2
        def crop(o):
            cx, cy = m["ctr"][o]
            return im.crop((max(0, cx-h), max(0, cy-h),
                            min(W, cx+h), min(W, cy+h))).resize((336, 336), Image.LANCZOS)
        def dst(o): return round(float((m.get("dist") or {}).get(o, -1)), 1)
        f1 = os.path.join(OUT, "cand_%04d.jpg" % k); crop(o1).save(f1, quality=92)
        meta.append(dict(cand=f1, enroll=f1, label=words(typ[o1]),
                         alt=words(typ[o2]), truth=1, dist=dst(o1))); k += 1; hk += 1
        if k >= N: break
        f2 = os.path.join(OUT, "cand_%04d.jpg" % k); crop(o2).save(f2, quality=92)
        meta.append(dict(cand=f2, enroll=f2, label=words(typ[o1]),
                         alt=words(typ[o2]), truth=0, dist=dst(o2))); k += 1; hk += 1
open(os.path.join(OUT, "meta.jsonl"), "w").write("\n".join(json.dumps(m) for m in meta))
print("쌍 %d → %s (크롭 %s)" % (k, OUT, CROP if CROP else "W/3 자동"))
