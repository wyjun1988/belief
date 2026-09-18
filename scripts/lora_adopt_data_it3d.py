#!/usr/bin/env python3
"""채택(재식별) 학습셋 — **실사** IT3DEgo 에서 (2026-09-18).

왜 실사인가: 시뮬 라벨은 공짜지만 실제 조명·모션블러·가림을 재현하지 않는다.
왜 IT3DEgo 인가: 한 장면 안에 `bowl_1`/`bowl_2` 처럼 **같은 종류 다른 개체**가 있다(50영상 중 26개).
  HSSD 는 질의 종류가 집마다 유일하도록 만들어져 이런 음성을 못 만든다. 교차 집으로 흉내냈더니
  "배경이 다르면 아니오" 라는 반대 규칙을 가르쳐 이동 후 AUC 가 0.780 → 0.707 로 떨어졌다(§166-86).
  **같은 장면·같은 종류·다른 개체**가 우리가 원하던 바로 그 음성이다.

표본 구성 (한 영상 = 한 장면):
  양성      같은 개체의 두 시점. **위치 구간이 다른 쌍을 우선**(= 물체가 옮겨진 뒤 = ② 그 자체).
  어려운음성 같은 종류 다른 개체 (bowl_1 ↔ bowl_2). 종류만 봐서는 못 푼다.
  쉬운음성   다른 종류 개체. 기존 시뮬 음성과 같은 난이도(균형용).

  python scripts/lora_adopt_data_it3d.py data/it3dego ~/khcache/lora_adopt_real --val-videos 10
→ <out>/{train,val}.jsonl + images/   (시뮬 셋과 같은 스키마: house/oid/type/ref/cands/label)
"""
import argparse, collections, io, json, os, random, re
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="data/it3dego")
ap.add_argument("out")
ap.add_argument("--val-videos", type=int, default=10, help="검증으로 뗄 영상 수(장면 단위 분리)")
ap.add_argument("--per-obj", type=int, default=6, help="개체당 양성 표본 수")
ap.add_argument("--pad", type=float, default=0.25, help="박스 둘레 여유")
ap.add_argument("--img-w", type=int, default=448)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
OUT = os.path.expanduser(a.out); os.makedirs(OUT + "/images", exist_ok=True)

def base_label(s): return re.sub(r"_\d+$", "", s.strip()).replace("_", " ").strip()

class PV:
    """pv/<video>.bin + .index.json 에서 타임스탬프로 프레임을 꺼낸다."""
    def __init__(self, root, vn):
        self.f = open(os.path.join(root, "pv", vn + ".bin"), "rb")
        idx = json.load(open(os.path.join(root, "pv", vn + ".index.json")))
        self.by_ts = {}
        for e in idx:
            m = re.search(r"/(\d+)\.png$", e["name"])
            if m: self.by_ts[int(m.group(1))] = (e["off"], e["size"])
        self.ts = np.array(sorted(self.by_ts), dtype=np.int64)
    def nearest(self, t, tol=2_000_000):            # 나노초 — 2ms 이내
        if not len(self.ts): return None
        i = int(np.argmin(np.abs(self.ts - t)))
        return int(self.ts[i]) if abs(int(self.ts[i]) - t) <= tol else None
    def img(self, t):
        off, size = self.by_ts[t]; self.f.seek(off)
        return Image.open(io.BytesIO(self.f.read(size))).convert("RGB")

def crop(pv, t, box, dst):
    """box = (x, y, w, h). 여유를 두고 잘라 저장. 못 쓰면 False."""
    if os.path.exists(dst): return True
    im = pv.img(t); W, H = im.size
    x, y, w, h = box
    px, py = w * a.pad, h * a.pad
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
    if x1 - x0 < 16 or y1 - y0 < 16: return False
    c = im.crop((x0, y0, x1, y1))
    if c.width > a.img_w: c = c.resize((a.img_w, max(1, int(c.height * a.img_w / c.width))))
    c.save(dst, quality=90); return True

AD = os.path.join(a.root, "ann", "annotations")
vids = sorted(v for v in os.listdir(AD) if os.path.isdir(os.path.join(AD, v))
              and os.path.exists(os.path.join(a.root, "pv", v + ".bin")))
rng.shuffle(vids)
val = set(vids[:a.val_videos])
print(f"영상 {len(vids)} · 검증 {len(val)} (장면 단위 분리)")

stat = collections.Counter()
rows = {"train": [], "val": []}
for vn in sorted(vids):
    d = os.path.join(AD, vn)
    labs = [l.strip() for l in open(d + "/labels.csv") if l.strip()]
    # 개체별 (타임스탬프 → 박스), 그리고 위치 구간
    boxes = {}
    bd = d + "/2d_bbox_annot"
    for f in sorted(os.listdir(bd)) if os.path.isdir(bd) else []:
        if not f.endswith(".txt"): continue
        oi = int(f[:-4]); bb = {}
        for line in open(os.path.join(bd, f)):
            p = line.split()
            if len(p) >= 5: bb[int(p[0])] = tuple(float(x) for x in p[1:5])
        if bb: boxes[oi] = bb
    segs = collections.defaultdict(list)
    for line in open(d + "/3d_center_annot.txt"):
        p = line.split()
        if len(p) >= 7: segs[int(p[2])].append((int(p[0]), int(p[1]), int(p[6])))
    if not boxes: stat["박스 없는 영상"] += 1; continue
    try: pv = PV(a.root, vn)
    except Exception as e: stat["pv 열기 실패"] += 1; continue
    split = "val" if vn in val else "train"
    def seg_of(oi, t):
        for k, (t0, t1, s) in enumerate(sorted(segs.get(oi, []))): 
            if t0 <= t <= t1: return k
        return -1
    # 같은 종류 다른 개체 묶기
    byb = collections.defaultdict(list)
    for oi in boxes: byb[base_label(labs[oi]) if oi < len(labs) else "?"].append(oi)
    def pick(oi, seg=None, avoid=None):
        ts = [t for t in boxes[oi] if (seg is None or seg_of(oi, t) == seg) and t != avoid]
        rng.shuffle(ts)
        for t in ts:
            nt = pv.nearest(t)
            if nt is None: continue
            nm = f"{vn}_o{oi}_t{t}.jpg"
            if crop(pv, nt, boxes[oi][t], f"{OUT}/images/{nm}"): return nm
        return None
    for oi, bb in boxes.items():
        typ = base_label(labs[oi]) if oi < len(labs) else "object"
        ks = sorted({seg_of(oi, t) for t in bb} - {-1})
        for _ in range(a.per_obj):
            # 양성: 가능하면 서로 다른 위치 구간(= 옮겨진 뒤)
            if len(ks) >= 2:
                s1, s2 = rng.sample(ks, 2); moved = 1
            else:
                s1 = s2 = (ks[0] if ks else None); moved = 0
            r = pick(oi, s1)
            c = pick(oi, s2, avoid=None)
            if not r or not c or r == c: stat["양성 실패"] += 1; continue
            rows[split].append(dict(house=vn, oid=f"{typ}|{oi}", type=typ, ref=r, cands=[c],
                                    label="yes", real=1, moved=moved))
            stat[f"{split} 양성" + ("(이동후)" if moved else "")] += 1
            # 어려운 음성: 같은 종류 다른 개체
            others = [o for o in byb.get(typ, []) if o != oi and o in boxes]
            if others:
                c2 = pick(rng.choice(others))
                if c2:
                    rows[split].append(dict(house=vn, oid=f"{typ}|{oi}", type=typ, ref=r, cands=[c2],
                                            label="no", real=1, hard=1))
                    stat[f"{split} 어려운음성"] += 1
            # 쉬운 음성: 다른 종류
            oth = [o for o in boxes if o != oi and base_label(labs[o] if o < len(labs) else "?") != typ]
            if oth:
                c3 = pick(rng.choice(oth))
                if c3:
                    rows[split].append(dict(house=vn, oid=f"{typ}|{oi}", type=typ, ref=r, cands=[c3],
                                            label="no", real=1))
                    stat[f"{split} 쉬운음성"] += 1
for s in ("train", "val"):
    with open(f"{OUT}/{s}.jsonl", "w") as fo:
        for r in rows[s]: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print("IT3D_ADOPT_DONE", dict(stat), "→", OUT)
