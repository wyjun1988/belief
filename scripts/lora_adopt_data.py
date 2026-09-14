#!/usr/bin/env python3
"""LoRA 학습셋 생성 — "② 채택 판정"(§166-69: 채택 단계는 문턱으로 포화, 판별 신호가 필요하다) (2026-09-14).

질문은 부재 판정과 다르다. 부재는 "이 자리에 아직 있나", 채택은 **"이 후보 크롭이 기록 시점의 그 물체인가"** 다.
파이프라인이 실제로 보는 것과 같은 재료만 쓴다 — 참조(기록 장면 + 빨간 박스) + 후보 프레임(빨간 박스) 1장.
라벨은 시뮬 GT 로 공짜: 후보 프레임 시각에 그 oid 가 GT 로 보이면 "yes", 아니면 "no".

후보는 **실제 검증기 통과 프레임**에서 뽑는다(VERIFY_JSONL) — 배포에서 만나는 분포 그대로.
없으면 OWL 점수 상위 프레임으로 후퇴한다(A3_PREFIX).

  VERIFY_JSONL=~/khcache/bench-v2full/scores/t1_all.jsonl A3_PREFIX=~/khcache/bench-v2full/cache/hs2_a_ \
  python scripts/lora_adopt_data.py data/hssd_v2 out/lora_adopt --per-target 6 --val-houses 25
→ out/{train,val}.jsonl + images/   (학습은 lora_presence_train.py 와 같은 형식: ref + cands + label)
"""
import argparse, json, glob, os, math, random, collections, numpy as np
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--per-target", type=int, default=6); ap.add_argument("--val-houses", type=int, default=25)
ap.add_argument("--img-w", type=int, default=448); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-houses", type=int, default=0); ap.add_argument("--neg-per-pos", type=float, default=2.0)
a = ap.parse_args(); rng = random.Random(a.seed)
VJ = os.path.expanduser(os.environ.get("VERIFY_JSONL", "")); A3P = os.path.expanduser(os.environ.get("A3_PREFIX", ""))
VTH, VTH2 = float(os.environ.get("VERIFY_TH", "2.069")), float(os.environ.get("VERIFY_TH2", "0.887"))
VSC = collections.defaultdict(list)
if VJ and os.path.exists(VJ):
    for ln in open(VJ):
        d = json.loads(ln); VSC[(d["house"], d["oid"])] = d.get("scored") or []
def save(src, dst, box=None):
    if os.path.exists(dst): return True
    if not os.path.exists(src): return False
    im = Image.open(src).convert("RGB"); w, h = im.size
    if box:
        x0, y0, x1, y1 = [max(0, min(w - 1 if i % 2 == 0 else h - 1, int(v))) for i, v in enumerate(box)]
        if x1 > x0 and y1 > y0: ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=max(3, w // 150))
    s = a.img_w / float(w); im.resize((a.img_w, max(8, int(h * s)))).save(dst, quality=90); return True
houses = sorted(glob.glob(os.path.join(a.root, "house_*")))
if a.max_houses: houses = houses[:a.max_houses]
rng.shuffle(houses); val = set(houses[:a.val_houses]); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
fo = {"train": open(os.path.join(a.out, "train.jsonl"), "w"), "val": open(os.path.join(a.out, "val.jsonl"), "w")}
stat = collections.Counter()
for hd in houses:
    hn = os.path.basename(hd); split = "val" if hd in val else "train"
    g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    ph = set()
    if os.path.exists(hd + "/phantom_ids.json"):
        _p = json.load(open(hd + "/phantom_ids.json")); _d = _p.get("phantom", _p) if isinstance(_p, dict) else {}
        ph = set(sum([v if isinstance(v, list) else v.get("ids", []) for v in _d.values()], [])) if isinstance(_d, dict) else set()
    cnt = collections.Counter(v["type"] for k, v in g["gt0"].items() if k not in ph)
    za = None; zp = A3P + hn + ".npz"
    if A3P and os.path.exists(zp):
        try: za = np.load(zp, allow_pickle=True)
        except Exception: za = None
    ts = za["ts"] if za is not None else None
    for oid, v0 in g["gt0"].items():
        if oid in ph or cnt[v0["type"]] > 1: continue
        typ = v0["type"].replace("_", " ").lower()
        # 참조: 물체가 가장 가까이 보이는 스캔(지도) 프레임 + 박스
        cands0 = [(m["dist"][oid], k) for k, m in enumerate(g["map"]) if oid in (m.get("ctr") or {}) and oid in (m.get("dist") or {})]
        if not cands0: stat["참조 없음"] += 1; continue
        d0, k0 = min(cands0); mk = g["map"][k0]; ctr0 = mk["ctr"][oid]
        bx = (mk.get("box") or {}).get(oid)
        ref = "%s_map%04d_%s.jpg" % (hn, k0, oid.replace("|", "_").replace(" ", "_"))
        rbox = bx if bx else [ctr0[0] - 60, ctr0[1] - 60, ctr0[0] + 60, ctr0[1] + 60]
        if not save(os.path.join(hd, "map", "%04d.jpg" % k0), os.path.join(a.out, "images", ref), rbox):
            stat["참조 파일 없음"] += 1; continue
        # 후보 프레임: 검증기 통과분 (없으면 OWL 상위)
        rows = VSC.get((hn, oid)) or []
        idx = [int(e[0]) for e in rows if e[1] >= VTH and (len(e) < 3 or e[2] >= VTH2)]
        if not idx and ts is not None: idx = list(range(0, len(ts), max(1, len(ts) // 40)))
        if ts is None or not idx: stat["후보 없음"] += 1; continue
        # 라벨은 프레임이 아니라 **박스**를 두고 매긴다 — 물체가 화면에 있어도 박스가 딴 데를 가리키면 그건 'no' 다.
        # (파일럿에서 프레임 기준 'yes' 의 22%가 박스는 엉뚱한 곳이었다 → 가르치려는 판별을 오히려 흐린다)
        def _box_of(i, t):
            if za is None or "bx" not in za.files: return None
            try:
                vocab = [str(x) for x in za["vocab"]]; ti = vocab.index(v0["type"])
                W = float(Image.open(os.path.join(hd, "live", "%06d.jpg" % t)).size[0])
                bcx, bcy, bw, bh = [float(x) * W for x in za["bx"][i, ti]]
                return (bcx, bcy, bw, bh) if (bw > 1 and bh > 1) else None
            except Exception: return None
        pos, neg = [], []
        for i in idx:
            if i >= len(ts): continue
            t = int(ts[i]); m = live.get(t)
            if m is None: continue
            b = _box_of(i, t); ctr = (m.get("ctr") or {}).get(oid)
            hit = bool(b and ctr and abs(ctr[0] - b[0]) <= b[2] / 2 and abs(ctr[1] - b[1]) <= b[3] / 2)
            (pos if hit else neg).append((i, t))
        if not pos and not neg: continue
        rng.shuffle(pos); rng.shuffle(neg)
        take_p = pos[:a.per_target]
        take_n = neg[:max(1, int(len(take_p) * a.neg_per_pos)) if take_p else min(2, len(neg))]
        for lab, group in (("yes", take_p), ("no", take_n)):
            for i, t in group:
                f = "%s_live%06d_%s.jpg" % (hn, t, oid.replace("|", "_").replace(" ", "_"))
                box = None
                if za is not None and "bx" in za.files:
                    try:
                        vocab = [str(x) for x in za["vocab"]]; ti = vocab.index(v0["type"])
                        # 박스는 max(W,H) 로 정규화돼 있다 (exp_t1_verify_mlx.py 와 같은 해석)
                        W = float(Image.open(os.path.join(hd, "live", "%06d.jpg" % t)).size[0])
                        bcx, bcy, bw, bh = [float(x) * W for x in za["bx"][i, ti]]
                        if bw > 1 and bh > 1: box = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
                    except Exception: box = None
                if not save(os.path.join(hd, "live", "%06d.jpg" % t), os.path.join(a.out, "images", f), box):
                    stat["후보 파일 없음"] += 1; continue
                fo[split].write(json.dumps(dict(house=hn, oid=oid, type=typ, ref=ref, cands=[f],
                                                label=lab, t=t, room=(live[t].get("room")), rec_room=v0.get("room"))) + "\n")
                stat[(split, lab)] += 1
for f in fo.values(): f.close()
print("LORA_ADOPT_DATA_DONE", dict(stat), "→", a.out)
