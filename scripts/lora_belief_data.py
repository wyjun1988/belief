#!/usr/bin/env python3
"""belief 답변 학습셋 — "사라졌다, 어느 방으로 갔을까" (2026-09-18).

왜: ③ 에서 인계가 성공해도 **최종 답은 belief 몫**인데 지금이 최악이다.
  c3big 실측 — 인계 73건 중 정답 5건 = **0.068**. 집당 방이 10개라 **무작위가 0.109** 다.
  현행은 `사전확률(type,room) × 방 통계` 인데 질의 종류 대부분이 사전확률 표에 없어 평평해지고,
  카메라가 오래 머문 방으로 끌려간다(고른 방 상위: 화장실·기타방 / 실제: 욕실·침실).

형식: 다른 세 판정기와 **같은 yes/no 마진** 구조. 방마다 한 번 물어 argmax 를 답으로 쓴다.
  "머그가 kitchen 에서 사라졌다. 지금 bathroom 에 있을까?" → yes/no
  이러면 기존 학습기·추론기·문턱 기계를 그대로 쓴다(방 이름이 집마다 달라 생성형 답은 채점이 어렵다).

  python scripts/lora_belief_data.py data/hssd_c3big data/hssd_c2set ~/khcache/lora_belief --val-houses 20
"""
import argparse, collections, json, math, os, random

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+", help="마지막 인자가 출력 경로")
ap.add_argument("--val-houses", type=int, default=20)
ap.add_argument("--neg-per-pos", type=int, default=3, help="정답 방 1개당 오답 방 표본 수")
ap.add_argument("--pad", type=float, default=0.3)
ap.add_argument("--img-w", type=int, default=448)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
OUT = os.path.expanduser(a.roots[-1]); roots = [os.path.expanduser(r) for r in a.roots[:-1]]
rng = random.Random(a.seed); os.makedirs(OUT + "/images", exist_ok=True)
from PIL import Image

def save_crop(src, box, W, H, dst):
    if os.path.exists(dst): return True
    if not os.path.exists(src): return False
    try: im = Image.open(src).convert("RGB")
    except Exception: return False
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
    x0, y0 = max(0, int(x0 - pw)), max(0, int(y0 - ph))
    x1, y1 = min(im.width, int(x1 + pw)), min(im.height, int(y1 + ph))
    if x1 - x0 < 16 or y1 - y0 < 16: return False
    c = im.crop((x0, y0, x1, y1))
    if c.width > a.img_w: c = c.resize((a.img_w, max(1, int(c.height * a.img_w / c.width))))
    c.save(dst, quality=90); return True

houses = []
for r in roots:
    for h in sorted(os.listdir(r)):
        if h.startswith("house_") and os.path.exists(f"{r}/{h}/gt.json"): houses.append((r, h))
rng.shuffle(houses); val = set(houses[:a.val_houses])
stat = collections.Counter(); rows = {"train": [], "val": []}
for r, hn in sorted(houses):
    g = json.load(open(f"{r}/{hn}/gt.json"))
    gt0 = g["gt0"]; live = {l["t"]: l for l in g["live"]}
    rooms = sorted({v["room"] for v in gt0.values() if v.get("room")})
    if len(rooms) < 3: stat["방 3개 미만"] += 1; continue
    split = "val" if (r, hn) in val else "train"
    cnt = collections.Counter(v["type"] for v in gt0.values())
    for m in (g.get("moves") or []):
        oid = m["oid"]; v0 = gt0.get(oid)
        if not v0 or not v0.get("room") or cnt[v0["type"]] > 1: continue
        dest = m.get("to") or m.get("room") or m.get("to_room")
        if not dest:                                    # 목적지 방을 좌표로 찾는다
            polys = (g.get("scene_meta") or {}).get("polys") or {}
            dest = None
            if m.get("pos") and polys:
                px, pz = m["pos"][0], m["pos"][2]
                for rr, pl in polys.items():
                    pls = pl if (pl and isinstance(pl[0][0], (list, tuple))) else [pl]
                    for q in pls:
                        ins = False; n = len(q)
                        for k in range(n):
                            x1, z1 = q[k][0], q[k][-1]; x2, z2 = q[(k + 1) % n][0], q[(k + 1) % n][-1]
                            if (z1 > pz) != (z2 > pz) and px < (x2 - x1) * (pz - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
                        if ins: dest = rr; break
                    if dest: break
        if not dest or dest not in rooms or dest == v0["room"]: stat["목적지 불명"] += 1; continue
        # 참조 크롭: 물체가 가장 가까이 보이는 **스캔(지도) 프레임** + 박스 (lora_adopt_data.py 와 같은 방식)
        c0 = [(mm["dist"][oid], k) for k, mm in enumerate(g["map"])
              if oid in (mm.get("ctr") or {}) and oid in (mm.get("dist") or {})]
        if not c0: stat["참조 없음"] += 1; continue
        _, k0 = min(c0); mk = g["map"][k0]
        b = (mk.get("box") or {}).get(oid)
        if not b:
            c = mk["ctr"][oid]; b = [c[0] - 60, c[1] - 60, c[0] + 60, c[1] + 60]
        nm = "%s_map%04d_%s.jpg" % (hn, k0, oid.replace("|", "_").replace(" ", "_"))
        if not save_crop(f"{r}/{hn}/map/%04d.jpg" % k0, b, 0, 0, f"{OUT}/images/{nm}"):
            stat["크롭 실패"] += 1; continue
        negs = [x for x in rooms if x not in (dest, v0["room"])]
        rng.shuffle(negs)
        base = dict(house=hn, oid=oid, type=v0["type"], ref=nm, cands=[nm],
                    from_room=v0["room"], rooms=rooms)
        rows[split].append(dict(base, room=dest, label="yes")); stat[f"{split} 양성"] += 1
        for q in negs[:a.neg_per_pos]:
            rows[split].append(dict(base, room=q, label="no")); stat[f"{split} 음성"] += 1
for s in ("train", "val"):
    with open(f"{OUT}/{s}.jsonl", "w") as fo:
        for x in rows[s]: fo.write(json.dumps(x, ensure_ascii=False) + "\n")
print("BELIEF_DATA_DONE", dict(stat), "→", OUT)
