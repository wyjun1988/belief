#!/usr/bin/env python3
"""시뮬 프로브 공통 쌍 생성기 — og_probe/hab_probe 의 덤프에서 검증기 쌍을 만든다.

    python scripts/probe_pairs.py --probe /tmp/og_probe --n 400 --out /tmp/og_pairs

입력 형식 (probe/meta.jsonl, 프레임당 한 줄):
    {"img": "frames/000012.jpg", "objs": [{"label": "sofa", "ctr": [u,v], "dist": 3.2}, ...]}
출력: exp_vlm_verify3 호환 meta.jsonl (cand/enroll/label/alt/truth/dist) + 크롭 jpg.
양성 = 그 물체 크롭 · 음성 = 같은 프레임 **다른 라벨** 물체 크롭에 이 라벨 (§make_sim_pairs 규약).
크롭 상자 = 프레임 짧은 변의 1/3 (THOR §101 기하와 동일 취지).
"""
import argparse, json, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(0)

frames = [json.loads(l) for l in open(os.path.join(args.probe, "meta.jsonl"))]
frames = [f for f in frames if len({o["label"] for o in f["objs"]}) >= 2]
assert frames, "라벨 2종+ 프레임이 없다 — 프로브 덤프 확인"
rng.shuffle(frames)

meta = []; k = 0
per = max(4, -(-args.n // max(len(frames), 1)))     # 프레임당 상한 (편중 방지)
for f in frames:
    if k >= args.n: break
    im = Image.open(os.path.join(args.probe, f["img"])).convert("RGB")
    W, H = im.size; h2 = min(W, H) // 6
    fk = 0
    objs = list(f["objs"]); rng.shuffle(objs)
    for a in range(len(objs)):
        if k >= args.n or fk >= per: break
        for b in range(len(objs)):
            if a == b or objs[a]["label"] == objs[b]["label"]: continue
            o1, o2 = objs[a], objs[b]
            def crop(o):
                cx, cy = int(o["ctr"][0]), int(o["ctr"][1])
                return im.crop((max(0, cx-h2), max(0, cy-h2),
                                min(W, cx+h2), min(H, cy+h2))).resize((336, 336), Image.LANCZOS)
            f1 = os.path.join(args.out, "cand_%04d.jpg" % k); crop(o1).save(f1, quality=92)
            meta.append(dict(cand=f1, enroll=f1, label=o1["label"], alt=o2["label"],
                             truth=1, dist=round(float(o1["dist"]), 1))); k += 1; fk += 1
            if k >= args.n: break
            f2 = os.path.join(args.out, "cand_%04d.jpg" % k); crop(o2).save(f2, quality=92)
            meta.append(dict(cand=f2, enroll=f2, label=o1["label"], alt=o2["label"],
                             truth=0, dist=round(float(o2["dist"]), 1))); k += 1; fk += 1
            break
assert k > 0, "쌍 0개 — 조용히 성공한 척 금지 (§make_sim_pairs 교훈)"
open(os.path.join(args.out, "meta.jsonl"), "w").write("\n".join(json.dumps(m) for m in meta))
print("쌍 %d → %s" % (k, args.out))
