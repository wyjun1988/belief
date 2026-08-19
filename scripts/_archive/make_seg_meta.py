#!/usr/bin/env python3
"""SAM/BotSort 트랙에 메타데이터를 붙여 `seg_ids.json` 호환 파일을 만든다.

GT 마스크는 세 가지를 공짜로 줬다: ①마스크 ②연관 ③카테고리. SAM+BotSort 가 ①②를
대신하면 남는 ③을 채워야 파이프라인(구역 시드·이동 임계)이 돈다. 두 모드:

    --mode gt-match   트랙 수명 동안 가장 많이 겹친 GT 인스턴스의 메타를 물려준다.
                      → **연관까지만 SAM 것**이고 카테고리는 GT. 마스크·연관 품질만 격리.
    --mode clip       CLIP 제로샷으로 구역 시드 어휘를 분류한다. → 완전 자립.

동시에 트랙 품질 통계를 낸다 — 몇 프레임을 살아남았나, GT 인스턴스와 얼마나 순수하게
대응하나(한 트랙이 여러 물체를 오가면 랜드마크로 못 쓴다).

    $P scripts/make_seg_meta.py --seq <name> --sam sam_daaam --mode gt-match
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 구역 시드 어휘 — CLIP 제로샷의 후보. kx/graph/regions.SEED_CATEGORIES 와 짝을 맞춘다.
CLIP_VOCAB = [
    ("refrigerator", "a refrigerator"), ("oven", "an oven"), ("stove", "a kitchen stove"),
    ("dining table", "a dining table"), ("dining chair", "a dining chair"),
    ("couch", "a sofa couch"), ("television", "a television screen"),
    ("tv stand", "a tv stand cabinet"), ("coffee table", "a coffee table"),
    ("armchair", "an armchair"), ("bed frame", "a bed"), ("nightstand", "a nightstand"),
    ("dressing table", "a dressing table"), ("shelf", "a shelf"),
    ("cabinets and shelves", "a cabinet"), ("door", "a door"), ("wall artwork", "a picture frame"),
    ("house plant", "a house plant"), ("lamp", "a lamp"), ("box", "a box"),
    ("cup", "a cup"), ("book", "a book"), ("person", "a person"),
    ("floor", "a floor"), ("wall", "a plain wall"), ("ceiling", "a ceiling"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--sam", default="sam_daaam")
    ap.add_argument("--mode", default="gt-match", choices=["gt-match", "clip"])
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=5,
                    help="이만큼의 프레임에 나타나야 물체 노드 후보로 남긴다")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    sam_dir = os.path.join(seq_dir, args.sam)
    gt_ids = json.load(open(os.path.join(seq_dir, "gt", "seg_ids.json")))
    gt_obj = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    local2gt = {int(k): str(v["instance_id"]) for k, v in gt_ids.items()}

    frames = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(sam_dir, "seg")))
    overlap = defaultdict(Counter)      # track → GT instance 픽셀 겹침
    seen = Counter()
    areas = defaultdict(list)
    for i in frames[::args.every]:
        sp = os.path.join(sam_dir, "seg", "%06d.png" % i)
        gp = os.path.join(seq_dir, "gt", "seg", "%06d.png" % i)
        if not os.path.exists(gp):
            continue
        S = np.array(Image.open(sp))
        G = np.array(Image.open(gp))
        for tid in np.unique(S):
            if tid == 0:
                continue
            m = S == tid
            seen[int(tid)] += 1
            areas[int(tid)].append(int(m.sum()))
            g, c = np.unique(G[m], return_counts=True)
            for gg, cc in zip(g, c):
                if gg != 0:
                    overlap[int(tid)][local2gt.get(int(gg))] += int(cc)

    out, stats = {}, []
    for tid, nf in seen.items():
        if nf < args.min_frames:
            continue
        tot = sum(overlap[tid].values())
        best, bc = (overlap[tid].most_common(1)[0] if tot else (None, 0))
        purity = bc / max(tot, 1)
        rec = gt_obj.get(best) if best else None
        stats.append({"track": tid, "frames": nf, "purity": purity,
                      "gt": rec["name"] if rec else None,
                      "median_area": int(np.median(areas[tid]))})
        if args.mode == "gt-match" and rec is not None:
            # ⚠️ 노드 키는 **track id** 여야 한다. 여기에 GT 인스턴스 id 를 넣었더니
            # 파편난 트랙들이 그래프 빌더에서 같은 키로 덮어써져 저절로 합쳐졌고
            # (601트랙 → 140노드), 재연결 로직이 시험조차 되지 않았다.
            # GT 는 `gt_instance` 로 따로 두고 채점에만 쓴다.
            out[str(tid)] = {"instance_id": int(tid), "gt_instance": int(best),
                             "name": rec["name"],
                             "category": rec["category"],
                             "motion_type": (rec["motion_type"] or "").upper(),
                             "instance_type": "OBJECT",
                             "extent_m": rec.get("extent_m"),
                             "track_frames": nf, "gt_purity": round(purity, 3)}
        elif args.mode == "clip":
            out[str(tid)] = {"instance_id": tid, "name": "track_%d" % tid,
                             "category": None, "motion_type": "UNKNOWN",
                             "instance_type": "OBJECT", "track_frames": nf}

    p = os.path.join(sam_dir, "seg_ids.json")
    json.dump(out, open(p, "w"), indent=1)

    S = [s for s in stats]
    pur = np.array([s["purity"] for s in S]) if S else np.zeros(0)
    nf = np.array([s["frames"] for s in S]) if S else np.zeros(0)
    print("트랙 %d개 (>=%d프레임) / 전체 고유 id %d" % (len(S), args.min_frames, len(seen)))
    if len(S):
        print("  수명(프레임): 중앙 %d  p90 %d  최대 %d" % (np.median(nf), np.percentile(nf, 90), nf.max()))
        print("  GT 순도: 중앙 %.3f  · 0.8 이상 %d개 (%.1f%%)"
              % (np.median(pur), (pur >= 0.8).sum(), 100 * (pur >= 0.8).mean()))
        matched = len({s["gt"] for s in S if s["gt"]})
        print("  대응된 GT 인스턴스 %d개 (GT 총 %d개)" % (matched, len(gt_obj)))
        top = sorted(S, key=lambda s: -s["frames"])[:10]
        print("  긴 트랙 상위:")
        for s in top:
            print("     id=%-5d %4d프레임  순도 %.2f  %s" % (s["track"], s["frames"], s["purity"], s["gt"]))
    print("→ %s  (%d개 기재)" % (p, len(out)))


if __name__ == "__main__":
    main()
