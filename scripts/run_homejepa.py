#!/usr/bin/env python3
"""최종 파이프라인 — 우리 지각 씬그래프에 **home-jepa 학습 모델**을 태운다.

    $P scripts/run_homejepa.py --seq <name> --graphs graph_gtdepth graph_da3lc_aligned \
        --models supervised_two_head_v5

역할 분담:
    우리      RGB → 뎁스·포즈 → 씬그래프 → **관측 이벤트(POS/NEG)**
    home-jepa 그 이벤트를 읽고 **안 보이는 사이 물체가 어디 있는지** 추론
    채점      GT 좌표로 매긴 수용체가 정답

베이스라인(last-known)도 함께 낸다 — 모델이 그걸 넘어서야 의미가 있다.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HJ = os.path.expanduser("~/work/home-jepa")
sys.path.insert(0, HJ)

from kx.eval.homejepa_export import build_episode           # noqa: E402
from kx.eval.room_belief import load_regions                # noqa: E402
from kx.graph.regions import assign                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--ref", default="gtdepth", help="구역지도 태그")
    ap.add_argument("--models", default="supervised_two_head_v5")
    ap.add_argument("--hj", default=HJ)
    ap.add_argument("--map", default="", help='어휘 밖 카테고리 억지 사상. 예: "wall artwork=book"')
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.hj, "scripts"))
    from homejepa.model import EpTensors                     # noqa
    from reeval import build_jepa_probe, build_supervised    # noqa
    from train_supervised import evaluate                    # noqa

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    meta = json.load(open(os.path.join(seq_dir, "graph_%s.json" % args.ref)))["regions"]
    ref = load_regions(np.load(os.path.join(seq_dir, "regions_%s.npz" % args.ref)),
                       meta["zone_names"], meta["up"])

    def zone_of(p):
        return assign(ref, p)[1]

    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    extra = dict(kv.split('=', 1) for kv in args.map.split(',') if '=' in kv)
    dev = torch.device(args.device)
    results = {}

    for gname in args.graphs:
        p = os.path.join(seq_dir, gname if gname.endswith(".json") else gname + ".json")
        if not os.path.exists(p):
            print("(없음) %s" % gname)
            continue
        g = json.load(open(p))
        ep = build_episode(g, gt, zone_of, poses=poses, extra_cls=extra)
        s = ep["source"]
        ep_path = os.path.join(seq_dir, "hjepa_ep_%s.json" % gname.replace("graph_", "").replace(".json", ""))
        json.dump(ep, open(ep_path, "w"))
        print("\n=== %s" % gname)
        print("   에피소드: 방 %d · 수용체 %d · 물체 %d · 이벤트 %d · 질의 %d"
              % (len(ep["home"]["rooms"]), s["n_recepts"], s["n_objects"],
                 s["n_events"], s["n_queries"]))
        if s["n_queries"] == 0 or s["n_objects"] == 0:
            print("   질의/물체가 없어 건너뜀")
            continue

        # last-known 베이스라인 (모델 없이)
        qs = ep["queries"]
        base = float(np.mean([q["last_recept"] == q["gt_recept"] for q in qs]))
        base_room = float(np.mean([q["last_room"] == q["gt_room"] for q in qs]))
        moved = [q for q in qs if q["moved"]]
        base_moved = (float(np.mean([q["last_recept"] == q["gt_recept"] for q in moved]))
                      if moved else None)
        print("   last-known  수용체 %.3f · 방 %.3f · (이동한 질의 %d개에서 %s)"
              % (base, base_room, len(moved),
                 "%.3f" % base_moved if base_moved is not None else "-"))

        eps = [EpTensors(ep, 256, noid=True)]
        row = {"n_queries": len(qs), "baseline_recept": base, "baseline_room": base_room,
               "n_moved": len(moved), "baseline_recept_moved": base_moved, "models": {}}
        for name in args.models.split(","):
            fp = os.path.join(args.hj, "results", name + ".pt")
            if not os.path.exists(fp):
                print("   (체크포인트 없음) %s" % name)
                continue
            ck = torch.load(fp, map_location=dev, weights_only=False)
            builder = build_jepa_probe if name.startswith("jepa_") else build_supervised
            model, _ = builder(ck, dev)
            model = model.to(dev).eval()
            nll, agg = evaluate(model, eps, dev)
            summ = agg.summary()
            row["models"][name] = {"nll": float(nll), "summary": summ}
            a = summ.get("all", {})
            print("   %-26s top1 %.3f  top2 %.3f  (nll %.3f)"
                  % (name, a.get("top1", float("nan")), a.get("top2", float("nan")), nll))
            for strat in ("stayed", "moved", "moved_room"):
                if strat in summ:
                    print("      %-12s top1 %.3f  (n=%s)"
                          % (strat, summ[strat].get("top1", float("nan")), summ[strat].get("n")))
            del model
        results[gname] = row

    out = args.out or os.path.join(seq_dir, "homejepa_eval.json")
    json.dump(results, open(out, "w"), indent=1)
    print("\n→ %s" % out)


if __name__ == "__main__":
    main()
