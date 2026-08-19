#!/usr/bin/env python3
"""**home-jepa 모델에게 직접 묻는다** — "이 물건 지금 어디 있어?"

    $P scripts/ask_homejepa.py --seq <name> --graph graph_gtdepth --obj PictureFrame

앞선 `run_homejepa.py` 는 집계 점수만 냈다. 여기서는 특정 물체에 대해 **매 틱 모델이
무엇이라고 답했는지**(예측 수용체와 그 방), GT 는 무엇인지, last-known 은 무엇인지를
나란히 찍는다. "마지막에 액자 어디 있냐고 물으면 거실이라고 답하는가"를 직접 확인하기 위해서다.
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
sys.path.insert(0, os.path.join(HJ, "scripts"))

from kx.eval.homejepa_export import build_episode      # noqa: E402
from kx.eval.room_belief import load_regions           # noqa: E402
from kx.graph.regions import assign                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--ref", default="gtdepth")
    ap.add_argument("--model", default="supervised_two_head_v5")
    ap.add_argument("--obj", required=True, help="물체 이름 부분일치")
    ap.add_argument("--map", default="", help='어휘 밖 카테고리 억지 사상. 예: "wall artwork=book"')
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from homejepa.model import EpTensors, make_batch
    from reeval import build_jepa_probe, build_supervised

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    meta = json.load(open(os.path.join(seq_dir, "graph_%s.json" % args.ref)))["regions"]
    ref = load_regions(np.load(os.path.join(seq_dir, "regions_%s.npz" % args.ref)),
                       meta["zone_names"], meta["up"])
    g = json.load(open(os.path.join(seq_dir, args.graph + ".json")))
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    extra = dict(kv.split('=', 1) for kv in args.map.split(',') if '=' in kv)
    ep = build_episode(g, gt, lambda p: assign(ref, p)[1], poses=poses, extra_cls=extra)

    # 질의 대상 물체 찾기
    name2oid = {}
    for o in ep["home"]["objects"]:
        src = g["objects"].get(o["src_instance"], {})
        name2oid.setdefault(src.get("name") or "", []).append(o["id"])
    hits = {n: v for n, v in name2oid.items() if args.obj.lower() in n.lower()}
    if not hits:
        sys.exit("에피소드에 그 물체가 없다: %s  (있는 것: %s)"
                 % (args.obj, sorted(k for k in name2oid if k)[:12]))
    oids = {o for v in hits.values() for o in v}
    print("대상: %s → 에피소드 물체 id %s" % (list(hits), sorted(oids)))

    rec = {r["id"]: r for r in ep["home"]["recepts"]}
    room = {r["id"]: r for r in ep["home"]["rooms"]}

    def label(lid):
        r = rec[lid]
        return "%s(%s)" % (r.get("name") or r["type"], room[r["room"]]["type"])

    dev = torch.device(args.device)
    fp = os.path.join(HJ, "results", args.model + ".pt")
    ck = torch.load(fp, map_location=dev, weights_only=False)
    builder = build_jepa_probe if args.model.startswith("jepa_") else build_supervised
    model, _ = builder(ck, dev)
    model = model.to(dev).eval()

    eps = [EpTensors(ep, 256, noid=True)]
    E = eps[0]
    sel = [(0, qi) for qi in range(len(E.queries)) if E.queries[qi]["meta"]["obj"] in oids]
    if not sel:
        sys.exit("그 물체에 대한 질의가 없다 (관측 이력이 없어 질의가 생기지 않음)")

    with torch.no_grad():
        b = make_batch(eps, sel, dev)
        prob = model.log_prob(b).exp().cpu().numpy()

    loc_ids = [r["id"] for r in E.recepts]
    print("\n%-6s %-26s %-26s %-26s %s" % ("틱", "모델 답", "GT", "last-known", "확률"))
    print("-" * 118)
    for j, (_, qi) in enumerate(sel):
        q = E.queries[qi]["meta"]
        k = int(np.argmax(prob[j]))
        pred = loc_ids[k]
        mark = "O" if pred == q["gt_recept"] else ("~방맞음" if rec[pred]["room"] == q["gt_room"] else "X")
        print("%-6d %-26s %-26s %-26s %.2f  %s"
              % (q["qt"], label(pred)[:26], label(q["gt_recept"])[:26],
                 label(q["last_recept"])[:26], prob[j, k], mark))

    last = sel[-1]
    q = E.queries[last[1]]["meta"]
    k = int(np.argmax(prob[-1]))
    print("\n마지막 질의(틱 %d, %.0f초):" % (q["qt"], q["qt"] * 5))
    print("   home-jepa 모델 답 : %s" % label(loc_ids[k]))
    print("   GT                : %s" % label(q["gt_recept"]))


if __name__ == "__main__":
    main()
