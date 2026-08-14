"""한 물체의 시간축 3단 비교 — GT 구역 / 씬그래프 구역 / home-jepa 모델 답.

    $P kx/viz/obj_timeline.py --seq <name> --obj BlackSquarePictureFrame

씬그래프가 "언제" 바뀌는지, 모델 답이 그것을 언제 따라잡는지를 한 장으로 본다.
모델 답은 `ask_homejepa.py` 와 같은 경로(build_episode → 학습 체크포인트)로 다시 낸다.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
HJ = os.path.expanduser("~/work/home-jepa")
sys.path.insert(0, HJ)
sys.path.insert(0, os.path.join(HJ, "scripts"))

from kx.eval.homejepa_export import build_episode      # noqa: E402
from kx.eval.room_belief import load_regions           # noqa: E402
from kx.graph.regions import assign                    # noqa: E402

COL = {"kitchen": "#f0aa3c", "living": "#4aa0f0", "dining": "#78c878",
       "bedroom": "#c86ec8", "bathroom": "#8ecfe0", "office": "#c8c878", None: "#dddddd"}
ORDER = ["bedroom", "kitchen", "dining", "living"]
Z2R = {"living_room": "living", "kitchen": "kitchen", "dining_room": "dining",
       "bedroom": "bedroom", "bathroom": "bathroom", "study": "office"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--ref", default="gtdepth")
    ap.add_argument("--obj", default="BlackSquarePictureFrame")
    ap.add_argument("--model", default="supervised_two_head_v5")
    ap.add_argument("--map", default="wall artwork=book")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "obj_timeline.png"))
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph + ".json")))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    meta = json.load(open(os.path.join(seq_dir, "graph_%s.json" % args.ref)))["regions"]
    ref = load_regions(np.load(os.path.join(seq_dir, "regions_%s.npz" % args.ref)),
                       meta["zone_names"], meta["up"])
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    N = g["n_frames"]

    iid = next(k for k, o in g["objects"].items() if args.obj.lower() in (o.get("name") or "").lower())
    obj = g["objects"][iid]

    # ⚠️ GT 레인은 **채점에 쓰는 정답 그대로**(에피소드 질의의 gt_recept 의 방) 그린다.
    # GT 6DoF 를 바닥투영해 구역을 찍으면 벽걸이 액자처럼 경계 셀에 앉은 물체가
    # 2~3cm 흔들림에 bedroom↔living 을 오간다 — 그림이 정답을 잘못 그리게 된다.
    gt_zone = [None] * N

    # --- 2) 씬그래프 구역: 관측된 배치 구간만 (빈 구간 = 미관측) ------------------
    sg_zone = [None] * N
    for pl in obj["placements"]:
        for f in range(pl["start_frame"], min(pl["end_frame"], N - 1) + 1):
            sg_zone[f] = pl.get("zone")

    # --- 3) 모델 답: 틱마다 argmax 수용체의 방 ----------------------------------
    from homejepa.model import EpTensors, make_batch
    from reeval import build_jepa_probe, build_supervised
    extra = dict(kv.split('=', 1) for kv in args.map.split(',') if '=' in kv)
    ep = build_episode(g, gt, lambda p: assign(ref, p)[1], poses=poses, extra_cls=extra)
    oids = {o["id"] for o in ep["home"]["objects"] if o["src_instance"] == str(iid)}
    rec = {r["id"]: r for r in ep["home"]["recepts"]}
    room = {r["id"]: r for r in ep["home"]["rooms"]}
    dev = torch.device("cpu")
    ck = torch.load(os.path.join(HJ, "results", args.model + ".pt"),
                    map_location=dev, weights_only=False)
    builder = build_jepa_probe if args.model.startswith("jepa_") else build_supervised
    model, _ = builder(ck, dev)
    model = model.to(dev).eval()
    E = EpTensors(ep, 256, noid=True)
    sel = [(0, qi) for qi in range(len(E.queries)) if E.queries[qi]["meta"]["obj"] in oids]
    with torch.no_grad():
        prob = model.log_prob(make_batch([E], sel, dev)).exp().cpu().numpy()
    loc_ids = [r["id"] for r in E.recepts]
    tick = 50
    gt_txt = {}
    md_zone, md_txt = [None] * N, {}
    for j, (_, qi) in enumerate(sel):
        q = E.queries[qi]["meta"]
        k = int(np.argmax(prob[j]))
        r = rec[loc_ids[k]]
        z = Z2R.get(room[r["room"]]["type"])
        f0, f1 = q["qt"] * tick, min((q["qt"] + 1) * tick, N)
        for f in range(f0, f1):
            md_zone[f] = z
        md_txt[q["qt"]] = (r.get("name") or r["type"], z, float(prob[j, k]))
        gr = rec[q["gt_recept"]]
        gz = Z2R.get(room[gr["room"]]["type"])
        for f in range(f0, f1):
            gt_zone[f] = gz
        gt_txt[q["qt"]] = (gr.get("name") or gr["type"], gz)

    # --- 그리기 ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 4.6))
    lanes = [("GT (정답)", gt_zone), ("씬그래프 (지각)", sg_zone), ("home-jepa 모델 답", md_zone)]
    t = np.arange(N) / args.fps
    for li, (nm, seq) in enumerate(lanes):
        y = 2 - li
        f = 0
        while f < N:
            e = f
            while e + 1 < N and seq[e + 1] == seq[f]:
                e += 1
            if seq[f] is not None:
                ax.barh(y, (e - f + 1) / args.fps, left=f / args.fps, height=0.62,
                        color=COL.get(seq[f], "#ccc"), edgecolor="white", linewidth=0.4)
                if (e - f) / args.fps > 6:
                    ax.text((f + e) / 2 / args.fps, y, seq[f], ha="center", va="center",
                            fontsize=9.5, weight="bold", color="white")
            f = e + 1
        ax.text(-1.2, y, nm, ha="right", va="center", fontsize=11, weight="bold")

    # 씬그래프의 안정 배치(수용체) 표시
    for pl in obj["placements"]:
        if not pl.get("stable"):
            continue
        f0, f1 = pl["start_frame"] / args.fps, pl["end_frame"] / args.fps
        ax.text((f0 + f1) / 2, 1.42, pl.get("support") or "?", ha="center", fontsize=9,
                color="#333")
        ax.plot([f0, f1], [1.36, 1.36], lw=1.6, color="#333")

    # GT 이동 순간
    prev = gt_zone[0]
    for f in range(1, N):
        if gt_zone[f] != prev and gt_zone[f] is not None:
            ax.axvline(f / args.fps, color="crimson", ls="--", lw=1.0, alpha=0.55)
            prev = gt_zone[f]

    ax.set_yticks([])
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(-1.3, N / args.fps + 0.5)
    ax.set_xlabel("시간 [s]")
    ax.set_title("%s — 구역 이력: GT vs 씬그래프 vs belief 모델  (%s, %s)"
                 % (obj["name"], g["sequence"][18:], args.graph), fontsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("→ %s" % args.out)
    print("%-5s %-26s %-26s %s" % ("틱", "모델 답", "GT", "p"))
    for tk in sorted(md_txt):
        m, gq = md_txt[tk], gt_txt[tk]
        mk = "O" if m[0] == gq[0] else ("~방맞음" if m[1] == gq[1] else "X")
        print("%-5d %-26s %-26s %.2f  %s" % (tk, "%s(%s)" % m[:2], "%s(%s)" % gq, m[2], mk))


if __name__ == "__main__":
    main()
