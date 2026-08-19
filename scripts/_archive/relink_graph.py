#!/usr/bin/env python3
"""트랙 파편을 belief 층에서 잇는다 — "여기서 사라지고 저기서 나타났다".

    $P scripts/relink_graph.py --seq <name> --graph graph_sam.json --sam sam_daaam

부재 증거(그 자리를 다시 봤는데 없더라)를 카메라 기하로 계산하고, 떠난 조각과 새로
나타난 조각을 외형(CLIP)·크기·시간으로 이어 하나의 물체 노드로 합친다.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.graph.relink import departure_times, link_fragments, merge   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def track_embeddings(sam_dir, every=4, max_frames=400):
    """트랙별 평균 CLIP 임베딩 — meta 의 순서가 clip 배열의 행 순서와 같다."""
    md, cd = os.path.join(sam_dir, "meta"), os.path.join(sam_dir, "clip")
    if not os.path.isdir(cd):
        return {}
    acc = {}
    files = sorted(os.listdir(md))[::every][:max_frames]
    for f in files:
        i = os.path.splitext(f)[0]
        cp = os.path.join(cd, i + ".npy")
        if not os.path.exists(cp):
            continue
        meta = json.load(open(os.path.join(md, f)))
        E = np.load(cp).astype(np.float32)
        for k, m in enumerate(meta):
            if k < len(E):
                acc.setdefault(int(m["track_id"]), []).append(E[k])
    out = {}
    for tid, v in acc.items():
        e = np.mean(v, axis=0)
        n = np.linalg.norm(e)
        if n > 0:
            out[tid] = e / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", required=True)
    ap.add_argument("--sam", default="sam_daaam")
    ap.add_argument("--depth", default=None, help="가림 판정용 뎁스 폴더 (없으면 생략)")
    ap.add_argument("--pose", default=None)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--link-min", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"])
    poses = np.loadtxt(os.path.join(seq_dir, args.pose or g.get("pose_file", "pose/poses.txt"))
                       ).reshape(-1, 4, 4)
    sam_dir = os.path.join(seq_dir, args.sam)
    seg_dir = os.path.join(sam_dir, "seg")
    dep_dir = os.path.join(seq_dir, args.depth) if args.depth else None

    def seg_reader(i):
        p = os.path.join(seg_dir, "%06d.png" % i)
        return np.array(Image.open(p)) if os.path.exists(p) else None

    def depth_reader(i):
        p = os.path.join(dep_dir, "%06d.png" % i)
        return (np.array(Image.open(p)).astype(np.float32) / 1000.0
                if os.path.exists(p) else None)

    print("부재 증거 계산 중 (그 자리를 다시 봤는데 없었나)...", flush=True)
    dep = departure_times(g, poses, K, cam["width"], cam["height"], seg_reader,
                          depth_reader if dep_dir else None, every=args.every)
    print("  떠난 것으로 판정된 조각: %d / %d" % (len(dep), len(g["objects"])))

    emb = track_embeddings(sam_dir)
    print("  CLIP 임베딩 있는 트랙 %d개" % len(emb))

    links = link_fragments(g, dep, emb=emb, fps=cam.get("fps", 10.0), link_min=args.link_min)
    print("  연결 가설 %d건" % len(links))
    for L in sorted(links, key=lambda x: -x["score"])[:12]:
        print("     %s → %s  점수 %.2f (외형 %.2f 크기 %.2f)  f%d→f%d  %.1fs  이동 %.2fm"
              % (L["from"][:8], L["to"][:8], L["score"], L["appearance"], L["size"],
                 L["departed_at"], L["appeared_at"], L["gap_s"],
                 float(np.linalg.norm(np.array(L["to_pos"]) - np.array(L["from_pos"])))))

    # --- 채점: 연결이 실제로 같은 물체였나 (gt_instance 로) ---
    ids = json.load(open(os.path.join(sam_dir, "seg_ids.json")))
    gtof = {str(v.get("instance_id")): v.get("gt_instance") for v in ids.values()}
    tp = sum(1 for L in links
             if gtof.get(str(L["from"])) is not None
             and gtof.get(str(L["from"])) == gtof.get(str(L["to"])))
    print("\n[채점] 연결 %d건 중 같은 GT 물체 %d건 = precision %.3f"
          % (len(links), tp, tp / max(len(links), 1)))
    # 이을 수 있었던 쌍(같은 GT 물체의 서로 다른 조각) 대비 recall
    from collections import defaultdict as _dd
    byg = _dd(list)
    for iid, o in g["objects"].items():
        gg = gtof.get(str(iid))
        if gg is not None:
            byg[gg].append(iid)
    linkable = sum(len(v) - 1 for v in byg.values() if len(v) > 1)
    print("      이을 수 있었던 조각쌍 %d개 (파편난 GT 물체 %d개) → recall %.3f"
          % (linkable, sum(1 for v in byg.values() if len(v) > 1), tp / max(linkable, 1)))

    g2, mapping = merge(g, links)
    for iid, o in g2["objects"].items():
        o["departure"] = dep.get(iid)
    out = args.out or args.graph.replace(".json", "_relinked.json")
    json.dump(g2, open(os.path.join(seq_dir, out), "w"))
    print("\n물체 노드 %d → %d (연결 %d건)" % (g2["relink"]["n_before"], g2["relink"]["n_after"], len(links)))
    print("→ %s" % out)


if __name__ == "__main__":
    main()
