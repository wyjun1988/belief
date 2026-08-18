#!/usr/bin/env python3
"""Nymeria 다세션 씬그래프 — (a) 공통 좌표계 + (b) OWLv2 지각을 합친다.

    $P scripts/nymeria_graph.py --stage detect     # 1fps 프레임 → OWLv2 검출
    $P scripts/nymeria_graph.py --stage build      # 검출 + 3D 방 → 씬그래프
    $P scripts/nymeria_graph.py --stage belief     # 세션 간 물체 위치 추론

두 실측 위에 선다:
  (a) Loc_49 11세션이 **하나의 좌표계**를 공유한다(graph_uid 1종, 좌표 ±8m).
      SuperMemory 의 최대 제약(세션별 좌표계)이 없다.
  (b) OWLv2 박스 검출이 CLIP 전체프레임보다 **F1 0.27→0.68**(재현율 0.20→0.72).

그래서 여기서는 지각을 OWLv2 로 하고, 검출된 물체를 **카메라 위치의 방**에 귀속시켜
다세션 씬그래프를 만든다. 벽이 있는 진짜 집이라 방 경계가 실재한다.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "nymeria")

# 집 안 물체·가구 어휘 (열린 어휘 — 필요하면 추가만 하면 된다)
VOCAB = ["refrigerator", "stove", "sink", "microwave", "oven", "kitchen counter",
         "kitchen cabinet", "drawer", "dining table", "chair", "sofa", "coffee table",
         "tv", "bookshelf", "bed", "nightstand", "wardrobe", "desk", "laptop",
         "mug", "bottle", "plate", "bowl", "pot", "pan", "knife", "cutting board",
         "backpack", "shoes", "door", "window", "lamp", "plant", "trash can"]


def traj_of(seq_dir, every=50):
    f = os.path.join(seq_dir, "recording_head", "mps", "slam",
                     "closed_loop_trajectory.csv")
    t = pd.read_csv(f, usecols=["tracking_timestamp_us", "tx_world_device",
                                "ty_world_device", "tz_world_device"])
    t = t.iloc[::every]
    sec = (t["tracking_timestamp_us"].values - t["tracking_timestamp_us"].values[0]) / 1e6
    return sec, t[["tx_world_device", "ty_world_device", "tz_world_device"]].values


def house_frame(seqs):
    """전 세션 공통 중력축·바닥 기저. (a) 에서 좌표계가 공통임을 확인했다."""
    P = np.concatenate([p for _, p in seqs.values()])
    C = P - P.mean(0)
    idx = np.random.default_rng(0).choice(len(C), min(20000, len(C)), replace=False)
    _, _, Vt = np.linalg.svd(C[idx], full_matrices=True)
    g = Vt[-1] / np.linalg.norm(Vt[-1])
    g = g if g[2] > 0 else -g
    e1 = np.array([1.0, 0, 0])
    e1 = e1 - g * (g @ e1)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    uv = np.stack([P @ e1, P @ e2], 1)
    return g, e1, e2, uv.mean(0)


def stage_detect(args):
    """1fps 프레임 → OWLv2 박스 검출 (프레임당 어휘 전체)."""
    import cv2
    from PIL import Image
    from scripts.owl_detect import detect, load_owl
    proc, model = load_owl(args.device)
    # ⚠️ Nymeria 의 video_main_rgb 는 확장자가 .zip 이지만 실제로는 mp4 다(실측).
    vids = sorted(glob.glob(os.path.join(D, "loc49_rgb", "*.mp4")))
    print("영상 %d개" % len(vids))
    out = {}
    for v in vids:
        name = os.path.basename(v)[:-4]
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        for sec in range(0, int(n / fps), args.every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, fr = cap.read()
            if not ok:
                break
            im = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (640, 640)),
                                              cv2.COLOR_BGR2RGB))
            d = detect(proc, model, [im], ["a photo of a " + w for w in VOCAB],
                       args.device, args.thr)[0]
            rows.append(dict(sec=sec, det={VOCAB[k]: round(v_, 3) for k, v_ in d.items()}))
        cap.release()
        out[name] = rows
        print("  %-44s %d프레임 · 검출 중앙 %.1f개"
              % (name[:44], len(rows), np.median([len(r["det"]) for r in rows]) if rows else 0))
        json.dump(out, open(os.path.join(D, "owl_det.json"), "w"))
    print("→ %s" % os.path.join(D, "owl_det.json"))


def stage_build(args):
    """검출 + 3D 방 → 다세션 씬그래프."""
    from scipy.cluster.vq import kmeans2
    seqs = {}
    for sd in sorted(glob.glob(os.path.join(D, "loc49", "*"))):
        if not os.path.isdir(sd):
            continue
        try:
            seqs[os.path.basename(sd)] = traj_of(sd)
        except Exception:
            continue
    g, e1, e2, ctr = house_frame(seqs)
    P = np.concatenate([p for _, p in seqs.values()])
    U = np.stack([P @ e1, P @ e2], 1) - ctr
    cen, _ = kmeans2(U, args.k, minit="++", seed=0, iter=60)
    print("집 지도: %d방 · 점유 %.1f×%.1f m · 세션 %d"
          % (args.k, U[:, 0].ptp(), U[:, 1].ptp(), len(seqs)))

    det_p = os.path.join(D, "owl_det.json")
    if not os.path.exists(det_p):
        print("검출 결과 없음 — --stage detect 를 먼저 돌려라")
        return
    det = json.load(open(det_p))
    graph = defaultdict(lambda: defaultdict(list))     # 방 → 물체 → [(세션, 초, 점수)]
    for name, rows in det.items():
        if name not in seqs:
            continue
        sec, Pp = seqs[name]
        uv = np.stack([Pp @ e1, Pp @ e2], 1) - ctr
        for r in rows:
            k = np.argmin(np.abs(sec - r["sec"]))
            room = int(np.argmin(np.linalg.norm(uv[k] - cen, axis=1)))
            for w, s in r["det"].items():
                graph[room][w].append((name, r["sec"], s))
    print("\n방별 물체(관측 수 상위):")
    for room in sorted(graph):
        top = sorted(graph[room].items(), key=lambda t: -len(t[1]))[:8]
        nsess = len({n for v in graph[room].values() for n, _, _ in v})
        print("  방%d (세션 %d개): %s" % (room, nsess,
              ", ".join("%s×%d" % (w, len(v)) for w, v in top)))
    json.dump({str(k): {w: [(n, s, sc) for n, s, sc in v] for w, v in vv.items()}
               for k, vv in graph.items()},
              open(os.path.join(D, "scene_graph.json"), "w"))
    print("→ %s" % os.path.join(D, "scene_graph.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["detect", "build"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=10, help="검출 간격(초)")
    ap.add_argument("--thr", type=float, default=0.12)
    ap.add_argument("--k", type=int, default=6, help="방 개수")
    args = ap.parse_args()
    (stage_detect if args.stage == "detect" else stage_build)(args)


if __name__ == "__main__":
    main()
