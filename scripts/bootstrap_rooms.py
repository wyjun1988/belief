#!/usr/bin/env python3
"""**기하 없이** 물체→방 사전을 만든다 — LLM 시드 + 동시가시성 전파.

    $P scripts/bootstrap_rooms.py --margin 2.0

지금 방 서명의 사전은 v1 그래프의 3D 위치를 구역지도에 넣어 만든다. 그래서 포즈가
나쁘면(DA3) 사전이 통째로 틀어졌다(0.971 → 0.673). 여기서는 3D 를 **한 번도 쓰지 않고**
같은 사전을 만든다:

    ① 시드   LLM 이 이름만 보고 확신하는 소수 물체 (마진 게이트)
    ② 전파   **같이 보이면 같은 방** — 프레임 동시가시성만으로 나머지를 채운다

②가 핵심이다. 화병이 어느 방인지는 상식으로 못 맞히지만(어제 0.342 = 우연),
"그 화병이 냉장고와 늘 같이 보인다"는 관측은 3D 없이도 얻어진다.

⚠️ 시드가 한쪽 방에만 몰리면 전파가 그쪽으로 쏠린다. LLM 고신뢰 21개가 전부
kitchen·bedroom 이었으므로(living/dining 0개), 전파만으로 개방공간이 갈릴지는
이 스크립트가 답해야 할 질문이다.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kx.eval.room_belief import load_regions        # noqa: E402
from kx.graph.regions import assign                 # noqa: E402

VIS_MIN = 0.05


def co_visibility(gt_dir, step_ns=int(1e9)):
    """프레임별 동시가시 집합 — **기하 없음**. GT 는 '무엇이 보였나'에만 쓴다."""
    bb = pd.read_csv(os.path.join(gt_dir, "2d_bounding_box.csv"),
                     usecols=["object_uid", "timestamp[ns]", "visibility_ratio[%]"])
    bb = bb[bb["visibility_ratio[%]"] >= VIS_MIN]
    vis = bb.groupby("timestamp[ns]")["object_uid"].apply(set)
    ts = np.sort(bb["timestamp[ns]"].unique())
    keep, last = [], -(1 << 62)
    for t in ts:
        if t - last >= step_ns:
            keep.append(t)
            last = t
    return [set(int(x) for x in vis[t]) for t in keep]


def propagate(frames, seeds, rooms, iters=3, alpha=0.6):
    """동시가시성 전파. seeds = {uid: room}. 반환 {uid: (room, 확신도)}"""
    freq = Counter()
    co = defaultdict(Counter)
    for f in frames:
        fl = list(f)
        for a in fl:
            freq[a] += 1
        for i, a in enumerate(fl):
            for b in fl[i + 1:]:
                co[a][b] += 1
                co[b][a] += 1

    ridx = {r: i for i, r in enumerate(rooms)}
    objs = list(freq)
    oi = {o: i for i, o in enumerate(objs)}
    P = np.zeros((len(objs), len(rooms)))
    for o, r in seeds.items():
        if o in oi and r in ridx:
            P[oi[o], ridx[r]] = 1.0
    fixed = np.array([o in seeds for o in objs])

    # 코사인 정규화 동시발생 (흔한 물체가 과도한 표를 갖지 않게)
    W = np.zeros((len(objs), len(objs)))
    for a, cc in co.items():
        if a not in oi:
            continue
        for b, n in cc.items():
            if b in oi:
                W[oi[a], oi[b]] = n / np.sqrt(freq[a] * freq[b])
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)

    for _ in range(iters):
        Q = alpha * (W @ P) + (1 - alpha) * P
        Q[fixed] = P[fixed]                     # 시드는 고정
        s = Q.sum(1, keepdims=True)
        P = np.where(s > 0, Q / np.maximum(s, 1e-9), Q)

    out = {}
    for o, i in oi.items():
        if P[i].sum() <= 0:
            continue
        k = int(np.argmax(P[i]))
        srt = np.sort(P[i])[::-1]
        out[o] = (rooms[k], float(srt[0] - (srt[1] if len(srt) > 1 else 0)))
    return out


def cluster_rooms(frames, k, objs_keep=None):
    """동시가시성으로 물체를 k 개 군집으로 — **시드도 기하도 없이.**

    시드 전파는 시드가 없는 방으로 못 간다(실측: kitchen 시드 17개가 292/296 을 먹었다).
    군집은 그 문제가 없다 — 먼저 '같이 보이는 것끼리' 묶고, 묶인 뒤에 이름을 붙인다.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    freq = Counter()
    co = defaultdict(Counter)
    for f in frames:
        fl = [o for o in f if objs_keep is None or o in objs_keep]
        for a in fl:
            freq[a] += 1
        for i, a in enumerate(fl):
            for b in fl[i + 1:]:
                co[a][b] += 1
                co[b][a] += 1
    objs = [o for o, c in freq.items() if c >= 2]
    oi = {o: i for i, o in enumerate(objs)}
    S = np.zeros((len(objs), len(objs)))
    for a, cc in co.items():
        if a not in oi:
            continue
        for b, n in cc.items():
            if b in oi:
                S[oi[a], oi[b]] = n / np.sqrt(freq[a] * freq[b])
    np.fill_diagonal(S, 1.0)
    D = 1.0 - np.clip(S, 0, 1)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=k, criterion="maxclust")
    return objs, lab, freq


def name_clusters(objs, lab, freq, uid_name, rooms, model, top=12):
    """군집마다 대표 물체 이름을 모아 LLM 에게 '무슨 방이냐' 고 묻는다."""
    from kx.llm.room_namer import RoomNamer, _pretty
    namer = RoomNamer(rooms, model=model)
    out = {}
    for c in sorted(set(lab)):
        mem = [objs[i] for i in range(len(objs)) if lab[i] == c]
        mem.sort(key=lambda o: -freq[o])
        names = [_pretty(uid_name.get(o, "")) for o in mem[:top]]
        names = [n for n in names if n]
        if not names:
            continue
        # 군집 전체를 한 덩어리로 준다 — 개별 소품은 방을 못 정해도 조합은 정한다
        q = ", ".join(names)
        z, m = namer.label(q)
        out[c] = (z, m, len(mem), names[:6])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Apartment_release_decoration_seq137_M1292")
    ap.add_argument("--gt-root", default=os.path.join(ROOT, "data", "adt", "gt"))
    ap.add_argument("--seq-root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--llm-probe", default="/tmp/llm_dict_probe.json")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--mode", default="seed", choices=["seed", "cluster"])
    ap.add_argument("--k", type=int, default=4, help="군집 수")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="/tmp/bootstrap_dict.json")
    args = ap.parse_args()

    gd = os.path.join(args.gt_root, args.train)
    sd = os.path.join(args.seq_root, args.train)
    meta = json.load(open(os.path.join(sd, "graph_gtdepth.json")))["regions"]
    ref = load_regions(np.load(os.path.join(sd, "regions_gtdepth.npz")),
                       meta["zone_names"], meta["up"])
    rooms = meta["zone_names"]

    inst = json.load(open(os.path.join(gd, "instances.json")))
    name2uid, static = {}, set()
    for k, v in inst.items():
        name2uid.setdefault(v.get("instance_name") or "", int(k))
        if (v.get("motion_type") or "").lower() == "static":
            static.add(int(k))

    # --- 채점 기준: 기하로 만든 사전 (**비교용으로만**) --------------------------
    so = pd.read_csv(os.path.join(gd, "scene_objects.csv"),
                     usecols=["object_uid", "t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"])
    so = so.groupby("object_uid").first()
    truth = {}
    for uid, r in so.iterrows():
        if int(uid) not in static:
            continue
        z = assign(ref, np.array([r["t_wo_x[m]"], r["t_wo_y[m]"], r["t_wo_z[m]"]]))[1]
        if z:
            truth[int(uid)] = z

    probe = json.load(open(args.llm_probe))
    seeds = {}
    for r in probe:
        if r["margin"] >= args.margin and r["llm"]:
            u = name2uid.get(r["name"])
            if u is not None:
                seeds[u] = r["llm"]
    ok = sum(truth.get(u) == z for u, z in seeds.items())
    print("시드 %d개 (마진≥%.1f) · 분포 %s · 기하사전과 일치 %d/%d"
          % (len(seeds), args.margin, dict(Counter(seeds.values())), ok, len(seeds)))

    frames = co_visibility(gd)
    print("동시가시 프레임 %d · 관측 물체 %d"
          % (len(frames), len({o for f in frames for o in f})))

    if args.mode == "cluster":
        uid_name = {int(k): (v.get("instance_name") or "") for k, v in inst.items()}
        objs, cl, freq = cluster_rooms(frames, args.k, objs_keep=static)
        named = name_clusters(objs, cl, freq, uid_name, rooms, args.llm)
        print("군집 %d개:" % len(named))
        for c, (z, m, n, ex) in sorted(named.items()):
            gtd = Counter(truth.get(objs[i]) for i in range(len(objs))
                          if cl[i] == c and objs[i] in truth)
            print("  #%d (%3d개) → LLM **%-8s** (마진 %.2f) | 기하 분포 %s"
                  % (c, n, str(z), m, dict(gtd)))
            print("       예: %s" % ", ".join(ex))
        lab = {objs[i]: (named[cl[i]][0], named[cl[i]][1])
               for i in range(len(objs)) if cl[i] in named and named[cl[i]][0]}
    else:
        lab = {u: v for u, v in propagate(frames, seeds, rooms, args.iters, args.alpha).items()
               if u in static}
    hit = [(truth[u], z) for u, (z, _) in lab.items() if u in truth]
    acc = np.mean([a == b for a, b in hit]) if hit else 0
    base = max(Counter(a for a, _ in hit).values()) / len(hit) if hit else 0
    print("전파 후 사전 %d개 · 기하사전과 일치 **%.3f** (최빈 %.3f, 대상 %d)"
          % (len(lab), acc, base, len(hit)))
    print("  예측 분포 %s" % dict(Counter(z for z, _ in lab.values())))
    print("  기하 분포 %s" % dict(Counter(truth.values())))
    for th in [0.0, 0.05, 0.1, 0.2]:
        sub = [(truth[u], z) for u, (z, m) in lab.items() if u in truth and m >= th]
        if sub:
            print("    확신도≥%.2f : %3d개 · 일치 %.3f"
                  % (th, len(sub), np.mean([a == b for a, b in sub])))

    json.dump({str(u): z for u, (z, _) in lab.items()}, open(args.out, "w"))
    print("→ %s" % args.out)


if __name__ == "__main__":
    main()
