"""씬그래프 → 웹 뷰어용 scene.json (노드·구역·강조 궤적·관찰자 경로).

    $P kx/viz/export_scene.py --seq <name> --graph graph_gtdepth --highlight BlackSquarePictureFrame

좌표는 **월드 그대로** 내보낸다(Y 위). 뷰어가 점구름과 같은 좌표계에서 겹쳐 그린다.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COL = {"kitchen": "#e0a03c", "living": "#4aa0f0", "dining": "#6cc47a",
       "bedroom": "#c86ec8", "bathroom": "#8ecfe0", "office": "#c8c878"}
KO = {"kitchen": "부엌", "living": "거실", "dining": "다이닝", "bedroom": "침실",
      "bathroom": "욕실", "office": "서재"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--highlight", default="BlackSquarePictureFrame")
    ap.add_argument("--max-objects", type=int, default=220)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph + ".json")))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    poses = np.loadtxt(os.path.join(seq_dir, g.get("pose_file", "pose/poses.txt"))).reshape(-1, 4, 4)

    objs = [dict(o, _iid=k) for k, o in g["objects"].items() if o.get("placements")]
    objs.sort(key=lambda o: -o["n_obs"])
    nodes = []
    for o in objs[:args.max_objects]:
        pl = o["placements"][-1]
        dyn = (o.get("gt_motion_type") or "").lower() == "dynamic"
        nodes.append(dict(name=o.get("name") or "?", cat=o.get("category") or "",
                          pos=[round(float(x), 3) for x in pl["position"]],
                          zone=pl.get("zone"), n_obs=o["n_obs"], dyn=dyn,
                          support=pl.get("support")))

    by_name = {}
    for o in objs:
        if o.get("name"):
            by_name.setdefault(o["name"], o["placements"][-1]["position"])

    zones = {}
    for n in nodes:
        if n["zone"]:
            zones.setdefault(n["zone"], []).append(n["pos"])
    zone_list = [dict(name=z, ko=KO.get(z, z), color=COL.get(z, "#999"),
                      centroid=[round(float(v), 3) for v in np.array(p).mean(0)],
                      n=len(p)) for z, p in zones.items()]

    hl = None
    for o in objs:
        if args.highlight.lower() in (o.get("name") or "").lower():
            st = [p for p in o["placements"] if p.get("stable")]
            if len(st) > 1 or hl is None:
                hl = (o, st)
    if hl:
        o, st = hl
        gi = str(o.get("gt_instance") or o["_iid"])
        rec = gt.get(gi)
        # GT 궤적도 같이 — "그래프가 잰 위치"와 "실제 위치"를 눈으로 대조할 수 있게
        gtp, track = None, None
        if rec:
            P = np.array(rec["positions"])
            gtp = [[round(float(v), 3) for v in P[f]] for f in range(0, len(P), 3)]
            # 시간 스크럽용: 프레임마다 GT 위치 + 그 시점 그래프가 붙인 지지면
            track = []
            for f in range(0, len(P), 3):
                pl = next((q for q in o["placements"]
                           if q["start_frame"] <= f <= q["end_frame"]), None)
                track.append(dict(t=round(f / 10.0, 1),
                                  gt=[round(float(v), 3) for v in P[f]],
                                  sg=[round(float(v), 3) for v in pl["position"]] if pl else None,
                                  sup=pl.get("support") if pl else None,
                                  zone=pl.get("zone") if pl else None,
                                  stable=bool(pl.get("stable")) if pl else False))
        hl = dict(name=o.get("name"), cat=o.get("category"),
                  gt_path=gtp, track=track,
                  sup_pos={k: [round(float(v), 3) for v in by_name[k]]
                           for k in {q.get("support") for q in o["placements"]} if k in by_name},
                  stops=[dict(pos=[round(float(v), 3) for v in p["position"]],
                              support=p.get("support"), zone=p.get("zone"),
                              t0=round(p["start_frame"] / 10.0, 1),
                              t1=round(p["end_frame"] / 10.0, 1),
                              n_obs=p["n_obs"]) for p in st])

    # 구역 래스터 — 뷰어가 점구름을 방 색으로 은은하게 물들이는 데 쓴다
    from kx.graph.frames import floor_basis
    E1, E2, UP = floor_basis(np.array(g["regions"]["up"], float))
    zt = args.graph.replace("graph_", "")
    zr = np.load(os.path.join(seq_dir, "regions_%s.npz" % zt))
    zg = zr["zones"].astype(np.int32)
    names = g["regions"]["zone_names"]
    idx = {n: i for i, n in enumerate([z["name"] for z in zone_list])}
    remap = np.full(len(names), 255, np.uint8)
    for i, n in enumerate(names):
        if n in idx:
            remap[i] = idx[n]
    grid = np.where(zg >= 0, remap[np.clip(zg, 0, len(names) - 1)], 255).astype(np.uint8)
    # 자유공간 래스터는 **벽을 뺀** 것이라 그대로 클리핑하면 벽이 통째로 사라진다.
    # 인접 구역으로 몇 셀 부풀려서 "집 안" 마스크로 쓴다 (벽·가구 표면도 살아남는다).
    for _ in range(int(round(0.55 / float(zr["res"])))):
        g2 = grid.copy()
        for ax in (0, 1):
            for sh in (1, -1):
                nb = np.roll(grid, sh, axis=ax)
                m = (g2 == 255) & (nb != 255)
                g2[m] = nb[m]
        grid = g2

    # --- 바닥 평면 도면: 집 실루엣 + 방 사이 벽 ---------------------------------
    # 자유공간 래스터를 그대로 윤곽 따면 가구가 만든 구멍이 전부 선이 되어 지저분하다.
    # 바깥에서 채우기(flood fill)로 내부 구멍을 메운 뒤 윤곽을 딴다.
    rm = zr["rooms"].astype(np.int32)
    reach = zr["reach"].astype(bool) if "reach" in zr.files else (zg >= 0)
    inside = reach | (rm > 0) | (zg >= 0)
    H_, W_ = inside.shape
    out = np.zeros_like(inside)
    seeds = [(i, j) for i in (0, H_ - 1) for j in range(W_) if not inside[i, j]]
    seeds += [(i, j) for j in (0, W_ - 1) for i in range(H_) if not inside[i, j]]
    for i, j in seeds:
        out[i, j] = True
    stack = list(seeds)
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < H_ and 0 <= b < W_ and not inside[a, b] and not out[a, b]:
                out[a, b] = True
                stack.append((a, b))
    inside = ~out

    lo2, res2 = zr["lo"], float(zr["res"])
    # 구멍을 메운 자리는 방 라벨이 0 이라 그대로 비교하면 방 안에 가짜 경계가 생긴다.
    # (a) 집 바깥 경계 (b) **둘 다 방 라벨이 있는** 서로 다른 방 사이만 선으로 뽑는다.

    def w(u, v):
        return (u * E1[0] + v * E2[0], u * E1[2] + v * E2[2])

    segs = []
    for ax in (0, 1):
        ia = (inside[:-1, :], inside[1:, :]) if ax == 0 else (inside[:, :-1], inside[:, 1:])
        ra = (rm[:-1, :], rm[1:, :]) if ax == 0 else (rm[:, :-1], rm[:, 1:])
        edge = (ia[0] != ia[1]) | (ia[0] & ia[1] & (ra[0] > 0) & (ra[1] > 0) & (ra[0] != ra[1]))
        ii, jj = np.nonzero(edge)
        for i, j in zip(ii, jj):
            u0 = lo2[0] + (i + 1) * res2 if ax == 0 else lo2[0] + i * res2
            v0 = lo2[1] + j * res2 if ax == 0 else lo2[1] + (j + 1) * res2
            p0, p1 = (w(u0, v0), w(u0, v0 + res2)) if ax == 0 else (w(u0, v0), w(u0 + res2, v0))
            segs.append([round(p0[0], 3), round(p0[1], 3), round(p1[0], 3), round(p1[1], 3)])

    oi, oj = np.nonzero(inside)
    XX = (lo2[0] + oi * res2) * E1[0] + (lo2[1] + oj * res2) * E2[0]
    ZZ = (lo2[0] + oi * res2) * E1[2] + (lo2[1] + oj * res2) * E2[2]
    floor_box = dict(lo=[float(XX.min()), float(ZZ.min())],
                     hi=[float(XX.max()), float(ZZ.max())])

    t = poses[::3, :3, 3]
    P = np.array([n["pos"] for n in nodes])
    out = dict(sequence=g["sequence"], graph=args.graph, n_frames=g["n_frames"],
               nodes=nodes, zones=zone_list, highlight=hl,
               trail=[[round(float(v), 3) for v in p] for p in t],
               bounds=dict(lo=P.min(0).round(2).tolist(), hi=P.max(0).round(2).tolist()),
               floor_y=float(np.percentile(P[:, 1], 1)),
               floor_outline=segs, floor_box=floor_box,
               zone_grid=dict(w=int(grid.shape[0]), h=int(grid.shape[1]),
                              lo=[float(zr["lo"][0]), float(zr["lo"][1])],
                              res=float(zr["res"]),
                              e1=[float(v) for v in E1], e2=[float(v) for v in E2],
                              data=base64.b64encode(grid.tobytes()).decode()))
    p = os.path.join(seq_dir, "cloud", "scene.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    print("노드 %d · 구역 %d · 강조 %s(%d정지) → %s"
          % (len(nodes), len(zone_list), hl and hl["name"], hl and len(hl["stops"]), p))


if __name__ == "__main__":
    main()
