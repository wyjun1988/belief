"""3D 계층 씬그래프 그림 — 바닥 평면 위로 물체·구역·건물 층이 쌓이는 전형적 DSG 도해.

    $P kx/viz/dsg_3d.py --seq <name> --graph graph_gtdepth --highlight PictureFrame

층 (아래에서 위로):
    z=0   바닥 — 구역 지도(색)와 관찰자 궤적
    z=1   물체 노드 — 바닥의 제 위치에서 수직선으로 올라온다
    z=2   구역 노드 — 그 구역 물체들의 중심
    z=3   건물 노드

강조 물체는 배치 이력을 시간순 화살표로 잇는다(액자: 방 → 거실 탁자 → 거실 선반).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Z_OBJ, Z_ZONE, Z_BLD = 1.6, 3.2, 5.0
COL = {"kitchen": "#f0aa3c", "living": "#4aa0f0", "dining": "#78c878",
       "bedroom": "#c86ec8", "bathroom": "#8ecfe0", "office": "#c8c878"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--highlight", default="BlackSquarePictureFrame")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "dsg_3d.png"))
    ap.add_argument("--max-objects", type=int, default=110)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph + ".json")))
    tag = args.graph.replace("graph_", "")
    z = np.load(os.path.join(seq_dir, "regions_%s.npz" % tag))
    zones, lo, res = z["zones"], z["lo"], float(z["res"])
    names = g["regions"]["zone_names"]
    # ⚠️ 구역 래스터는 **중력정렬 바닥좌표 (u,v)** 다 — world 의 (x,z) 가 아니다.
    # 이 시퀀스는 e2 = -z 라서 그냥 (x,z) 로 찍으면 바닥 지도가 z축으로 뒤집힌다.
    from kx.graph.frames import floor_basis
    E1, E2, _ = floor_basis(np.array(g["regions"]["up"], float))
    poses = np.loadtxt(os.path.join(seq_dir, g.get("pose_file", "pose/poses.txt"))).reshape(-1, 4, 4)

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    # --- z=0 바닥: 구역 지도 -----------------------------------------------------
    for zi, zn in enumerate(names):
        ii, jj = np.nonzero(zones == zi)
        if not len(ii):
            continue
        sub = slice(None, None, max(len(ii) // 3000, 1))
        uv = np.stack([ii[sub], jj[sub]], 1) * res + lo
        W = uv[:, :1] * E1[None, :] + uv[:, 1:2] * E2[None, :]
        ax.scatter(W[:, 0], W[:, 2], np.zeros(len(uv)), s=3, c=COL.get(zn, "#999"),
                   alpha=0.35, linewidths=0, depthshade=False)
    t = poses[:, :3, 3]
    ax.plot(t[:, 0], t[:, 2], np.zeros(len(t)), lw=0.8, color="0.25", alpha=0.8)

    # --- z=1 물체 노드 -----------------------------------------------------------
    objs = [o for o in g["objects"].values() if o.get("placements")]
    objs.sort(key=lambda o: -o["n_obs"])
    objs = objs[:args.max_objects]
    zone_pts = {}
    hl = None
    for o in objs:
        pl = o["placements"][-1]
        p = np.array(pl["position"], float)
        zn = pl.get("zone")
        c = COL.get(zn, "#aaa")
        x, y = p[0], p[2]
        ax.plot([x, x], [y, y], [0, Z_OBJ], color=c, lw=0.5, alpha=0.45)
        ax.scatter([x], [y], [Z_OBJ], s=26, c=c, edgecolors="k", linewidths=0.3,
                   depthshade=False, zorder=5)
        zone_pts.setdefault(zn, []).append((x, y))
        nm = (o.get("name") or "").lower()
        if args.highlight.lower() == nm or (hl is None and args.highlight.lower() in nm):
            if len([p for p in o["placements"] if p.get("stable")]) > 1 or hl is None:
                hl = o

    # --- z=2 구역 노드, z=3 건물 --------------------------------------------------
    zc = {}
    for zn, pts in zone_pts.items():
        if zn is None:
            continue
        P = np.array(pts)
        cx, cy = P[:, 0].mean(), P[:, 1].mean()
        zc[zn] = (cx, cy)
        for x, y in pts:
            ax.plot([x, cx], [y, cy], [Z_OBJ, Z_ZONE], color=COL.get(zn, "#aaa"),
                    lw=0.35, alpha=0.30)
        ax.scatter([cx], [cy], [Z_ZONE], s=340, c=COL.get(zn, "#aaa"),
                   edgecolors="k", linewidths=1.2, depthshade=False, zorder=6)
        ax.text(cx, cy, Z_ZONE + 0.28, zn, ha="center", fontsize=12, weight="bold")
    if zc:
        bx = np.mean([v[0] for v in zc.values()])
        by = np.mean([v[1] for v in zc.values()])
        for cx, cy in zc.values():
            ax.plot([cx, bx], [cy, by], [Z_ZONE, Z_BLD], color="0.35", lw=0.9, alpha=0.7)
        ax.scatter([bx], [by], [Z_BLD], s=520, c="0.75", edgecolors="k",
                   linewidths=1.5, depthshade=False, zorder=7)
        ax.text(bx, by, Z_BLD + 0.3, "apartment", ha="center", fontsize=13, weight="bold")

    # --- 강조 물체의 배치 이력 ---------------------------------------------------
    if hl is not None:
        st = [p for p in hl["placements"] if p.get("stable")] or hl["placements"]
        P = np.array([p["position"] for p in st], float)
        ax.plot(P[:, 0], P[:, 2], np.full(len(P), Z_OBJ), "-", color="crimson",
                lw=2.6, zorder=9)
        for k, (p, pl) in enumerate(zip(P, st)):
            ax.scatter([p[0]], [p[2]], [Z_OBJ], s=190, c="crimson", edgecolors="k",
                       linewidths=1.2, depthshade=False, zorder=10)
            ax.text(p[0], p[2], Z_OBJ + 0.22,
                    "%d. %s\n(%s)" % (k + 1, (pl.get("support") or "?")[:18], pl.get("zone")),
                    ha="center", fontsize=9, weight="bold", color="darkred")
        for a, b in zip(P[:-1], P[1:]):
            ax.quiver(a[0], a[2], Z_OBJ, (b - a)[0], (b - a)[2], 0, color="crimson",
                      arrow_length_ratio=0.12, lw=2.0, zorder=9)

    # 물체가 모인 곳으로 축을 좁힌다 (빈 바닥이 그림을 잡아먹지 않게)
    XY = np.array([[o["placements"][-1]["position"][0], o["placements"][-1]["position"][2]]
                   for o in objs])
    ax.set_xlim(np.percentile(XY[:, 0], 1) - 1, np.percentile(XY[:, 0], 99) + 1)
    ax.set_ylim(np.percentile(XY[:, 1], 1) - 1, np.percentile(XY[:, 1], 99) + 1)
    ax.set_zlim(0, Z_BLD + 0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_zticks([0, Z_OBJ, Z_ZONE, Z_BLD])
    ax.set_zticklabels(["바닥/구역지도", "물체", "구역", "건물"], fontsize=11)
    ax.view_init(elev=26, azim=-58)
    ax.set_title("4D 씬그래프 — %s\n%s" % (g["sequence"][18:], args.graph), fontsize=13)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[n], label=n)
                       for n in names if n in COL]
              + [Line2D([], [], color="crimson", lw=2.5, marker="o",
                        label=(hl.get("name") if hl else "강조") + " 이동")],
              loc="upper left", fontsize=10)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print("→ %s" % args.out)


if __name__ == "__main__":
    main()
