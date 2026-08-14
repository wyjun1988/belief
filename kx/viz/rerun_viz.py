"""rerun 시각화 — 시간축을 끌어 물체가 옮겨지는 순간을 눈으로 본다.

    $P -m kx.viz.rerun_viz --seq <name> --graph graph_gtdepth.json --save out.rrd

층:
    world/rooms   방 경계 (벽으로 갈린 위상적 방)
    world/zones   기능적 구역 (kitchen/living/dining/bedroom) — 바닥 색칠
    world/objects 물체 노드. 시간에 따라 위치가 바뀐다
    world/moved   이동한 물체만 굵게 + 이전 자리에서 새 자리로 화살표
    world/camera  관찰자 궤적
"""
import argparse
import json
import os

import numpy as np

ZONE_COLOR = {"kitchen": (240, 170, 60), "living": (70, 160, 240),
              "dining": (120, 200, 120), "bedroom": (200, 110, 200),
              "bathroom": (140, 200, 220), "office": (200, 200, 120)}


def main():
    import rerun as rr

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth.json")
    ap.add_argument("--save", default=None, help=".rrd 로 저장 (뷰어 없이)")
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    reg_p = os.path.join(seq_dir, "regions_%s.npz" % args.graph[6:-5])
    up = np.array(g["regions"]["up"])

    rr.init("khronos-%s" % g["sequence"], spawn=args.save is None)
    if args.save:
        rr.save(args.save)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # --- 구역 바닥 색칠 --------------------------------------------------------
    if os.path.exists(reg_p):
        z = np.load(reg_p)
        zones, lo, res = z["zones"], z["lo"], float(z["res"])
        names = g["regions"]["zone_names"]
        e1 = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
        e1 = e1 - np.dot(e1, up) * up
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        h = g["regions"]["floor_h"] + 0.02
        for zi, zn in enumerate(names):
            ii, jj = np.nonzero(zones == zi)
            if not len(ii):
                continue
            uv = np.stack([ii, jj], 1) * res + lo
            P = uv[:, :1] * e1 + uv[:, 1:2] * e2 + h * up
            rr.log("world/zones/%s" % zn,
                   rr.Points3D(P, colors=ZONE_COLOR.get(zn, (150, 150, 150)), radii=res * 0.6),
                   static=True)

    # --- 관찰자 궤적 -----------------------------------------------------------
    rr.log("world/camera/path", rr.LineStrips3D([poses[:, :3, 3]],
                                                colors=(255, 255, 255)), static=True)

    moved = {i: o for i, o in g["objects"].items()
             if len([p for p in o["placements"] if p.get("stable")]) > 1}

    # --- 시간축 ---------------------------------------------------------------
    for t in range(0, g["n_frames"], args.step):
        rr.set_time_sequence("frame", t)
        pts, cols, labels, radii = [], [], [], []
        for iid, o in g["objects"].items():
            cur = None
            for pl in o["placements"]:
                if pl["start_frame"] <= t <= pl["end_frame"]:
                    cur = pl
                    break
                if pl["end_frame"] < t:
                    cur = pl                       # 마지막으로 본 자리
            if cur is None:
                continue
            pts.append(cur["position"])
            is_moved = iid in moved
            cols.append((255, 80, 80) if is_moved else (170, 170, 170))
            radii.append(0.09 if is_moved else 0.035)
            labels.append(o["name"] if is_moved else "")
        if pts:
            rr.log("world/objects", rr.Points3D(np.array(pts), colors=cols,
                                                radii=radii, labels=labels))
        if t < len(poses):
            rr.log("world/camera/pose", rr.Points3D(poses[t:t + 1, :3, 3],
                                                    colors=(255, 255, 0), radii=0.06))
        # 이동 화살표 — 방금 감지된 변화
        arr_o, arr_v = [], []
        for o in moved.values():
            for c in o["changes"]:
                if 0 <= t - c["detected_at_frame"] < 40 and c["distance_m"] > 0.3:
                    arr_o.append(c["from"])
                    arr_v.append(np.array(c["to"]) - np.array(c["from"]))
        rr.log("world/moved", rr.Arrows3D(origins=np.array(arr_o), vectors=np.array(arr_v),
                                          colors=(255, 40, 40))
               if arr_o else rr.Clear(recursive=False))

    print("이동 물체 %d개, 프레임 %d" % (len(moved), g["n_frames"]))
    if args.save:
        print("→ %s" % args.save)


if __name__ == "__main__":
    main()
