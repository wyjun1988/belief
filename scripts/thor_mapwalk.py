#!/usr/bin/env python3
"""**매핑 워크** — 초기 맵을 만들기 위한 촬영 이동을 실제로 걷는다.

    python scripts/thor_mapwalk.py --root data/thor3 [--gt-depth]

⚠️ **왜 필요한가.** 기존 맵 프레임은 방마다 3지점 × 4방향 **순간이동 정지샷**이라
프레임 간 연속성이 0 이다. mono-depth 파이프라인(DA3 정합·SfM)은 사람이 걸으며
찍은 **연속 영상의 시차**를 전제한다 — 정지샷에는 넣을 수조차 없다.

여기서는 사람이 새 집을 둘러보듯 걷는다:
  · 문 연결(BFS) 순서로 방을 차례로 방문
  · 방 안에서는 도달가능 격자(0.25 m)를 최근접 이웃 순회 — 걸음마다 프레임
  · 진행 방향으로 yaw 를 두고, 방마다 몇 지점에서 360° 스캔(45° × 8)
  · 기록: RGB + **포즈(pos, yaw)** + (--gt-depth 면) GT 뎁스 + bbox

포즈는 저장한다 — 설계상 초기 맵은 무겁게 만들 수 있다(pose·depth 허용).
실행(1fps 배회)에는 여전히 RGB 만 쓴다. GT 뎁스는 상한 대조용이고,
실전 눈금은 DA3/mono-depth 로 같은 프레임을 다시 푼 것이다.
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image



def walk(ctrl, out, polys, doors, size, grid_skip=2, scan_every=6, gt_depth=False):
    """한 집에서 매핑 워크를 실행해 out/ 에 프레임·포즈·(뎁스)·bbox 를 남긴다.

    ⚠️ `thor_gen2 --mapwalk` 가 **t=0 배치 직후** 이 함수를 부른다. 독립 스크립트로
    house 를 reset 하면 이동 물체가 기본 위치로 돌아가 타겟 초기 방을 검출로 못
    만든다 — 배치 → 매핑워크 → 배회 순서일 때만 타겟이 검출 맵에 들어온다."""
    import numpy as np
    from PIL import Image

    def _in(x, z, pts):
        c = False; n = len(pts)
        for i in range(n):
            x1, z1 = pts[i]; x2, z2 = pts[(i + 1) % n]
            if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12) + x1:
                c = not c
        return c

    pos = ctrl.step("GetReachablePositions").metadata["actionReturn"] or []
    byroom = {}
    for p in pos:
        for r, pts in polys.items():
            if _in(p["x"], p["z"], pts):
                byroom.setdefault(r, []).append(p); break
    adjr = {}
    for a, b in doors:
        adjr.setdefault(a, set()).add(b); adjr.setdefault(b, set()).add(a)
    rids = [r for r in sorted(byroom) if byroom[r]]
    if not rids:
        return 0
    order, seen, q = [], set(), [rids[0]]
    while q:
        r = q.pop(0)
        if r in seen or r not in byroom:
            continue
        seen.add(r); order.append(r)
        q += sorted(adjr.get(r, ()))
    order += [r for r in rids if r not in seen]
    os.makedirs(out, exist_ok=True)
    if gt_depth:
        os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    rec = []; state = dict(k=0, cur=None)

    def shoot(p, yaw, room, scan):
        e = ctrl.step("Teleport", position=p, rotation=dict(x=0, y=yaw, z=0), horizon=10)
        if not e.metadata["lastActionSuccess"]:
            return
        k = state["k"]
        Image.fromarray(e.frame).save(os.path.join(out, "%05d.jpg" % k), quality=88)
        if gt_depth and e.depth_frame is not None:
            np.save(os.path.join(out, "depth", "%05d.npy" % k),
                    e.depth_frame.astype(np.float16))
        bx = e.instance_detections2D
        rec.append(dict(k=k, room=room, scan=scan,
                        pos=[round(p["x"], 3), round(p["z"], 3)], yaw=int(yaw),
                        box={o["objectId"]: [int(x) for x in bx[o["objectId"]]]
                             for o in e.metadata["objects"]
                             if o.get("visible") and bx.get(o["objectId"]) is not None}))
        state["k"] = k + 1; state["cur"] = p

    for r in order:
        pts = byroom[r][::grid_skip] or byroom[r]
        left = pts[:]; here = state["cur"] or left[0]
        tour = []
        while left:
            left.sort(key=lambda p: (p["x"]-here["x"])**2 + (p["z"]-here["z"])**2)
            here = left.pop(0); tour.append(here)
        for i, p in enumerate(tour):
            nxt = tour[i+1] if i+1 < len(tour) else p
            yaw = float(np.degrees(np.arctan2(nxt["x"]-p["x"], nxt["z"]-p["z"]))) % 360
            shoot(p, round(yaw), r, False)
            if i % scan_every == 0:
                for dy in range(45, 360, 45):
                    shoot(p, round((yaw+dy) % 360), r, True)
    json.dump(dict(frames=rec, fov=90.0, size=size, order=order),
              open(os.path.join(out, "walk.json"), "w"))
    return state["k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--grid-skip", type=int, default=2, help="격자 몇 칸마다 서나")
    ap.add_argument("--scan-every", type=int, default=6, help="몇 걸음마다 360° 스캔")
    ap.add_argument("--gt-depth", action="store_true")
    ap.add_argument("--platform", default=None, choices=[None, "cloud"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import prior
    from ai2thor.controller import Controller
    ds = prior.load_dataset("procthor-10k")["train"]
    kw = {}
    if args.platform == "cloud":
        from ai2thor.platform import CloudRendering
        kw["platform"] = CloudRendering

    def inside(x, z, pts):
        c = False; n = len(pts)
        for i in range(n):
            x1, z1 = pts[i]; x2, z2 = pts[(i + 1) % n]
            if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12) + x1:
                c = not c
        return c

    ctrl = None
    for hd in sorted(glob.glob(os.path.join(args.root, "house_*"))):
        rd = os.path.realpath(hd)
        out = os.path.join(rd, "mapwalk")
        if os.path.exists(os.path.join(out, "walk.json")) and not args.force:
            continue
        g = json.load(open(os.path.join(rd, "gt.json")))
        sm = g.get("scene_meta")
        if not sm:
            print("  %s scene_meta 없음" % os.path.basename(hd), flush=True); continue
        h = ds[g["house"]]
        polys = {r: [(c[0], c[1]) for c in v] for r, v in sm["polys"].items()}
        ctrl = (Controller(scene=h, width=args.size, height=args.size, quality="Low",
                           visibilityDistance=20.0, renderInstanceSegmentation=True,
                           renderDepthImage=args.gt_depth, **kw)
                if ctrl is None else ctrl)
        ctrl.reset(scene=h,
                   renderDepthImage=args.gt_depth, renderInstanceSegmentation=True)
        pos = ctrl.step("GetReachablePositions").metadata["actionReturn"] or []
        byroom = {}
        for p in pos:
            for r, pts in polys.items():
                if inside(p["x"], p["z"], pts):
                    byroom.setdefault(r, []).append(p); break
        # 방 방문 순서: 문 그래프 BFS (외딴 방은 뒤에 붙인다)
        adjr = {}
        for a, b in sm["doors"]:
            adjr.setdefault(a, set()).add(b); adjr.setdefault(b, set()).add(a)
        rids = [r for r in sorted(byroom) if byroom[r]]
        order, seen, q = [], set(), [rids[0]]
        while q:
            r = q.pop(0)
            if r in seen or r not in byroom: continue
            seen.add(r); order.append(r)
            q += sorted(adjr.get(r, ()))
        order += [r for r in rids if r not in seen]
        os.makedirs(out, exist_ok=True)
        if args.gt_depth:
            os.makedirs(os.path.join(out, "depth"), exist_ok=True)
        rec = []; k = 0; cur = None

        def shoot(p, yaw, room, scan):
            nonlocal k, cur
            e = ctrl.step("Teleport", position=p, rotation=dict(x=0, y=yaw, z=0), horizon=10)
            if not e.metadata["lastActionSuccess"]:
                return
            Image.fromarray(e.frame).save(os.path.join(out, "%05d.jpg" % k), quality=88)
            if args.gt_depth and e.depth_frame is not None:
                np.save(os.path.join(out, "depth", "%05d.npy" % k),
                        e.depth_frame.astype(np.float16))
            bx = e.instance_detections2D
            rec.append(dict(k=k, room=room, scan=scan,
                            pos=[round(p["x"], 3), round(p["z"], 3)], yaw=int(yaw),
                            box={o["objectId"]: [int(x) for x in bx[o["objectId"]]]
                                 for o in e.metadata["objects"]
                                 if o.get("visible") and bx.get(o["objectId"]) is not None}))
            k += 1; cur = p

        for r in order:
            pts = byroom[r][::args.grid_skip] or byroom[r]
            # 최근접 이웃 순회
            left = pts[:]; here = cur or left[0]
            tour = []
            while left:
                left.sort(key=lambda p: (p["x"]-here["x"])**2 + (p["z"]-here["z"])**2)
                here = left.pop(0); tour.append(here)
            for i, p in enumerate(tour):
                nxt = tour[i+1] if i+1 < len(tour) else p
                yaw = float(np.degrees(np.arctan2(nxt["x"]-p["x"], nxt["z"]-p["z"]))) % 360
                shoot(p, round(yaw), r, False)
                if i % args.scan_every == 0:
                    for dy in range(45, 360, 45):
                        shoot(p, round((yaw+dy) % 360), r, True)
        json.dump(dict(frames=rec, fov=90.0, size=args.size, order=order),
                  open(os.path.join(out, "walk.json"), "w"))
        print("  %s · 프레임 %d · 방 %d" % (os.path.basename(hd), k, len(order)), flush=True)
    print("완료")


if __name__ == "__main__":
    main()
