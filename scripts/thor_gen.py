#!/usr/bin/env python3
"""**다중 방 엔드투엔드 데이터 생성** — ProcTHOR 주택에서 우리 시나리오를 직접 만든다.

    $P scripts/thor_gen.py --houses 30 --min-rooms 4 --out data/thor

### 왜 만드나

우리 핵심 시나리오("안 보는 사이에 누가 옮겼다")가 실촬영 데이터에 5~11% 밖에 없고(㉟),
3RScan 은 100% 지만 **스캔당 방이 하나**라 "어느 방" 질문이 퇴화한다.
시뮬레이터는 **이동을 우리가 통제**하므로 세 상태를 의도적으로 만들 수 있다 —
특히 지금까지 **한 건도 못 만든 (b) 상태**(재방문 안 한 장소)를.

### 생성 절차

    1세션  에이전트가 **전 방**을 돌며 RGB 기록      → 물체 위치·방 GT 완비
           ↓ 물체 K 개를 **다른 방**으로 이동 (에이전트 없는 사이 = 미관측 100%)
    2세션  **일부 방만** 재방문                      → (b) 상태를 의도적으로 생성

GT 로 모든 것을 안다: 각 물체가 언제 어느 방에 있었는지, 무엇이 옮겨졌는지,
어느 방을 다시 봤는지.
"""
import argparse, json, os, random

import numpy as np


def room_of(pt, rooms):
    """(x,z) 가 어느 방 폴리곤 안인가 — ray casting."""
    x, z = pt
    for r in rooms:
        poly = [(p["x"], p["z"]) for p in r["floorPolygon"]]
        inside = False
        n = len(poly)
        for i in range(n):
            x1, z1 = poly[i]; x2, z2 = poly[(i + 1) % n]
            if (z1 > z) != (z2 > z):
                xin = (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1
                if x < xin:
                    inside = not inside
        if inside:
            return r["id"], r["roomType"]
    return None, None


def views(ctrl, rooms, per_room, rng):
    """방마다 서 볼 자리 — 도달 가능 위치를 방별로 나눠 균등 표본."""
    ev = ctrl.step("GetReachablePositions")
    pos = ev.metadata["actionReturn"] or []
    byroom = {}
    for p in pos:
        rid, _ = room_of((p["x"], p["z"]), rooms)
        if rid:
            byroom.setdefault(rid, []).append(p)
    out = {}
    for rid, ps in byroom.items():
        if len(ps) < 2:
            continue
        idx = np.linspace(0, len(ps) - 1, min(per_room, len(ps))).astype(int)
        out[rid] = [ps[i] for i in idx]
    return out


def capture(ctrl, spots, yaws):
    """자리마다 여러 방향을 보며 프레임과 가시 물체를 모은다."""
    frames, meta = [], []
    for rid, ps in spots.items():
        for p in ps:
            for y in yaws:
                e = ctrl.step("Teleport", position=p, rotation=dict(x=0, y=y, z=0), horizon=10)
                if not e.metadata["lastActionSuccess"]:
                    continue
                frames.append(e.frame)
                meta.append(dict(room=rid, yaw=y,
                                 visible=[o["objectId"] for o in e.metadata["objects"]
                                          if o["visible"]]))
    return frames, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, default=30)
    ap.add_argument("--min-rooms", type=int, default=4)
    ap.add_argument("--per-room", type=int, default=3, help="방당 서 볼 자리")
    ap.add_argument("--yaws", type=int, default=4, help="자리당 방향 수")
    ap.add_argument("--move", type=int, default=6, help="옮길 물체 수")
    ap.add_argument("--revisit", type=float, default=0.6, help="2세션에 다시 볼 방 비율")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    import prior
    from ai2thor.controller import Controller
    from PIL import Image
    ds = prior.load_dataset("procthor-10k")["train"]
    yaws = [int(360 * i / args.yaws) for i in range(args.yaws)]

    ctrl = Controller(scene=ds[0], width=args.size, height=args.size,
                      quality="Low", renderInstanceSegmentation=False)
    made = 0
    for hi in range(len(ds)):
        if made >= args.houses:
            break
        h = ds[hi]
        if len(h["rooms"]) < args.min_rooms:
            continue
        try:
            ctrl.reset(scene=h)
        except Exception as e:
            print("  주택 %d 로드 실패 %s" % (hi, str(e)[:80]), flush=True); continue
        rooms = h["rooms"]
        spots = views(ctrl, rooms, args.per_room, rng)
        if len(spots) < args.min_rooms:
            continue

        # ── 1세션: 전 방
        f1, m1 = capture(ctrl, spots, yaws)
        ev = ctrl.step("Pass")
        objs = {o["objectId"]: o for o in ev.metadata["objects"]}
        gt1 = {oid: dict(type=o["objectType"],
                         room=room_of((o["position"]["x"], o["position"]["z"]), rooms)[0],
                         pos=o["position"])
               for oid, o in objs.items() if o.get("pickupable")}

        # ── 물체 이동 (에이전트 없는 사이 = 미관측)
        movable = [oid for oid, g in gt1.items() if g["room"]]
        rng.shuffle(movable)
        moved = {}
        for oid in movable:
            if len(moved) >= args.move:
                break
            here = gt1[oid]["room"]
            others = [rid for rid in spots if rid != here]
            if not others:
                continue
            tgt = rng.choice(others)
            p = rng.choice(spots[tgt])
            e = ctrl.step("PlaceObjectAtPoint", objectId=oid,
                          position=dict(x=p["x"], y=p["y"] + 0.6, z=p["z"]))
            if e.metadata["lastActionSuccess"]:
                moved[oid] = dict(frm=here, to=tgt)
        if len(moved) < 2:
            print("  주택 %d 이동 실패 — 건너뜀" % hi, flush=True); continue

        # ── 2세션: **일부 방만** 재방문 → (b) 상태
        rids = sorted(spots)
        k = max(1, int(len(rids) * args.revisit))
        seen = set(rng.sample(rids, k))
        f2, m2 = capture(ctrl, {r: spots[r] for r in seen}, yaws)
        ev = ctrl.step("Pass")
        objs2 = {o["objectId"]: o for o in ev.metadata["objects"]}
        gt2 = {oid: dict(room=room_of((o["position"]["x"], o["position"]["z"]), rooms)[0],
                         pos=o["position"])
               for oid, o in objs2.items() if oid in gt1}

        hd = os.path.join(args.out, "house_%04d" % hi)
        for tag, fr in (("s1", f1), ("s2", f2)):
            os.makedirs(os.path.join(hd, tag), exist_ok=True)
            for i, im in enumerate(fr):
                Image.fromarray(im).save(os.path.join(hd, tag, "%04d.jpg" % i), quality=88)
        json.dump(dict(house=hi, rooms=[dict(id=r["id"], type=r["roomType"]) for r in rooms],
                       spots={k2: len(v) for k2, v in spots.items()},
                       revisited=sorted(seen), moved=moved,
                       gt1=gt1, gt2=gt2, m1=m1, m2=m2),
                  open(os.path.join(hd, "gt.json"), "w"))
        made += 1
        print("  주택 %d · 방 %d · 재방문 %d · 이동 %d · 프레임 %d+%d"
              % (hi, len(spots), len(seen), len(moved), len(f1), len(f2)), flush=True)
    ctrl.stop()
    print("생성 완료 %d채 → %s" % (made, args.out))


if __name__ == "__main__":
    main()
