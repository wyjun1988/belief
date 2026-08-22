#!/usr/bin/env python3
"""**2차 — 1~2시간 · 1fps · pose/depth 없음.** 초기 맵만 만들고 이후는 RGB 뿐.

    $P scripts/thor_gen2.py --houses 8 --hours 2 --out data/thor2

### 1차와 무엇이 다른가

1차는 "두 세션(전 방 → 일부 방)" 이라 시간이 압축돼 있었다. 2차는 **실사용 조건**이다:

    t=0        **초기 맵 형성** — 전 방을 돌며 씬그래프를 만든다.
               이때만 pose·방 폴리곤을 쓴다(설치 시 1회).
    t>0        **1fps 로 1~2시간** 배회. **RGB 만.** pose·depth 없음.
               물체는 에이전트가 **다른 방에 있을 때** 옮겨진다(미관측 100%).
    질의       아무 시각에 "X 어디 있어?"

`--stride` 로 **무엇을 남길지**도 실험한다 — 1fps 전부 저장하면 2시간에 7,200장이다.
30초마다 1장이면 240장. 저장·계산 정책의 실측 근거를 만든다.

⚠️ 배회는 **방 단위 체류**로 만든다(무작위 순간이동이 아니라). 실제로는 한 방에
몇 분 머물다 옮기므로, 그래야 "그 방을 안 봤다"(b 상태)가 자연스럽게 생긴다.
"""
import argparse, json, os, random

import numpy as np


def room_of(pt, rooms):
    x, z = pt
    for r in rooms:
        poly = [(p["x"], p["z"]) for p in r["floorPolygon"]]
        inside = False; n = len(poly)
        for i in range(n):
            x1, z1 = poly[i]; x2, z2 = poly[(i + 1) % n]
            if (z1 > z) != (z2 > z):
                if x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1:
                    inside = not inside
        if inside:
            return r["id"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, default=8)
    ap.add_argument("--min-rooms", type=int, default=4)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dwell", type=int, default=90, help="한 방 체류 초(평균)")
    ap.add_argument("--moves", type=int, default=10, help="세션 중 이동 사건 수")
    ap.add_argument("--map-per-room", type=int, default=3)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", default=None,
                    help="**Qwen 이 만든 배치 분포**(scripts/thor_prior_llm.py). 주면 t=0 에 "
                         "물체를 그 분포에서 뽑은 방으로 재배치한다. "
                         "⚠️ ProcTHOR 기본 배치는 유형 규칙이라 사전확률이 실제보다 강하고, "
                         "그 때문에 ㊻ 에서 belief 단독이 전체 시스템을 이겼다. 편향 제거용.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    T = int(args.hours * 3600 * args.fps)

    PRIOR = json.load(open(args.prior)) if args.prior else None
    import prior
    from ai2thor.controller import Controller
    from PIL import Image
    ds = prior.load_dataset("procthor-10k")["train"]
    ctrl = Controller(scene=ds[0], width=args.size, height=args.size, quality="Low")
    made = 0
    for hi in range(len(ds)):
        if made >= args.houses:
            break
        h = ds[hi]
        if len(h["rooms"]) < args.min_rooms:
            continue
        try:
            ctrl.reset(scene=h)
        except Exception:
            continue
        rooms = h["rooms"]
        ev = ctrl.step("GetReachablePositions")
        pos = ev.metadata["actionReturn"] or []
        byroom = {}
        for p in pos:
            rid = room_of((p["x"], p["z"]), rooms)
            if rid:
                byroom.setdefault(rid, []).append(p)
        byroom = {k: v for k, v in byroom.items() if len(v) >= 3}
        if len(byroom) < args.min_rooms:
            continue
        hd = os.path.join(args.out, "house_%04d" % hi)
        os.makedirs(os.path.join(hd, "map"), exist_ok=True)
        os.makedirs(os.path.join(hd, "live"), exist_ok=True)

        rt = {r["id"]: r["roomType"] for r in rooms}
        # ⚠️ 초기 맵은 **재배치 뒤**에 찍어야 한다 — 맵이 곧 씬그래프이므로
        # 재배치 전 상태를 찍으면 시작부터 어긋난다.
        # ── t=0 배치 다양화 (LLM 분포에서 표본)
        if PRIOR:
            ev = ctrl.step("Pass")
            byt = {}
            for rid in byroom:
                byt.setdefault(rt[rid], []).append(rid)
            for o in ev.metadata["objects"]:
                if not o.get("pickupable"):
                    continue
                d = PRIOR.get(o["objectType"])
                if not d:
                    continue
                ks = [k for k in d if k in byt]
                if not ks:
                    continue
                w = np.array([d[k] for k in ks], float)
                if w.sum() <= 0:
                    continue
                tgt_type = ks[int(rng.choices(range(len(ks)), weights=w / w.sum())[0])]
                tgt = rng.choice(byt[tgt_type])
                pt = rng.choice(byroom[tgt])
                ctrl.step("PlaceObjectAtPoint", objectId=o["objectId"],
                          position=dict(x=pt["x"], y=pt["y"] + 0.6, z=pt["z"]))
        mp = []
        for rid, ps in byroom.items():
            idx = np.linspace(0, len(ps) - 1, min(args.map_per_room, len(ps))).astype(int)
            for i in idx:
                for y in (0, 90, 180, 270):
                    e = ctrl.step("Teleport", position=ps[i],
                                  rotation=dict(x=0, y=y, z=0), horizon=10)
                    if e.metadata["lastActionSuccess"]:
                        Image.fromarray(e.frame).save(
                            os.path.join(hd, "map", "%04d.jpg" % len(mp)), quality=88)
                        mp.append(dict(room=rid, yaw=y))
        ev = ctrl.step("Pass")
        gt0 = {o["objectId"]: dict(type=o["objectType"],
                                   room=room_of((o["position"]["x"], o["position"]["z"]), rooms))
               for o in ev.metadata["objects"] if o.get("pickupable")}
        gt0 = {k: v for k, v in gt0.items() if v["room"]}

        # ── 1fps 배회 (RGB 만) + 미관측 이동
        rids = sorted(byroom)
        cur = rng.choice(rids)
        move_at = sorted(rng.sample(range(int(T * 0.1), T), min(args.moves, T)))
        live, events = [], []
        state = dict(gt0)
        mi = 0
        for t in range(T):
            if rng.random() < 1.0 / args.dwell:
                cur = rng.choice(rids)
            p = rng.choice(byroom[cur])
            y = rng.choice((0, 45, 90, 135, 180, 225, 270, 315))
            e = ctrl.step("Teleport", position=p, rotation=dict(x=0, y=y, z=0), horizon=10)
            if not e.metadata["lastActionSuccess"]:
                continue
            live.append(dict(t=t, room=cur))
            Image.fromarray(e.frame).save(
                os.path.join(hd, "live", "%06d.jpg" % t), quality=85)
            # 에이전트가 **다른 방**에 있을 때만 옮긴다 → 미관측
            while mi < len(move_at) and move_at[mi] <= t:
                mi += 1
                cands = [o for o, v in state.items() if v["room"] and v["room"] != cur]
                if not cands:
                    continue
                oid = rng.choice(cands)
                tgt = rng.choice([r for r in rids if r != state[oid]["room"]])
                pt = rng.choice(byroom[tgt])
                r2 = ctrl.step("PlaceObjectAtPoint", objectId=oid,
                               position=dict(x=pt["x"], y=pt["y"] + 0.6, z=pt["z"]))
                if r2.metadata["lastActionSuccess"]:
                    events.append(dict(t=t, oid=oid, frm=state[oid]["room"], to=tgt))
                    state[oid] = dict(state[oid]); state[oid]["room"] = tgt
        json.dump(dict(house=hi, rooms=[dict(id=r["id"], type=r["roomType"]) for r in rooms],
                       room_types=rt, map=mp, live=live, gt0=gt0,
                       moves=events, gt_end=state, fps=args.fps, T=T),
                  open(os.path.join(hd, "gt.json"), "w"))
        made += 1
        print("  주택 %d · 방 %d · 맵 %d장 · 배회 %d장 · 이동 %d건"
              % (hi, len(byroom), len(mp), len(live), len(events)), flush=True)
    ctrl.stop()
    print("완료 %d채" % made)


if __name__ == "__main__":
    main()
