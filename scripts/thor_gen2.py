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
    # ⚠️ **한국 주거 환경에 맞춘다.** ProcTHOR 에는 방이 10개 넘는 집이 섞여 있고,
    # 그런 집이 들어가면 후보 방이 많아져 앵커 국소화의 천장이 무너진다
    # (3주택 0.897 → 20주택 0.621). 침실 4개짜리 집에서 "어느 침실" 을 가르는 것은
    # 우리 시나리오가 아니다. 화장실 뺀 방 2~6, 화장실 포함 8 이하로 제한한다.
    ap.add_argument("--max-rooms", type=int, default=8,
                    help="화장실 포함 전체 방 수 상한")
    ap.add_argument("--max-nonbath", type=int, default=6,
                    help="화장실 뺀 방 수 상한")
    ap.add_argument("--min-nonbath", type=int, default=2,
                    help="화장실 뺀 방 수 하한")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dwell", type=int, default=90, help="한 방 체류 초(평균)")
    ap.add_argument("--moves", type=int, default=10, help="세션 중 이동 사건 수")
    ap.add_argument("--map-per-room", type=int, default=3)
    ap.add_argument("--platform", default=None, choices=[None, "cloud"],
                    help="리눅스 헤드리스 GPU 에서는 `cloud`(CloudRendering). "
                         "Unity 창이 없으므로 이걸 안 주면 컨트롤러가 뜨지 않는다. "
                         "Vulkan 드라이버가 필요하다 — `vulkaninfo` 로 먼저 확인할 것.")
    ap.add_argument("--move", default=None,
                    help="Qwen 움직임 사전확률(scripts/thor_move_llm.py). 주면 방 체류·"
                         "물체 이동성향·이동 목적지를 현실 분포로 뽑는다. 없으면 균등.")
    ap.add_argument("--vis-dist", type=float, default=20.0,
                    help="가시성 판정 거리. 기본 1.5m 는 '보인다' 를 거리로 잘라버린다.")
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
    # ⚠️ `rng` 는 random.Random 이라 가중 추출(p=)을 못 받는다. 가중이 필요한 곳만
    # 별도 numpy RNG 를 쓴다. 같은 seed 에서 갈라지므로 재현성은 유지된다.
    nrng = np.random.default_rng(args.seed)

    def bytype(seq, table, rtmap, dflt=.25):
        """방 **유형** 확률을 방 **인스턴스** 가중으로 편다.

        ⚠️ 유형 확률을 인스턴스마다 그대로 주면 유형 분포가 깨진다. ProcTHOR 집은
        화장실·침실이 여럿이라, 화장실 3개에 각각 0.15 를 주면 합이 0.45 가 된다.
        실측에서 그렇게 물렸다(거실 의도 0.40 → 실측 0.22, 화장실 0.15 → 0.28).
        유형 몫을 **그 유형의 방 개수로 나눠** 인스턴스에 배분한다."""
        n = {}
        for r in seq:
            n[rtmap.get(r)] = n.get(rtmap.get(r), 0) + 1
        return [table.get(rtmap.get(r), dflt) / max(n.get(rtmap.get(r), 1), 1) for r in seq]

    def wpick(seq, w):
        """가중 추출. w 가 비거나 합이 0 이면 균등."""
        a = np.asarray(w, float)
        if len(a) != len(seq) or not np.isfinite(a).all() or a.sum() <= 0:
            return seq[nrng.integers(len(seq))]
        return seq[nrng.choice(len(seq), p=a / a.sum())]
    T = int(args.hours * 3600 * args.fps)

    PRIOR = json.load(open(args.prior)) if args.prior else None
    MOVE = json.load(open(args.move)) if args.move else None
    if MOVE:
        print("움직임 사전확률: 체류 " + " ".join("%s %.2f" % (k, v)
              for k, v in MOVE["dwell"].items()), flush=True)
    import prior
    from ai2thor.controller import Controller
    from PIL import Image
    ds = prior.load_dataset("procthor-10k")["train"]
    # ⚠️ **visibilityDistance 를 반드시 올려야 한다.** 기본 1.5 m 라서
    # 방 건너편에 뻔히 보이는 물체가 전부 `visible: False` 로 라벨된다
    # (실측: 한 장면에서 pickupable 52개 중 visible=True 가 **0개**, 가장 가까운 것이 1.6 m).
    # 그 GT 로 검출을 채점하면 멀리 있는 물체를 잡을 때마다 **오검출로 세어진다.**
    kw = {}
    if args.platform == "cloud":
        from ai2thor.platform import CloudRendering
        kw["platform"] = CloudRendering
    ctrl = Controller(scene=ds[0], width=args.size, height=args.size, quality="Low", **kw,
                      visibilityDistance=args.vis_dist,
                      renderInstanceSegmentation=True)
    made = 0
    for hi in range(len(ds)):
        if made >= args.houses:
            break
        h = ds[hi]
        nb = sum(1 for r in h["rooms"] if r["roomType"] != "Bathroom")
        if not (args.min_rooms <= len(h["rooms"]) <= args.max_rooms
                and args.min_nonbath <= nb <= args.max_nonbath):
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
        if not (args.min_rooms <= len(byroom) <= args.max_rooms):
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
                        # 씬그래프를 처음 만들 때 **물체의 bbox 도 같이 남긴다.**
                        # 질의 시점에 이 crop 을 이미지 질의(exemplar)로 쓰기 위해서다 —
                        # 글자 "머그컵" 이 아니라 **그 머그컵** 을 찾아야 한다.
                        bx = e.instance_detections2D
                        mb = {o["objectId"]: [int(v) for v in bx[o["objectId"]]]
                              for o in e.metadata["objects"]
                              if o.get("visible") and o.get("pickupable")
                              and bx.get(o["objectId"]) is not None}
                        mp.append(dict(room=rid, yaw=y, box=mb))
        ev = ctrl.step("Pass")
        gt0 = {o["objectId"]: dict(type=o["objectType"],
                                   room=room_of((o["position"]["x"], o["position"]["z"]), rooms))
               for o in ev.metadata["objects"] if o.get("pickupable")}
        gt0 = {k: v for k, v in gt0.items() if v["room"]}

        # ── 1fps 배회 (RGB 만) + 미관측 이동
        # ⚠️ 예전엔 방·물체·목적지가 **전부 균등 난수**였다. 배치만 Qwen 으로
        # 현실화하고 움직임을 균등으로 두면 두 군데가 망가진다:
        #  (1) 목적지가 균등이면 belief 가 원리적으로 이동 물체를 못 맞힌다.
        #      실제로는 머그컵이 부엌→거실로 가지 부엌→화장실로 가지 않는다.
        #      belief 몫이 전체의 38% 인데 그 성능이 실제보다 나쁘게 측정된다.
        #  (2) 방마다 재방문 확률이 같아져 (b)"있을 것이다" 가 비현실적으로 고르게 난다.
        #      실제 가정에서 화장실 체류는 짧고 거실은 길다.
        rids = sorted(byroom)
        rw = (bytype(rids, MOVE["dwell"], rt) if MOVE
              else [1.0] * len(rids))            # 방 체류 가중 (Qwen, 유형→인스턴스)
        cur = wpick(rids, rw)
        move_at = sorted(rng.sample(range(int(T * 0.1), T), min(args.moves, T)))
        live, events = [], []
        state = dict(gt0)
        mi = 0
        for t in range(T):
            if rng.random() < 1.0 / args.dwell:
                cur = wpick(rids, rw)
            p = rng.choice(byroom[cur])
            y = rng.choice((0, 45, 90, 135, 180, 225, 270, 315))
            e = ctrl.step("Teleport", position=p, rotation=dict(x=0, y=y, z=0), horizon=10)
            if not e.metadata["lastActionSuccess"]:
                continue
            # ⚠️ 프레임별 **가시 물체**를 기록한다. 2차에 이게 없어서
            # "배회 중 검출이 얼마나 되나" 를 직접 못 쟀다(1차 맵 프레임의 0.949 는
            # 자리마다 4방향 정지 촬영이라 조건이 다르다).
            # 물체까지의 수평거리도 같이 남긴다. vis_dist 를 20m 로 풀면 물체가
            # **옆방에서도 보이므로** "보인 프레임의 에이전트 방" 을 물체의 방으로
            # 읽으면 안 된다 — 거리별로 그 대응이 언제 깨지는지 재려면 거리가 필요하다.
            # 물체까지의 거리와 **화면상 bbox 중심**을 남긴다. vis_dist 를 20m 로 풀면
            # 물체가 옆방에서도 보이므로 "보인 프레임의 에이전트 방" 을 물체의 방으로
            # 읽으면 안 된다(실측 일치 0.257). 대신 최초 맵에서 depth 로 위치를 박아둔
            # **정적 물체를 앵커로** 쓰려면, 타겟과 앵커의 화면상 거리가 필요하다.
            ap = e.metadata["agent"]["position"]
            box = e.instance_detections2D
            vd = {}; vs = {}; vc = {}
            for o in e.metadata["objects"]:
                if not o.get("visible"):
                    continue
                oid = o["objectId"]; op = o["position"]
                d = round(((op["x"] - ap["x"]) ** 2 + (op["z"] - ap["z"]) ** 2) ** .5, 2)
                b = box.get(oid)
                c = [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)] if b is not None else None
                if o.get("pickupable"):
                    vd[oid] = d; vc[oid] = c
                else:
                    vs[oid] = c
            live.append(dict(t=t, room=cur, vis=list(vd), dist=vd, ctr=vc,
                             anch=vs, apos=[round(ap["x"], 2), round(ap["z"], 2)]))
            Image.fromarray(e.frame).save(
                os.path.join(hd, "live", "%06d.jpg" % t), quality=85)
            # 에이전트가 **다른 방**에 있을 때만 옮긴다 → 미관측
            while mi < len(move_at) and move_at[mi] <= t:
                mi += 1
                cands = [o for o, v in state.items() if v["room"] and v["room"] != cur]
                if not cands:
                    continue
                # 물체별 **이동 성향**으로 뽑는다 — 머그컵은 자주, 야구방망이는 거의 안.
                oid = wpick(cands, [MOVE["mobility"].get(state[o]["type"], .5)
                                    for o in cands] if MOVE else [1.0] * len(cands))
                pool = [r for r in rids if r != state[oid]["room"]]
                if not pool:
                    continue
                # 목적지는 **배치 사전확률과 다르다.** 머그컵은 부엌에 놓이지만
                # 옮겨지면 거실에서 발견된다. 그래서 dest 를 따로 물었다.
                dd = MOVE["dest"].get(state[oid]["type"], {}) if MOVE else {}
                tgt = wpick(pool, bytype(pool, dd, rt) if dd else [1.0] * len(pool))
                pt = rng.choice(byroom[tgt])
                r2 = ctrl.step("PlaceObjectAtPoint", objectId=oid,
                               position=dict(x=pt["x"], y=pt["y"] + 0.6, z=pt["z"]))
                if r2.metadata["lastActionSuccess"]:
                    events.append(dict(t=t, oid=oid, frm=state[oid]["room"], to=tgt))
                    state[oid] = dict(state[oid]); state[oid]["room"] = tgt
        # ⚠️ **분석에 필요한 씬 메타를 여기서 같이 저장한다.** 방 폴리곤·문 연결·
        # 정적 물체의 방은 `prior`/`ai2thor` 를 다시 띄워야 얻을 수 있는데, 캐시만
        # 가져가는 원격 실행에서는 그게 불가능하다. 실제로 물렸다 — 4090 산출물을
        # 받아도 `import prior` 에서 막힌다.
        ev_m = ctrl.step("Pass")
        meta = dict(
            polys={r["id"]: [[c["x"], c["z"]] for c in r["floorPolygon"]] for r in h["rooms"]},
            doors=[[d.get("room0"), d.get("room1")] for d in h.get("doors", [])
                   if d.get("room0") and d.get("room1")],
            # ⚠️ **3D 좌표도 남긴다.** 앵커를 가방(어떤 타입이 보이나)으로만 쓰면
            # 배치 정보를 버리게 된다. "소파 왼쪽에 TV" 는 {소파, TV} 보다 변별력이 크다.
            static={o["objectId"]: dict(
                        type=o["objectType"],
                        pos=[round(o["position"]["x"], 2), round(o["position"]["z"], 2)],
                        room=room_of((o["position"]["x"], o["position"]["z"]), rooms))
                    for o in ev_m.metadata["objects"] if not o.get("pickupable")})
        meta["static"] = {k: v for k, v in meta["static"].items() if v["room"]}
        json.dump(dict(house=hi, scene_meta=meta,
                       rooms=[dict(id=r["id"], type=r["roomType"]) for r in rooms],
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
