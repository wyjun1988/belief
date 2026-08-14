"""우리 씬그래프 → home-jepa 에피소드(v1 스키마). **학습된 모델을 실제로 태우기 위한 다리.**

home-jepa 는 지금까지 시뮬레이터가 만든 에피소드, 또는 ADT **GT** 로 만든 에피소드로만
학습·평가했다(`homejepa/adt.py`). 이 모듈은 같은 스키마를 **실센서 지각**으로 채운다:

    관측 이벤트(POS/NEG)   ← 우리 씬그래프가 무엇을 언제 어디서 봤는가
    수용체·방 목록          ← 우리가 만든 정적 가구 노드 + 구역 분할
    질의의 **정답**         ← GT (채점 기준은 바꾸지 않는다)

즉 역할이 이렇게 갈린다: **우리는 관측 스트림을 만들고, home-jepa 모델이 안 보이는
사이의 위치를 추론한다.** 그래서 이 파일에는 추론이 없다 — 지각이 본 것만 적는다.

⚠️ 어휘 제약: home-jepa 모델은 고정 어휘로 학습됐다(방 8종, 수용체 타입 목록, 물체
24클래스, 수용체 최대 32개). 우리 카테고리를 그 어휘로 사상해야 하고, 사상 실패한
물체는 **제외**한다(억지로 끼워 넣으면 모델이 본 적 없는 입력이 된다).
"""
import os
import sys
from collections import Counter

import numpy as np

HJ = os.path.expanduser("~/work/home-jepa")
if HJ not in sys.path:
    sys.path.insert(0, HJ)

from homejepa.adt import CLS_MAP, FURN_CATS, HYST, NEAR_MAX, SIZE   # noqa: E402
from homejepa.model import MAX_LOC                                   # noqa: E402
from homejepa.world import ROOM_TYPES                                # noqa: E402

TICK_FRAMES = 50            # 5초 = home-jepa 의 틱
ZONE_TO_ROOM = {"living": "living_room", "kitchen": "kitchen",
                "dining": "dining_room", "bedroom": "bedroom",
                "bathroom": "bathroom", "office": "study"}


def _zone_room_type(z):
    return ZONE_TO_ROOM.get(z, "living_room")


def build_episode(graph, gt, assign_zone, ep_id=900000, tick_frames=TICK_FRAMES,
                  poses=None, cap=MAX_LOC, extra_cls=None):
    """씬그래프 → 에피소드 dict.

    graph        : 우리 4D 씬그래프 (placements/zone/support 포함)
    gt           : GT 타임라인 (질의 정답용)
    assign_zone  : world 좌표 → 구역 이름
    poses        : (F,4,4) 관찰자 궤적 — NEG(그 방에 있었다) 이벤트에 쓴다
    """
    # ⚠️ home-jepa 어휘는 **집에서 잃어버리는 개인 소지품 24종**이다(keys/phone/wallet/…).
    # 장식물·가구가 없어서 ADT 의 `wall artwork`(액자) 같은 것은 사상 규칙조차 없고,
    # 그대로 두면 에피소드에서 통째로 빠진다. `extra_cls` 로 억지 사상을 넣을 수 있지만
    # **모델이 학습 때 본 적 없는 조합**이 되므로 결과 해석에 단서를 달아야 한다.
    cls_map = dict(CLS_MAP)
    if extra_cls:
        cls_map.update({k.lower(): v for k, v in extra_cls.items()})
    objs = graph["objects"]
    n_frames = graph["n_frames"]
    T = max(int(np.ceil(n_frames / tick_frames)), 2)

    # --- 수용체: 우리가 관측한 정적 대형 가구 중 home-jepa 어휘에 있는 것 ---------
    cand = []
    for iid, o in objs.items():
        cat = (o.get("category") or "").lower()
        if cat not in FURN_CATS:
            continue
        if (o.get("gt_motion_type") or "").lower() not in ("static", ""):
            continue
        p = np.array(o["placements"][0]["position"], float)
        z = o["placements"][0].get("zone") or assign_zone(p)
        if z is None:
            continue
        cand.append((o["n_obs"], iid, p, FURN_CATS[cat], z, o.get("name")))
    cand.sort(key=lambda c: -c[0])          # 많이 본 것부터 (안정적인 표면)

    zones = []
    for _, _, _, _, z, _ in cand:
        if z not in zones:
            zones.append(z)
    rooms, tcount = [], Counter()
    room_idx = {}
    for i, z in enumerate(zones):
        t = _zone_room_type(z)
        rooms.append(dict(id=i, type=t, tidx=tcount[t], recepts=[]))
        tcount[t] += 1
        room_idx[z] = i

    recepts, rcount, loc_of = [], Counter(), {}
    for i in range(len(rooms)):             # 방마다 floor 유사 수용체 (home-jepa 관례)
        recepts.append(dict(id=len(recepts), type="floor", tidx=rcount["floor"],
                            room=i, pos=[0.5, 0.5]))
        rcount["floor"] += 1
        rooms[i]["recepts"].append(recepts[-1]["id"])
    floor_of_room = {i: rooms[i]["recepts"][0] for i in range(len(rooms))}

    for _, iid, p, rt, z, name in cand:
        if len(recepts) >= cap:
            break
        rid = room_idx[z]
        recepts.append(dict(id=len(recepts), type=rt, tidx=rcount[rt], room=rid,
                            pos=[float(p[0]), float(p[2])], name=name))
        rcount[rt] += 1
        rooms[rid]["recepts"].append(recepts[-1]["id"])
        loc_of[iid] = recepts[-1]["id"]
    R = np.array([[objs[i]["placements"][0]["position"] for i in loc_of]][0]) \
        if loc_of else np.zeros((0, 3))
    rec_ids = list(loc_of.values())

    def nearest(p, prev=None):
        if not len(R):
            return None
        d = np.linalg.norm(R - np.asarray(p, float), axis=1)
        j = int(np.argmin(d))
        if d[j] > NEAR_MAX:
            return None
        if prev is not None and rec_ids[j] != prev and prev in rec_ids:
            k = rec_ids.index(prev)
            if d[k] < d[j] + HYST:
                return prev
        return rec_ids[j]

    # --- 물체: 우리 그래프의 동적 물체 중 home-jepa 어휘에 있는 것 ---------------
    objects, oid_of, ccount = [], {}, Counter()
    for iid, o in objs.items():
        cat = (o.get("category") or "").lower()
        if cat not in cls_map:
            continue
        rec = gt.get(str(o.get("gt_instance", o.get("instance_id"))))
        if rec is None or rec["motion_type"] != "dynamic":
            continue
        cls = cls_map[cat]
        oid = len(objects)
        oid_of[iid] = oid
        p0 = o["placements"][0]["position"]
        home_rec = nearest(p0)
        objects.append(dict(id=oid, cls=cls, cidx=ccount[cls], owner=1,
                            size=SIZE.get(cls, "s"),
                            home_recept=home_rec if home_rec is not None else 0,
                            src_instance=str(iid), gt_instance=o.get("gt_instance"),
                            src_category=cat, substituted=bool(extra_cls and cat in
                                                              {k.lower() for k in extra_cls})))
        ccount[cls] += 1

    # --- 이벤트: **지각이 본 것만** ---------------------------------------------
    events, run = [], {}
    prev_loc = {}
    for tk in range(T):
        f0, f1 = tk * tick_frames, (tk + 1) * tick_frames - 1
        for iid, oid in oid_of.items():
            o = objs[iid]
            pl = next((p for p in o["placements"]
                       if p["start_frame"] <= f1 and p["end_frame"] >= f0), None)
            if pl is None:
                continue                    # 이 틱에 관측 없음 → 이벤트 없음
            loc = nearest(pl["position"], prev_loc.get(oid))
            if loc is None:
                z = pl.get("zone") or assign_zone(pl["position"])
                loc = floor_of_room.get(room_idx.get(z, 0), 0)
            prev_loc[oid] = loc
            r = run.get(oid)
            if r and r[2] == loc and tk - r[1] <= 6:
                r[1] = tk
            else:
                if r:
                    events.append(dict(t0=r[0], t1=r[1], type="POS", obj=oid,
                                       recept=r[2], room=recepts[r[2]]["room"]))
                run[oid] = [tk, tk, loc]
    for oid, r in run.items():
        events.append(dict(t0=r[0], t1=r[1], type="POS", obj=oid,
                           recept=r[2], room=recepts[r[2]]["room"]))

    # NEG: 관찰자가 그 방에 있었다 = 그 방의 수용체들을 훑었다 (부재 증거의 근거)
    if poses is not None:
        nrun = None
        for tk in range(T):
            f = min(tk * tick_frames, len(poses) - 1)
            z = assign_zone(poses[f][:3, 3])
            if z is None or z not in room_idx:
                continue
            room = room_idx[z]
            if nrun and nrun[0] == room and tk - nrun[2] <= 2:
                nrun[2] = tk
            else:
                if nrun:
                    nrec = len(rooms[nrun[0]]["recepts"])
                    events.append(dict(t0=nrun[1], t1=nrun[2], type="NEG", room=nrun[0],
                                       nz=nrec, cov=1.0))
                nrun = [room, tk, tk]
        if nrun:
            nrec = len(rooms[nrun[0]]["recepts"])
            events.append(dict(t0=nrun[1], t1=nrun[2], type="NEG", room=nrun[0],
                               nz=nrec, cov=1.0))
    events.sort(key=lambda e: (e["t1"], e["t0"], e["type"]))

    # --- 질의: 정답은 **GT 좌표**로 매긴다 (채점 기준은 바꾸지 않는다) ------------
    queries, gtrec = [], {}
    for iid, oid in oid_of.items():
        gi = str(objs[iid].get("gt_instance", objs[iid].get("instance_id")))
        rec = gt.get(gi)
        if rec is None:
            continue
        P = np.array(rec["positions"])
        prev = None
        seq = []
        for tk in range(T):
            f = min(tk * tick_frames, len(P) - 1)
            loc = nearest(P[f], prev)
            if loc is None:
                z = assign_zone(P[f])
                loc = floor_of_room.get(room_idx.get(z, 0), 0)
            prev = loc
            seq.append(loc)
        gtrec[oid] = seq

    # ⚠️ 관측 시점은 이벤트의 **시작 틱(t0)** 으로 잡아야 한다. 처음에 종료 틱(t1)을
    # 썼더니, 아직 진행 중인 관측(t1 이 계속 늘어나는 run)이 영원히 "과거"가 되지 않아
    # last-known 이 한 자리 뒤처졌다 — 액자를 PictureLedge 에서 보고 있는데도
    # BlackCoffeeTable 로 답하고 있었다. 모델 입력과 베이스라인이 함께 틀어진다.
    obs = {}                                # oid → [(tick, recept)] 관측 이력
    for e in events:
        if e["type"] == "POS":
            obs.setdefault(e["obj"], []).append((e["t0"], e["recept"]))
    for v in obs.values():
        v.sort()
    for oid, seq in gtrec.items():
        hist = obs.get(oid, [])
        for tk in range(1, T):
            past = [h for h in hist if h[0] < tk]
            if not past:
                continue
            last_t, last_rec = past[-1]
            g = seq[tk]
            queries.append(dict(qt=tk, obj=oid, gt_recept=g,
                                gt_room=recepts[g]["room"],
                                last_recept=last_rec,
                                last_room=recepts[last_rec]["room"],
                                last_t=last_t, dt=(tk - last_t) * 5,
                                moved=int(g != last_rec),
                                moved_room=int(recepts[g]["room"] != recepts[last_rec]["room"]),
                                mover=None, n_moves=0, dtbin="<10m"))

    return dict(home=dict(id=ep_id, n_agents=1, rooms=rooms, recepts=recepts, objects=objects),
                days=1, events=events, glances=[],
                gt=[dict(obj=o["id"], moves=[[0, o["home_recept"]]]) for o in objects],
                gt_moves=[], queries=queries, seed=ep_id,
                source=dict(graph=graph.get("depth_dir"), sequence=graph.get("sequence"),
                            n_recepts=len(recepts), n_objects=len(objects),
                            n_events=len(events), n_queries=len(queries)))
