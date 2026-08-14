"""HOMER+ 루틴 → 물체 위치 타임라인 + **관측 모델**.

HOMER+ 는 가정 3채 × 75일의 물체 배치열이다(하루 ~210 스텝, 분 단위 타임스탬프).
ADT 로는 못 재는 것을 여기서 잰다: **안 보는 사이에 물건이 움직이는 상황**.

⚠️ 원본에는 에이전트 노드가 없다. 사용자가 직접 물건을 옮기므로 "사용자 = 카메라"로
두면 모든 이동이 관측되어 ADT 와 같은 퇴화가 반복된다(v1 에서 belief 모델이 last-known
과 소수점까지 같았던 이유). 그래서 **행위자와 관측자를 분리**한다 — 방을 순찰하는
보조 장치가 관측자다. 이게 v2 가 겨냥하는 실제 상황이고(밤에 배치 업데이트),
순찰 간격이 곧 '미관측 이동'의 크기를 정한다.
"""
import json
import os

ROOM_CATS = {"Rooms"}
MOVABLE = {"placable_objects"}


def _index(graph):
    """그래프 한 장 → (노드 사전, 물체 id → 방 이름)."""
    node = {n["id"]: n for n in graph["nodes"]}
    room_of_node = {}
    for n in graph["nodes"]:
        if n["category"] in ROOM_CATS:
            room_of_node[n["id"]] = n["class_name"]

    parent = {}
    for e in graph["edges"]:
        if e["relation_type"] in ("INSIDE", "ON"):
            parent[e["from_id"]] = e["to_id"]

    def room_of(i, depth=0):
        if i in room_of_node:
            return room_of_node[i]
        if depth > 8 or i not in parent:
            return None
        return room_of(parent[i], depth + 1)

    loc = {}
    for n in graph["nodes"]:
        if n["category"] not in MOVABLE:
            continue
        p = parent.get(n["id"])
        loc[n["id"]] = (p, room_of(n["id"]))       # (직접 지지면, 방)
    return node, loc


def load_day(path):
    """하루 → dict(times, activities, names, loc[t][obj] = (지지면, 방))"""
    d = json.load(open(path))
    node0, _ = _index(d["graphs"][0])
    names = {i: n["class_name"] for i, n in node0.items()}
    locs = []
    for g in d["graphs"]:
        _, loc = _index(g)
        locs.append(loc)
    return dict(times=d["times"], activities=d["activities"],
                names=names, locs=locs)


def load_household(root, split="train", limit=None):
    d = os.path.join(root, "routines_" + split)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if limit:
        fs = fs[:limit]
    return [load_day(os.path.join(d, f)) for f in fs]


def timeline(days):
    """여러 날을 이어 붙인 절대 시각 타임라인.

    반환 [(분, {obj: (지지면, 방)}, 활동)] — 분은 day*1440 + 분.
    """
    out = []
    for k, day in enumerate(days):
        for t, loc, act in zip(day["times"], day["locs"],
                               [None] + list(day["activities"])):
            out.append((k * 1440 + float(t), loc, act))
    return out


def moves(tl):
    """방이 **바뀐** 순간만 추린다. [(분, obj, from_room, to_room)]"""
    out = []
    prev = tl[0][1]
    for t, loc, _ in tl[1:]:
        for o, (_, r) in loc.items():
            pr = prev.get(o, (None, None))[1]
            if r is not None and pr is not None and r != pr:
                out.append((t, o, pr, r))
        prev = loc
    return out
