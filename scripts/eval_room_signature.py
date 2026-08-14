#!/usr/bin/env python3
"""방 서명(room signature) 교차검증 — **한 시퀀스에서 배우고 나머지 전부에서 시험한다.**

    $P scripts/eval_room_signature.py --train Apartment_release_decoration_seq137_M1292

v2 의 전제: 프레임에 **같이 보이는 정적 물체 조합**만으로 어느 방인지 안다.
포즈도 기하도 안 쓰므로 드리프트가 없고, 이게 되면 재측위 모듈이 통째로 필요 없어진다.

여기서 쓰는 것은 ADT **GT CSV 뿐**이다 — VRS 도 세그멘테이션도 받지 않는다:
    2d_bounding_box.csv  프레임별로 어떤 물체가 보였나 (가시율 포함)
    aria_trajectory.csv  프레임별 카메라 위치 → **채점용 정답**(학습에는 안 쓴다)
    instances.json       정적/동적 라벨
같은 아파트의 모든 시퀀스가 **동일한 world 좌표계**를 쓴다(GreySofa 위치가 소수점까지
일치하는 것으로 확인). 그래서 학습 시퀀스에서 만든 구역지도를 그대로 얹을 수 있다.

⚠️ 물체→구역 사전은 **학습 시퀀스에서만** 만든다. 시험 시퀀스의 정답은 채점에만 쓴다.
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

from kx.eval.room_belief import load_regions      # noqa: E402
from kx.graph.regions import assign               # noqa: E402

VIS_MIN = 0.05       # 가시율. ⚠️ 열 이름은 `[%]` 인데 실제 값은 **0~1 분수**다
STEP_NS = int(1e9)   # 1초 간격 = 1fps


def load_seq(d, step_ns=STEP_NS):
    """GT CSV → [(t_ns, {보이는 uid}, 카메라 위치)]"""
    bb = pd.read_csv(os.path.join(d, "2d_bounding_box.csv"),
                     usecols=["object_uid", "timestamp[ns]", "visibility_ratio[%]"])
    bb = bb[bb["visibility_ratio[%]"] >= VIS_MIN]
    tr = pd.read_csv(os.path.join(d, "aria_trajectory.csv"),
                     usecols=["tracking_timestamp_us", "tx_world_device",
                              "ty_world_device", "tz_world_device"])
    t_tr = tr["tracking_timestamp_us"].to_numpy() * 1000
    xyz = tr[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()

    ts = np.sort(bb["timestamp[ns]"].unique())
    keep, last = [], -(1 << 62)          # 1초 이상 벌어진 것만 (원본 프레임율에 무관)
    for t in ts:
        if t - last >= step_ns:
            keep.append(t)
            last = t
    vis = bb.groupby("timestamp[ns]")["object_uid"].apply(set)
    out = []
    for t in keep:
        j = int(np.argmin(np.abs(t_tr - t)))
        if abs(t_tr[j] - t) > 5e7:                 # 50ms 넘게 벌어지면 버린다
            continue
        out.append((int(t), vis.get(t, set()), xyz[j]))
    return out


def static_uids(d):
    inst = json.load(open(os.path.join(d, "instances.json")))
    return {int(k) for k, v in inst.items()
            if (v.get("motion_type") or "").lower() == "static"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Apartment_release_decoration_seq137_M1292")
    ap.add_argument("--gt-root", default=os.path.join(ROOT, "data", "adt", "gt"))
    ap.add_argument("--seq-root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default=None,
                    help="물체→방 사전의 출처. 미지정=GT(scene_objects.csv). "
                         "지정하면 v1 씬그래프(예: graph_sam.json, graph_allmodel_aligned.json)")
    ap.add_argument("--llm-dict", default=None,
                    help="물체→방 사전을 **LLM 상식**으로 만든다(기하 미사용). 예: Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--llm-cache", default="/tmp/kx_llm_rooms.json")
    ap.add_argument("--level", default="room", choices=["room", "zone"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # --- 구역지도: 학습 시퀀스에서 만든 것을 전 시퀀스에 공유 --------------------
    sd = os.path.join(args.seq_root, args.train)
    gname = (args.graph or "graph_gtdepth.json").replace(".json", "")
    rtag = gname.replace("graph_", "").replace("_aligned", "")
    meta = json.load(open(os.path.join(sd, gname + ".json")))["regions"]
    ref = load_regions(np.load(os.path.join(sd, "regions_%s.npz" % rtag)),
                       meta["zone_names"], meta["up"])
    label = (lambda p: assign(ref, p)[0]) if args.level == "room" \
        else (lambda p: assign(ref, p)[1])

    # --- 물체 → 방 사전: **학습 시퀀스에서만** ----------------------------------
    tdir = os.path.join(args.gt_root, args.train)
    if args.graph:
        # v1 씬그래프에서 뽑는다 — 위치도 정적/동적 판정도 **우리 지각 산출물**이다.
        # 정적 = 안정 배치가 하나뿐이고 변화가 기록되지 않은 노드(GT motion_type 을 안 쓴다).
        g = json.load(open(os.path.join(sd, args.graph)))
        obj_label, n_dyn = {}, 0
        for k, o in g["objects"].items():
            pls = o.get("placements") or []
            st = [p for p in pls if p.get("stable")]
            if len(st) > 1 or o.get("changes"):
                n_dyn += 1
                continue                       # 움직인 물체는 방 서명에 못 쓴다
            uid = o.get("gt_instance") or o.get("instance_id")
            if uid is None:
                continue
            y = label(np.array(pls[0]["position"], float))
            if y is not None:
                obj_label[int(uid)] = y
        stat = set(obj_label)
        print("사전 출처: %s (노드 %d · 동적 제외 %d · 정적 %d)"
              % (args.graph, len(g["objects"]), n_dyn, len(obj_label)))
    else:
        stat = static_uids(tdir)
        so = pd.read_csv(os.path.join(tdir, "scene_objects.csv"),
                         usecols=["object_uid", "t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"])
        so = so.groupby("object_uid").first()
        obj_label = {}
        for uid, r in so.iterrows():
            if int(uid) not in stat:
                continue
            y = label(np.array([r["t_wo_x[m]"], r["t_wo_y[m]"], r["t_wo_z[m]"]]))
            if y is not None:
                obj_label[int(uid)] = y
        print("사전 출처: GT scene_objects.csv (정적 %d)" % len(obj_label))

    # --- LLM 사전: 기하를 전혀 안 쓰고 **이름만으로** 방을 정한다 ----------------
    if args.llm_dict:
        from kx.llm.room_namer import RoomNamer
        zone_names = meta["zone_names"]
        # 방(room) 단위 실험이면 구역 이름을 방 번호로 되돌려야 한다 —
        # 학습 시퀀스에서 '구역 → 최빈 방' 대응을 만들어 쓴다.
        z2r = {}
        if args.level == "room":
            for uid, y in list(obj_label.items()):
                pass
        cache = {}
        if os.path.exists(args.llm_cache):
            cache = json.load(open(args.llm_cache))
        namer = RoomNamer(zone_names, model=args.llm_dict)
        inst = json.load(open(os.path.join(tdir, "instances.json")))
        uid_name = {int(k): (v.get("instance_name") or "") for k, v in inst.items()}
        # 구역 → 방 번호 대응(학습 시퀀스 기하로 한 번만; 라벨 이름을 방 번호로 옮기는 용도)
        zone2room = defaultdict(Counter)
        so2 = pd.read_csv(os.path.join(tdir, "scene_objects.csv"),
                          usecols=["object_uid", "t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"])
        so2 = so2.groupby("object_uid").first()
        for uid, r in so2.iterrows():
            p3 = np.array([r["t_wo_x[m]"], r["t_wo_y[m]"], r["t_wo_z[m]"]])
            rr, zz = assign(ref, p3)
            if rr is not None and zz is not None:
                zone2room[zz][rr] += 1
        z2r = {z: c.most_common(1)[0][0] for z, c in zone2room.items()}

        new, n_new = {}, 0
        for uid in list(obj_label):
            nm = uid_name.get(uid, "")
            if not nm:
                continue
            if nm in cache:
                z = cache[nm]
            else:
                z, _ = namer.label(nm)
                cache[nm] = z
                n_new += 1
            if z is None:
                continue
            new[uid] = z2r.get(z) if args.level == "room" else z
        json.dump(cache, open(args.llm_cache, "w"))
        obj_label = {k: v for k, v in new.items() if v is not None}
        stat = set(obj_label)
        print("사전 출처: LLM %s (%s · 새로 라벨 %d · 사전 %d개, **기하 미사용**)"
              % (args.llm_dict, args.level, n_new, len(obj_label)))

    train = load_seq(tdir)
    df = Counter()
    for _, vis, _ in train:
        for u in vis & stat:
            df[u] += 1
    N = max(len(train), 1)
    idf = {u: np.log(N / (1 + c)) for u, c in df.items()}
    print("학습: %s · 프레임 %d · 서명 물체 %d개 (%s 단위)"
          % (args.train[18:], len(train), len(obj_label), args.level))

    def predict(vis):
        sc = defaultdict(float)
        for u in vis:
            y = obj_label.get(u)
            if y is None:
                continue
            sc[y] += idf.get(u, np.log(N))
        return max(sc, key=sc.get) if sc else None

    # --- 시험: 나머지 전부 --------------------------------------------------------
    seqs = sorted(d for d in os.listdir(args.gt_root)
                  if os.path.isdir(os.path.join(args.gt_root, d))
                  and os.path.exists(os.path.join(args.gt_root, d, "2d_bounding_box.csv")))
    if args.limit:
        seqs = seqs[:args.limit]
    rows, by_act = [], defaultdict(lambda: [0, 0, 0.0])
    print("\n%-46s %-7s %-8s %-8s %s" % ("시퀀스", "프레임", "정확도", "최빈", "판정불가"))
    for s in seqs:
        try:
            fr = load_seq(os.path.join(args.gt_root, s))
        except Exception as e:
            print("%-46s 실패 %r" % (s[18:], e))
            continue
        fr = [(t, v, label(p)) for t, v, p in fr]
        fr = [(t, v, y) for t, v, y in fr if y is not None]
        if len(fr) < 10:
            continue
        pred = [predict(v) for _, v, _ in fr]
        n_none = sum(p is None for p in pred)
        ok = sum(p == y for p, (_, _, y) in zip(pred, fr))
        base = Counter(y for _, _, y in fr).most_common(1)[0][1] / len(fr)
        rows.append(dict(seq=s, n=len(fr), acc=ok / len(fr), base=base,
                         unresolved=n_none / len(fr), is_train=(s == args.train)))
        act = s.replace("Apartment_release_", "").split("_seq")[0]
        by_act[act][0] += ok
        by_act[act][1] += len(fr)
        by_act[act][2] += base * len(fr)
        print("%-46s %-7d %-8.3f %-8.3f %.2f%s"
              % (s[18:][:46], len(fr), ok / len(fr), base, n_none / len(fr),
                 "  ← 학습" if s == args.train else ""))

    print("\n=== 활동별 (학습 시퀀스 포함) ===")
    print("%-34s %-8s %-9s %-9s %s" % ("활동", "시퀀스", "프레임", "정확도", "최빈"))
    for act, (ok, n, b) in sorted(by_act.items()):
        cnt = sum(1 for r in rows if r["seq"].replace("Apartment_release_", "").split("_seq")[0] == act)
        print("%-34s %-8d %-9d %-9.3f %.3f" % (act, cnt, n, ok / n, b / n))
    held = [r for r in rows if not r["is_train"]]
    if held:
        w = sum(r["n"] for r in held)
        print("\n**미학습 %d 시퀀스 %d 프레임 → 정확도 %.3f (최빈 %.3f)**"
              % (len(held), w, sum(r["acc"] * r["n"] for r in held) / w,
                 sum(r["base"] * r["n"] for r in held) / w))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
