#!/usr/bin/env python3
"""4D 씬그래프 생성 — 물체 배치 이력 + 방/구역 층.

    $P scripts/build_graph.py --seq <name> --depth depth_t23 --tag t23
    $P scripts/build_graph.py --seq <name> --depth gt/depth  --tag gtdepth   # 상한 대조

산출: data/seq/<name>/graph_<tag>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.anchors import load_semidense, scene_bbox        # noqa: E402
from kx.graph.build import build_graph                          # noqa: E402
from kx.graph.frames import up_vector                           # noqa: E402
from kx.graph.regions import SEED_CATEGORIES, assign, build_regions, summary   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _merge_seeds(seeds, radius):
    """같은 카테고리의 공간적으로 붙은 시드를 하나로 합친다.

    ⚠️ FastSAM 은 소파 하나를 트랙 **30개**로 쪼갠다. 조각 하나하나가 동등한 한 표를 가지면
    living 시드가 7 → 61개로 폭증하고, 측지 보로노이에서 시드 2개뿐인 dining 을 통째로
    먹어버린다(dining 35.5 m² → 0). 조각 수는 가구의 개수가 아니라 **분할기의 성질**이므로
    표를 그렇게 세면 안 된다. 관측 수로 가중해 대표점 하나만 남긴다.
    """
    if radius <= 0 or not seeds:
        return seeds
    out = []
    for cat in {s["category"] for s in seeds}:
        grp = [s for s in seeds if s["category"] == cat]
        grp.sort(key=lambda s: -s.get("n_obs", 0))
        used = [False] * len(grp)
        for i, a in enumerate(grp):
            if used[i]:
                continue
            pa = np.asarray(a["position"], float)
            members = [a]
            used[i] = True
            for j in range(i + 1, len(grp)):
                if used[j]:
                    continue
                if np.linalg.norm(np.asarray(grp[j]["position"], float) - pa) <= radius:
                    used[j] = True
                    members.append(grp[j])
            w = np.array([max(m.get("n_obs", 1), 1) for m in members], float)
            P = np.array([m["position"] for m in members], float)
            rep = dict(members[0])
            rep["position"] = list((P * w[:, None]).sum(0) / w.sum())
            rep["n_merged"] = len(members)
            out.append(rep)
    return out


STABLE_DUR = 40        # 프레임(4초). 이보다 짧게 머문 배치는 '운반 중'으로 본다
STABLE_OBS = 10
SUPPORT_MAX = 1.6      # m. home-jepa 의 receptacle 규칙(NEAR_MAX)과 같은 값을 쓴다
SUPPORT_MIN_EXTENT = 0.3


def attach_supports(g):
    """배치마다 (a) 정지/운반 구분, (b) **무엇 위에 놓였는가** 관계를 붙인다.

    관계가 없으면 점 구름이지 씬그래프가 아니다. 지지 가구(receptacle)는 우리
    그래프가 관측한 정적 대형 물체 중 가장 가까운 것으로 잡는다 — home-jepa 의
    ADT 변환과 같은 1.6m 규칙이라 belief 질의로 바로 이어진다.
    """
    furn = [(o["name"], o["instance_id"], np.array(o["placements"][0]["position"]))
            for o in g["objects"].values()
            if (o["gt_motion_type"] or "").lower() == "static"
            and o["extent_m"] and max(o["extent_m"]) >= SUPPORT_MIN_EXTENT]
    F = np.array([p for _, _, p in furn]) if furn else np.zeros((0, 3))
    for o in g["objects"].values():
        for p in o["placements"]:
            p["stable"] = bool(p["end_frame"] - p["start_frame"] >= STABLE_DUR
                               and p["n_obs"] >= STABLE_OBS)
            p["support"], p["support_id"], p["support_dist_m"] = None, None, None
            if len(F) == 0:
                continue
            d = np.linalg.norm(F - np.array(p["position"]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= SUPPORT_MAX and furn[j][1] != o["instance_id"]:
                p["support"], p["support_id"] = furn[j][0], furn[j][1]
                p["support_dist_m"] = round(float(d[j]), 3)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--depth", default="depth_t23")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--seg", default="gt/seg", help="마스크 폴더 (SAM: sam_daaam/seg)")
    ap.add_argument("--seg-ids", default="gt/seg_ids.json")
    ap.add_argument("--pose", default="pose/poses.txt",
                    help="궤적 파일 (DA3 포즈 실험은 pose/poses_da3.txt)")
    ap.add_argument("--seed-merge", type=float, default=0.0,
                    help="같은 카테고리 시드를 이 거리(m) 안에서 하나로 합친다(FastSAM 조각용). "
                         "0=끄기. ⚠️ GT 시드에서도 실제 별개 가구(의자 2개, 0.85m)가 합쳐질 수 "
                         "있어 기본은 끔 — 실험 결과 구역 회복 효과도 없었다(dining 0 그대로)")
    ap.add_argument("--zone-seeds", default=None,
                    help="구역 시드 JSON (clip_rooms.py 산출)")
    ap.add_argument("--zone-seeds-mode", default="replace", choices=["replace", "merge"],
                    help="replace=물체 시드를 버림 / merge=물체 시드와 합침")
    ap.add_argument("--max-extent", type=float, default=3.0,
                    help="이보다 큰 인스턴스(건물 셸 등)는 물체 노드에서 제외")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    tag = args.tag or args.depth.replace("/", "_")
    stats = json.load(open(os.path.join(seq_dir, "export.json")))
    adt_dir = stats["seq_dir"]

    print("== 관측 → 물체 노드 (%s)" % args.depth, flush=True)
    g = build_graph(seq_dir, depth_dir=args.depth, every=args.every,
                    max_extent=args.max_extent, pose_file=args.pose,
                    seg_dir=args.seg, seg_ids=args.seg_ids)
    print("   프레임 %d, 물체 %d" % (g["frames_processed"], len(g["objects"])))

    print("== 방/구역 분할", flush=True)
    poses = np.loadtxt(os.path.join(seq_dir, args.pose)).reshape(-1, 4, 4)
    pts, _ = load_semidense(stats["mps_semidense"], bbox=scene_bbox(poses))
    up = up_vector(adt_dir)

    # 시드는 **우리 그래프가 관측한** 정적 대형 가구의 추정 위치를 쓴다(GT 위치가 아니라).
    seeds = []
    for o in g["objects"].values():
        if (o.get("category") or "").lower() not in SEED_CATEGORIES:
            continue
        if (o["gt_motion_type"] or "").lower() != "static":
            continue
        seeds.append({"name": o["name"], "category": o["category"], "n_obs": o["n_obs"],
                      "extent_m": o["extent_m"], "position": o["placements"][0]["position"]})
    seeds = _merge_seeds(seeds, args.seed_merge)
    if args.zone_seeds:
        # 프레임 단위 CLIP 방 분류로 만든 시드로 **갈아끼운다** — 물체 카테고리(=GT)에
        # 대한 의존을 여기서 끊는 게 목적이므로 섞지 않고 통째로 대체한다.
        p = args.zone_seeds if os.path.isabs(args.zone_seeds) \
            else os.path.join(seq_dir, args.zone_seeds)
        extra = json.load(open(p))
        if args.zone_seeds_mode == "merge":
            # 두 소스가 **자신 있는 것만** 낸다: 프레임 분류는 벽으로 갈린 방(침실)에,
            # 물체 분류는 개방공간의 가구(냉장고·식탁의자)에 강하다. 합치면 서로 메운다.
            print("   구역 시드 병합: 물체 %d + 프레임 %d" % (len(seeds), len(extra)))
            seeds = seeds + extra
        else:
            print("   구역 시드 대체: %s (%d개)" % (os.path.basename(p), len(extra)))
            seeds = extra
    reg = build_regions(pts, poses, up, seeds)
    print("   up=%s  방 %d개  시드 %d개  구역 %s"
          % (np.round(up, 2), reg["n_rooms"], reg["n_seeds"], reg["zone_names"]))

    # ⚠️ 벽걸이 물체는 바닥 투영이 **벽 위**에 떨어져 어느 방으로 스냅될지 모호하다.
    # (액자 시작 위치 FloatingShelf04_1 이 그 경우였고, 그래서 "방에서 거실로 나왔다"는
    #  사건이 통째로 사라졌다.) 물리적으로 맞는 규칙은 **어느 방에서 보았는가**다 —
    # 벽에 걸린 것은 그것을 마주 보는 방에 속한다. 물체 셀이 자유공간이면 그대로 쓰고,
    # 벽/가구 위라 스냅이 필요하면 **관측 시점 카메라의 구역**으로 대체한다.
    def observer_zone(pl):
        lo, hi = pl["start_frame"], min(pl["end_frame"], len(poses) - 1)
        zs = [assign(reg, poses[t][:3, 3]) for t in range(lo, hi + 1, max((hi - lo) // 8, 1))]
        zs = [z for z in zs if z[1]]
        if not zs:
            return None, None
        from collections import Counter
        top = Counter(z[1] for z in zs).most_common(1)[0][0]
        return next(r for r, z in zs if z == top), top

    for o in g["objects"].values():
        for p in o["placements"]:
            r, z = assign(reg, p["position"], require_free=True)
            if z is None:
                r, z = observer_zone(p)
                p["zone_from_observer"] = True
            p["room"], p["zone"] = r, z
        for c in o["changes"]:
            c["from_zone"] = assign(reg, c["from"])[1]
            c["to_zone"] = assign(reg, c["to"])[1]

    g["regions"] = {
        "n_rooms": reg["n_rooms"], "zone_names": reg["zone_names"],
        "n_seeds": reg["n_seeds"], "res": reg["res"], "floor_h": reg["floor_h"],
        "up": np.round(up, 6).tolist(),
        "grid_lo": reg["grid"].lo.tolist(), "grid_shape": list(reg["grid"].shape),
        "summary": summary(reg),
        "seeds": [{"name": s["name"], "category": s["category"],
                   "zone": SEED_CATEGORIES[s["category"].lower()]}
                  for s in seeds][:40],
    }
    np.savez_compressed(os.path.join(seq_dir, "regions_%s.npz" % tag),
                        rooms=reg["rooms"], zones=reg["zones"], reach=reg["reach"],
                        occ=reg["occ"], lo=reg["grid"].lo, res=reg["res"])

    attach_supports(g)

    out = os.path.join(seq_dir, "graph_%s.json" % tag)
    json.dump(g, open(out, "w"), default=_json_default)
    print("\n구역 요약:")
    for z, s in g["regions"]["summary"].items():
        print("   %-9s %6.1f m2   방분포 %s" % (z, s["area_m2"], s["rooms"]))
    moved = [o for o in g["objects"].values()
             if len([p for p in o["placements"] if p["stable"]]) > 1]
    print("\n**정지 배치가 바뀐 물체 %d개** (운반 중 구간 제외):" % len(moved))
    for o in sorted(moved, key=lambda x: -x["n_obs"]):
        chain = [p for p in o["placements"] if p["stable"]]
        print("   %-28s %-16s" % (o["name"], o["category"]))
        for p in chain:
            print("        f%-4d-%-4d  %-9s %-22s %s"
                  % (p["start_frame"], p["end_frame"], p["zone"] or "-",
                     (p["support"] or "(지지물 없음)")[:22], np.round(p["position"], 2)))
    print("\n→ %s" % out)


if __name__ == "__main__":
    main()
