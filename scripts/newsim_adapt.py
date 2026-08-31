#!/usr/bin/env python3
"""새 시뮬레이터 에피소드 → 우리 표준 형식 어댑터.

    python scripts/newsim_adapt.py --ep <에피소드 디렉터리> --out data/newsim/ep1

산출: out/gt.json (우리 형식: rooms·room_types·gt0·moves·live·scene_meta)
      out/live/*.jpg (ego_left 1fps)

매핑:
  방        rect_xy → 사각 폴리곤 (cm→m)
  타겟      is_structure 아님 + 부피 < 0.15 m³ (들 수 있는 크기)
  정적 앵커  나머지 (가구·구조)
  moves     GT 스트림의 in-관계 변화
  live.room 관측 스트림의 ego 카메라 위치 → 방 사각형 포함 판정
  live.vis  관측 스트림 ego-source 갱신에 등장한 물체 ≈ 그 순간 보임
            ⚠️ 진짜 프레임 가시성 GT 가 아니라 근사 — 생성측에 요청 목록 항목
"""
import argparse, collections, glob, json, os, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ep = args.ep
    os.makedirs(os.path.join(args.out, "live"), exist_ok=True)

    sg = json.load(open(os.path.join(ep, "scene_graph.json")))
    CM = 0.01
    rooms = {r["id"]: r for r in sg["rooms"]}
    rt = {rid: r["label"] for rid, r in rooms.items()}
    polys = {}
    for rid, r in rooms.items():
        lo, hi = r["rect_xy"]["min"], r["rect_xy"]["max"]
        polys[rid] = [[lo[0]*CM, lo[1]*CM], [hi[0]*CM, lo[1]*CM],
                      [hi[0]*CM, hi[1]*CM], [lo[0]*CM, hi[1]*CM]]

    def room_of(x, y):
        for rid, r in rooms.items():
            lo, hi = r["rect_xy"]["min"], r["rect_xy"]["max"]
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
                return rid
        return None

    gt0 = {}; static = {}
    for o in sg["objects"]:
        bb = o.get("aabb_world")
        loc = o["transform"]["location"]
        r = room_of(loc[0], loc[1])
        if not r:
            continue
        vol = 1e9
        if bb:
            d = [(bb["max"][i]-bb["min"][i])*CM for i in range(3)]
            vol = d[0]*d[1]*d[2]
        if not o.get("is_structure") and vol < 0.15:
            gt0[o["id"]] = dict(type=o["class"], room=r)
        else:
            static[o["id"]] = dict(type=o["class"], room=r,
                                   pos=[round(loc[0]*CM, 2), round(loc[1]*CM, 2)])

    # moves: GT 스트림 in-관계 변화 (타겟만)
    cur = {}; moves = []
    for line in open(os.path.join(ep, "ground_truth_updates.jsonl")):
        d = json.loads(line)
        for u in d["updates"]:
            if u["op"] == "relationship" and u["p"] == "in" and u["s"] in gt0 \
               and u["o"] in rooms:
                if cur.get(u["s"]) is not None and cur[u["s"]] != u["o"]:
                    moves.append(dict(oid=u["s"], t=int(d["time_s"]),
                                      frm=cur[u["s"]], to=u["o"]))
                cur[u["s"]] = u["o"]

    # live: 1fps — ego 카메라 방 + ego-source 로 그 초에 언급된 물체(≈보임)
    cams = {}; seen = collections.defaultdict(set)
    for line in open(os.path.join(ep, "observed_graph_updates.jsonl")):
        d = json.loads(line)
        sec = int(d["time_s"])
        for c in d.get("cameras", []):
            if c.get("source") == "ego":
                cams[sec] = c                    # 마지막 것이 남음 (location+rotation)
        for u in d["updates"]:
            if u.get("source") == "ego" and u.get("s") in gt0:
                seen[sec].add(u["s"])
    live = []
    for t in sorted(cams):
        c = cams[t]; loc = c["location"]
        r = room_of(loc[0], loc[1])
        if r:
            e2 = dict(t=t, room=r, vis=sorted(seen.get(t, ())),
                      # 투영 국소화(§106-111 이식)용 포즈 — 채점/상한용.
                      # 시스템측 포즈는 SfM 사슬 몫(사다리 ④).
                      apos=[round(loc[0]*CM, 2), round(loc[1]*CM, 2)])
            if c.get("rotation_pyr_deg"):
                e2["pitch"], e2["yaw"] = (round(float(c["rotation_pyr_deg"][0]), 1),
                                          round(float(c["rotation_pyr_deg"][1]), 1))
            live.append(e2)

    # 프레임 추출 (이미 있으면 생략)
    if not glob.glob(os.path.join(args.out, "live", "*.jpg")):
        subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(ep, "ego_left.mp4"),
                        "-vf", "fps=1", "-q:v", "3",
                        os.path.join(args.out, "live", "%06d.jpg")], check=True)

    json.dump(dict(rooms=[dict(id=r, type=rt[r]) for r in rooms],
                   room_types=rt, gt0=gt0, moves=moves, live=live,
                   T=len(live), fps=1,
                   scene_meta=dict(polys=polys, static=static, doors=[]),
                   map=[]),
              open(os.path.join(args.out, "gt.json"), "w"))
    print("방 %d · 타겟 %d · 정적 %d · 이동 %d건 · live %d초"
          % (len(rooms), len(gt0), len(static), len(moves), len(live)))
    for m in moves:
        print("  이동:", m)


if __name__ == "__main__":
    main()
