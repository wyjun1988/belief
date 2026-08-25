#!/usr/bin/env python3
"""이미 생성된 데이터에 **씬 메타**(방 폴리곤·문 연결·정적 물체의 방)를 채워 넣는다.

    python scripts/export_scene_meta.py --root data/thor3

⚠️ **왜 필요한가.** 분석 스크립트가 방 폴리곤과 문 연결을 쓰는데, 그건 `prior` 로
집 dict 를 다시 불러야 얻는다. 캐시만 가져오는 원격 실행에서는 불가능하다 —
4090 산출물을 받아도 `import prior` 에서 막힌다. `thor_gen2.py` 는 이제 생성 시
`gt.json["scene_meta"]` 에 저장하지만, **그 이전 데이터에는 없으므로** 이걸로 채운다.

`ai2thor`·`prior` 가 있는 곳에서 한 번만 돌리면 된다. 이후 분석은 어디서든 된다.
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    args = ap.parse_args()

    import prior
    from ai2thor.controller import Controller
    ds = prior.load_dataset("procthor-10k")["train"]

    def inside(x, z, pts):
        c = False; n = len(pts)
        for i in range(n):
            x1, z1 = pts[i]; x2, z2 = pts[(i + 1) % n]
            if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1:
                c = not c
        return c

    ctrl = None; n = 0
    for hd in sorted(glob.glob(os.path.join(args.root, "house_*"))):
        f = os.path.join(os.path.realpath(hd), "gt.json")
        g = json.load(open(f))
        if g.get("scene_meta") and not args.force:
            continue
        h = ds[g["house"]]
        polys = {r["id"]: [[c["x"], c["z"]] for c in r["floorPolygon"]] for r in h["rooms"]}
        ctrl = (Controller(scene=h, width=64, height=64, quality="Low")
                if ctrl is None else ctrl)
        ctrl.reset(scene=h)

        def room_of(p):
            for rid, pts in polys.items():
                if inside(p["x"], p["z"], pts):
                    return rid
            return None

        static = {}
        for o in ctrl.last_event.metadata["objects"]:
            if o.get("pickupable"):
                continue
            r = room_of(o["position"])
            if r:
                static[o["objectId"]] = dict(
                    type=o["objectType"], room=r,
                    pos=[round(o["position"]["x"], 2), round(o["position"]["z"], 2)])
        g["scene_meta"] = dict(
            polys=polys, static=static,
            doors=[[d.get("room0"), d.get("room1")] for d in h.get("doors", [])
                   if d.get("room0") and d.get("room1")])
        json.dump(g, open(f, "w"))
        n += 1
        print("  %s · 방 %d · 정적물체 %d · 문 %d"
              % (os.path.basename(hd), len(polys), len(static), len(g["scene_meta"]["doors"])),
              flush=True)
    print("완료 %d채" % n)


if __name__ == "__main__":
    main()
