#!/usr/bin/env python3
"""GPU 서버의 산출물 중 **가져올 것만** 골라 묶는다.

    python scripts/harvest.py --root data/thor4 --cache /tmp --out harvest.tar.gz

⚠️ **프레임은 두고 온다.** 실측으로 1채(1시간, 3,191프레임)가 이렇다:

    live/ JPEG   54 MB   ← 두고 옴 (전체의 99.6%)
    gt.json     3.5 MB   ← 축약해서 가져옴
    a3_ 캐시    176 KB   ← ★ OWL 추론 결과, 진짜 산출물
    qc_ 캐시     52 KB   ← ★

100채 × 4시간이어도 캐시는 90 MB, 축약 gt 는 200 MB 남짓이다.

⚠️ **gt.json 축약에서 무엇을 버리나.** 프레임마다 `ctr`(가시 물체 화면좌표)·
`anch`(정적 물체 화면좌표)·`dist` 를 저장하는데, **그 정보는 이미 앵커 캐시에
들어 있다.** GT 채점에는 `vis` 와 방 라벨만 있으면 된다. 다만 GT 기준 천장 실험
(`anch5.py` 계열)은 `ctr`/`anch` 를 쓰므로 `--keep-geom` 으로 남길 수 있다.
"""
import argparse, glob, json, os, shutil, subprocess, sys


def slim(g, keep_geom):
    """⚠️ `scene_meta`(방 폴리곤·문 연결·정적 물체의 방)는 **반드시 남긴다.**
    없으면 받은 쪽에서 `import prior` 로 막혀 분석 자체가 안 된다."""
    out = dict(g)
    assert g.get("scene_meta"), (
        "scene_meta 없음 — scripts/export_scene_meta.py 를 먼저 돌려라. "
        "이게 없으면 가져가도 분석을 못 한다.")
    live = []
    for m in g.get("live", []):
        r = dict(t=m["t"], room=m["room"], vis=m.get("vis", []))
        if keep_geom:
            for k in ("ctr", "anch", "dist", "apos", "yaw", "pitch"):
                if k in m:
                    r[k] = m[k]
        live.append(r)
    out["live"] = live
    if not keep_geom:
        out["map"] = [dict(room=x["room"], yaw=x["yaw"],
                           box={k: v for k, v in x.get("box", {}).items()})
                      for x in g.get("map", [])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="생성 데이터 루트 (data/thor4 등)")
    ap.add_argument("--cache", default="/tmp", help="npz 캐시가 있는 디렉터리")
    ap.add_argument("--prefix", default="a3_,qc_", help="가져올 캐시 접두어 (쉼표)")
    ap.add_argument("--out", default="harvest.tar.gz")
    ap.add_argument("--keep-geom", action="store_true",
                    help="ctr/anch/dist 를 남긴다. GT 기준 천장 실험에 필요하나 10배 커진다.")
    ap.add_argument("--stage", default="_harvest", help="임시 디렉터리")
    args = ap.parse_args()

    st = args.stage
    if os.path.exists(st):
        shutil.rmtree(st)
    os.makedirs(os.path.join(st, "gt"), exist_ok=True)
    os.makedirs(os.path.join(st, "cache"), exist_ok=True)

    n = 0; raw = 0; slimmed = 0
    for hd in sorted(glob.glob(os.path.join(args.root, "house_*"))):
        f = os.path.join(hd, "gt.json")
        if not os.path.exists(f):
            continue
        raw += os.path.getsize(f)
        g = slim(json.load(open(f)), args.keep_geom)
        d = os.path.join(st, "gt", os.path.basename(hd))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "gt.json")
        json.dump(g, open(p, "w"))
        slimmed += os.path.getsize(p)
        n += 1
    print("gt.json %d채 · 원본 %.1f MB → 축약 %.1f MB (%.0f%%)"
          % (n, raw / 1e6, slimmed / 1e6, 100 * slimmed / max(raw, 1)), flush=True)

    c = 0; cb = 0
    for pre in args.prefix.split(","):
        for f in sorted(glob.glob(os.path.join(args.cache, pre.strip() + "*.npz"))):
            shutil.copy2(f, os.path.join(st, "cache", os.path.basename(f)))
            cb += os.path.getsize(f); c += 1
    print("캐시 %d개 · %.1f MB" % (c, cb / 1e6), flush=True)

    for extra in ("data/thor_prior.json", "data/thor_move.json",
                  "data/thor_static_types.json", "data/thor_queries.json"):
        if os.path.exists(extra):
            shutil.copy2(extra, os.path.join(st, os.path.basename(extra)))

    # 실행 조건을 같이 남긴다 — 나중에 "무슨 설정이었나" 를 못 찾으면 비교가 깨진다
    json.dump(dict(root=args.root, houses=n, caches=c, keep_geom=args.keep_geom,
                   argv=" ".join(sys.argv)), open(os.path.join(st, "MANIFEST.json"), "w"), indent=1)

    subprocess.run(["tar", "czf", args.out, "-C", st, "."], check=True)
    print("→ %s · %.1f MB" % (args.out, os.path.getsize(args.out) / 1e6))
    print("   받은 뒤: tar xzf %s -C <원하는곳>" % args.out)


if __name__ == "__main__":
    main()
