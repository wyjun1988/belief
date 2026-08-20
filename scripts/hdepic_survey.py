#!/usr/bin/env python3
"""HD-EPIC 주석만으로 **부재 사례가 몇 건인지** 센다 — 영상 받기 전 타당성 확인.

    $P scripts/hdepic_survey.py --root /Volumes/External_SSD/khronos/hdepic

우리가 필요한 것은 **부재 라벨 + 긴 시간축**을 동시에 가진 데이터였다:

| | SceneDiff | ADT | **HD-EPIC** |
|---|---|---|---|
| 부재 라벨 | `Removed` 명시 | 없음(변위에서 유도 — ㉘ 에서 무리로 판명) | **`fixture` 변화** |
| 시간축 | 없음(3~19초) | 2분 | **41시간 · 다일간** |

`fixture` 는 **"어느 가구 위에 있었나"** 라 우리 과제와 정확히 일치한다. 같은 물체의
트랙에서 fixture 가 `counter.008` → `sink.002` 로 바뀌면 **"그 자리에 없다"** 가
직접 유도된다 — ADT 처럼 변위로 추측할 필요가 없다.

구조:
    assoc_info[video][obj_id] = {name, tracks:[{track_id, time_segment, masks:[mask_id]}]}
    mask_info[video][mask_id] = {frame_number, 3d_location, bbox, fixture}
"""
import argparse, json, os
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/hdepic")
    ap.add_argument("--min-gap", type=float, default=60.0,
                    help="fixture 가 바뀐 두 관측 사이 최소 간격(초). 짧으면 같은"
                         " 조작 중의 흔들림일 수 있다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assoc = json.load(open(os.path.join(args.root, "assoc_info.json")))
    mask = json.load(open(os.path.join(args.root, "mask_info.json")))
    print("녹화 %d개" % len(assoc))

    per_part = Counter()
    cases, mv_all = [], []
    n_obj = n_track = n_mask = 0
    for vid, objs in assoc.items():
        part = vid[:3]
        mi = mask.get(vid, {})
        for oid, o in objs.items():
            n_obj += 1
            # 이 물체의 모든 관측을 시간순으로: (시각, fixture, 위치)
            obs = []
            for t in o.get("tracks", []):
                n_track += 1
                for m in t.get("masks", []):
                    r = mi.get(m)
                    if not r:
                        continue
                    n_mask += 1
                    fx = r.get("fixture")
                    if fx:
                        obs.append((float(r["frame_number"]), fx,
                                    r.get("3d_location")))
            if len(obs) < 2:
                continue
            obs.sort()
            # fixture 가 바뀐 지점 = 그 자리를 떠났다
            for a, b in zip(obs, obs[1:]):
                if a[1] == b[1]:
                    continue
                dt = (b[0] - a[0]) / 30.0        # 30fps 가정
                if dt < args.min_gap:
                    continue
                d = None
                if a[2] and b[2]:
                    d = sum((x - y) ** 2 for x, y in zip(a[2], b[2])) ** 0.5
                cases.append(dict(video=vid, part=part, obj=o.get("name"),
                                  t_from=a[0] / 30.0, t_to=b[0] / 30.0,
                                  fx_from=a[1], fx_to=b[1], gap_s=dt, disp_m=d))
                per_part[part] += 1
                if d is not None:
                    mv_all.append(d)

    print("물체 %d · 트랙 %d · 관측 %d" % (n_obj, n_track, n_mask))
    print("\n**fixture 가 바뀐 사례 %d건** (간격 ≥ %.0fs)" % (len(cases), args.min_gap))
    print("참가자별:", dict(per_part.most_common()))
    if mv_all:
        import statistics as st
        mv_all.sort()
        print("변위: 중앙 %.2f m · p90 %.2f · 1m 이상 %.0f%%"
              % (st.median(mv_all), mv_all[int(len(mv_all) * .9)],
                 100 * sum(1 for x in mv_all if x >= 1) / len(mv_all)))
    gaps = sorted(c["gap_s"] for c in cases)
    if gaps:
        import statistics as st
        print("떠난 뒤 재관측까지 간격: 중앙 %.0f초 · p90 %.0f초 · 최대 %.0f초(%.1f시간)"
              % (st.median(gaps), gaps[int(len(gaps) * .9)], gaps[-1], gaps[-1] / 3600))
        for lo, hi in ((60, 300), (300, 1800), (1800, 7200), (7200, 1e9)):
            n = sum(1 for g in gaps if lo <= g < hi)
            lbl = "%d~%ds" % (lo, hi) if hi < 1e9 else ">%dh" % (lo / 3600)
            print("   %-10s %4d건  %s" % (lbl, n, "█" * (n // 20)))
    top = Counter(c["obj"] for c in cases)
    print("\n자주 옮겨지는 물체:", dict(top.most_common(8)))
    if args.out:
        json.dump(cases, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
