#!/usr/bin/env python3
"""3RScan 부재 측정 — **미관측 이동 100% · 인스턴스 ID 고정 · 사건 약 3,900건**.

    $P scripts/r3scan_absence.py --root /Volumes/exDisk/3rscan --cache …

### 왜 이 데이터인가

우리 시나리오("안 보는 사이에 누가 옮겼다")가 기존 데이터에 5~11% 밖에 없었다(㉟).
3RScan 은 변화가 **스캔 사이**에 일어나므로 **구성상 100%** 다. 그리고 인스턴스 ID 가
스캔 간 고정이라 ㉜ 에서 여섯 번 물린 라벨 문제가 원천적으로 안 생긴다.

| 사건 | 건수 |
|---|---|
| `removed` (씬에서 사라짐) | 542 (240쌍) |
| `rigid` (자리를 떠남, 씬엔 있음) | 3,323 (982쌍) |

### 설계 — `rigid` 는 **예측된 음성 대조군**이다

조건③(㊱: 떠남이 시야 이탈과 겹쳐야 한다)에 따르면 `rigid` 는 물체가 씬 안에 남으므로
**씬 수준 검출로는 안 잡혀야 한다.** 이것을 대규모로 확인한다 —
`removed` 는 잡히고 `rigid` 는 안 잡히면 조건③의 세 번째 독립 검증이다.

    양성  removed  인스턴스        → 검출도가 떨어져야 한다
    대조① rigid   인스턴스        → **안 떨어져야 한다**(씬엔 있으니까)
    대조② 변화 없는 인스턴스        → 안 떨어져야 한다

### 이번 세션에서 배운 통제를 전부 적용한다

  · 조건① 같은 라벨이 참조 스캔에 둘 이상이면 제외 + 데이터셋의 `ambiguity` 목록도 제외
  · 조건② 참조 스캔에서 실제로 검출된 것만 (`--cond2`)
  · **물체 자기 기준** — 절대 검출점수를 물체 간 비교하지 않는다
  · 창 대표값은 **중앙값이 아니라 상위 분위수**(㊴ — 물체는 있어도 대부분 프레임에 안 보인다)
"""
import argparse, json, os
from collections import Counter, defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--wq", type=float, default=0.90, help="창 대표 분위수")
    ap.add_argument("--cond2", type=float, default=0.0, help="참조 검출도 하한")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from scipy.stats import mannwhitneyu

    vocab = json.load(open(os.path.join(args.root, "vocab.json")))
    vi = {w: i for i, w in enumerate(vocab)}
    meta = json.load(open(os.path.join(args.root, "3RScan.json")))
    sd = os.path.join(args.root, "scans")

    def semseg(sid):
        p = os.path.join(sd, sid, "semseg.v2.json")
        if not os.path.exists(p):
            return None
        try:
            d = json.load(open(p))
        except Exception:
            return None
        return {int(g.get("objectId", g["id"])): g["label"] for g in d["segGroups"]}

    def owl(sid):
        p = os.path.join(args.cache, sid + ".npz")
        if not os.path.exists(p):
            return None
        return np.load(p, allow_pickle=True)["owl"]

    rows = []
    npair = 0
    for x in meta:
        R = x["reference"]
        lab_r = semseg(R); o_r = owl(R)
        if lab_r is None or o_r is None:
            continue
        # 조건① — 참조 스캔에서 라벨이 유일해야 한다
        cnt = Counter(lab_r.values())
        amb = set()
        for grp in (x.get("ambiguity") or []):
            for e in grp:
                amb.add(int(e.get("instance_source", -1)))
                amb.add(int(e.get("instance_target", -1)))
        for s in x.get("scans", []):
            S = s["reference"]
            o_s = owl(S)
            if o_s is None:
                continue
            npair += 1
            removed = set()
            for v in s.get("removed", []) or []:
                try:
                    removed.add(int(v) if not isinstance(v, dict)
                                else int(v.get("instance_reference", v.get("id", -1))))
                except (TypeError, ValueError):
                    pass
            # ⚠️ `rigid` 항목이 dict 인 곳과 정수 id 인 곳이 섞여 있다.
            moved = set()
            for r in s.get("rigid", []) or []:
                try:
                    moved.add(int(r["instance_reference"]) if isinstance(r, dict) else int(r))
                except (KeyError, TypeError, ValueError):
                    pass
            nonrig = set()
            for v in (s.get("nonrigid") or []):
                try:
                    nonrig.add(int(v) if not isinstance(v, dict)
                               else int(v.get("instance_reference", -1)))
                except (TypeError, ValueError):
                    pass
            for iid, lb in lab_r.items():
                if lb not in vi or cnt[lb] > 1 or iid in amb:
                    continue
                j = vi[lb]
                sb = float(np.quantile(o_r[:, j], args.wq))
                sa = float(np.quantile(o_s[:, j], args.wq))
                if sb < args.cond2:
                    continue
                kind = ("removed" if iid in removed else
                        "moved" if iid in moved else
                        "nonrigid" if iid in nonrig else "static")
                rows.append(dict(ref=R, rescan=S, iid=iid, label=lb, kind=kind,
                                 s_before=sb, s_after=sa, drop=sb - sa))

    if not rows:
        print("표본 없음 — 캐시가 아직 없을 수 있다"); return
    print("쌍 %d · 사건 %d" % (npair, len(rows)))
    print("  분포 %s" % dict(Counter(r["kind"] for r in rows)))
    st = [r["drop"] for r in rows if r["kind"] == "static"]
    print("\n%-10s %6s %10s %10s %9s" % ("종류", "n", "하락 중앙", "AUC vs 정적", "p"))
    for k in ("removed", "moved", "nonrigid"):
        v = [r["drop"] for r in rows if r["kind"] == k]
        if len(v) < 5 or len(st) < 5:
            continue
        u, p = mannwhitneyu(v, st, alternative="greater")
        print("  %-8s %6d %+10.4f %10.3f %9.3g"
              % (k, len(v), np.median(v), u / (len(v) * len(st)), p))
    print("  %-8s %6d %+10.4f %10s" % ("static", len(st), np.median(st), "—"))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
