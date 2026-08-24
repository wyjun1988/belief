#!/usr/bin/env python3
"""이동한 물체에 답하는 **두 경로** 중 어느 쪽이 실제로 열려 있나.

  (a) 원래 방에 **없음**을 확인 → belief 로 다른 방 추측   ← 부재 검출 필요
  (b) 새 방에서 **직접 발견**                              ← 검색만 필요

⚠️ (b) 가 열려 있으면 부재 검출이 아예 필요 없다. 지금까지 (a) 만 파고 있었다.
"""
import json, glob, os, numpy as np
ROOT = os.environ.get("THOR_ROOT", "data/thor3")
from collections import Counter

rows = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    g = json.load(open(hd + "/gt.json"))
    live = g["live"]; moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for oid, v in g["gt0"].items():
        if not v["room"] or cnt[v["type"]] > 1: continue
        mv = [x for x in moves if x["oid"] == oid]
        if not mv: continue
        t0 = mv[-1]["t"]; newr = mv[-1]["to"]; oldr = v["room"]
        aft = [m for m in live if m["t"] > t0]
        rows.append(dict(
            # (a) 원래 방을 다시 봤나
            revis_old=sum(m["room"] == oldr for m in aft),
            # (b) 새 방에 가봤나 / 거기서 실제로 물체가 보였나
            visit_new=sum(m["room"] == newr for m in aft),
            seen_new=sum(oid in m.get("vis", []) for m in aft)))

n = len(rows)
print("=== 이동한 물체 %d개 · 이동 이후 관측 ===" % n)
for k, nm in (("revis_old", "(a) 원래 방을 다시 봄"),
              ("visit_new", "(b) 새 방에 가봄"),
              ("seen_new",  "(b') 새 방에서 **실제로 보임**")):
    c = sum(r[k] >= 3 for r in rows)
    print("  %-26s %2d/%d = **%.3f**  (프레임 중앙 %.0f)"
          % (nm, c, n, c/n, np.median([r[k] for r in rows])))
both = sum(r["revis_old"] >= 3 and r["seen_new"] >= 3 for r in rows)
none = sum(r["revis_old"] < 3 and r["seen_new"] < 3 for r in rows)
only_b = sum(r["revis_old"] < 3 and r["seen_new"] >= 3 for r in rows)
only_a = sum(r["revis_old"] >= 3 and r["seen_new"] < 3 for r in rows)
print("\n  두 경로 다 열림      %2d (%.2f)" % (both, both/n))
print("  (b) 만 열림          %2d (%.2f)  ← 검색만으로 답 가능" % (only_b, only_b/n))
print("  (a) 만 열림          %2d (%.2f)  ← 부재 검출이 유일한 길" % (only_a, only_a/n))
print("  **둘 다 닫힘**       %2d (%.2f)  ← 원리적으로 belief 뿐" % (none, none/n))
