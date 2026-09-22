#!/usr/bin/env python3
"""장면그래프 문맥(ctx)을 학습셋 행에 붙인다 — "기록 자리 주변에 무엇이 있었나" (2026-09-22, 사용자 제안 2·3의 전제 검증).

지금까지 판정기에 준 것은 기록 한 줄(참조 크롭 + 타입)뿐이고 장면그래프는 정책 쪽에서만 썼다.
이 스크립트는 각 행의 기록 자리 반경 RADIUS m 안의 **정적 물체 타입**을 뽑아 `ctx` 필드로 넣는다.
학습에서는 GT 위치(gt0.pos)를 기준으로 뽑고, 추론에서는 초기맵 기록 자리를 쓴다(adopt_pack.py CTX=1) —
둘 다 "그 물건 주변" 이라는 같은 뜻이고 자리 오차가 작으면 목록이 거의 같다.

  python scripts/lora_ctx_augment.py --pack ~/khcache/lora_adopt_v2 --root data/hssd_v2 [--out <dir>]
  (--root 없이 --roots a b c 로 주면 집 이름으로 찾아 첫 일치를 쓴다 — 같은 집 이름이 여러 데이터셋에 있으면 쓰지 말 것)
"""
import argparse, json, os, glob, math, collections
ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True); ap.add_argument("--root", default=None); ap.add_argument("--roots", nargs="*", default=[])
ap.add_argument("--radius", type=float, default=2.5); ap.add_argument("--max", type=int, default=5)
ap.add_argument("--out", default=None); ap.add_argument("--splits", nargs="*", default=["train", "val"])
a = ap.parse_args()
OUT = os.path.expanduser(a.out or (a.pack.rstrip("/") + "_ctx")); os.makedirs(OUT, exist_ok=True)
ROOTS = [a.root] if a.root else a.roots
_G = {}
def gt_of(house):
    if house in _G: return _G[house]
    for r in ROOTS:
        p = os.path.join(r, house, "gt.json")
        if os.path.exists(p): _G[house] = json.load(open(p)); return _G[house]
    _G[house] = None; return None
def words(t): return set(str(t).replace("_", " ").lower().split())
def ctx_of(row):
    g = gt_of(row["house"])
    if not g: return ""
    v0 = (g.get("gt0") or {}).get(row["oid"])
    pos = (v0 or {}).get("pos")
    if not pos: return ""
    st = (g.get("scene_meta") or {}).get("static") or {}
    tw = words(row.get("type"))
    near = []
    for k, v in st.items():
        p = v.get("pos")
        if not p or (words(v.get("type")) & tw): continue
        d = math.hypot(p[0] - pos[0], p[2] - pos[2])
        if d <= a.radius: near.append((d, str(v.get("type")).replace("_", " ")))
    if not near: return ""
    seen = []; 
    for _, t in sorted(near):
        if t not in seen: seen.append(t)
        if len(seen) >= a.max: break
    return "Around the recorded place: " + ", ".join(seen) + "."
st = collections.Counter()
for sp in a.splits:
    fi = os.path.join(os.path.expanduser(a.pack), sp + ".jsonl")
    if not os.path.exists(fi): continue
    n = 0
    with open(os.path.join(OUT, sp + ".jsonl"), "w") as fo:
        for l in open(fi):
            r = json.loads(l); c = ctx_of(r)
            if c: st[sp + " ctx있음"] += 1
            else: st[sp + " ctx없음"] += 1
            r["ctx"] = c; fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print("  %s %d행 → %s" % (sp, n, OUT))
print("집계:", dict(st))
ex = [json.loads(l) for l in open(os.path.join(OUT, a.splits[0] + ".jsonl"))][:3]
for r in ex: print("  예:", r.get("type"), "|", (r.get("ctx") or "(없음)")[:100])
print("CTX_AUGMENT_DONE")
