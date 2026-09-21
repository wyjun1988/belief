#!/usr/bin/env python3
"""제로샷 탈락 프레임에 LoRA 채택 판정을 물은 결과 분석 + '제로샷 ∨ LoRA≥X' 규칙용 검증 파일 생성 (§166-99, 2026-09-21).

  python scripts/adopt_or_rule_eval.py <verdict.jsonl> <verify_subset.jsonl> <champ.jsonl> <out_prefix>
    verdict: lora_adopt_infer 산출(house,oid,t,i,ans,margin) — 문턱 없이 걸은 프레임 전부
    verify_subset: 같은 물체들의 t1 원본(scored=[i,s_ab,s_ac]) / champ: 챔피언 필터 파일(덮어쓸 바탕)
  출력: 표(제로샷 통과/탈락 × 맞는 상자/딴 상자 × LoRA 마진 구간) + <out_prefix>_orX.jsonl (X=0,1,2)
"""
import json, os, sys, collections, numpy as np
V = os.path.expanduser(os.environ.get("BENCH_DIR", "~/khcache/bench-v2full")); ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2")
VTH, VTH2 = float(os.environ.get("VERIFY_TH", "2.069")), float(os.environ.get("VERIFY_TH2", "0.887"))
vf, sf, cf, outp = sys.argv[1:5]
M = {}
for l in open(os.path.expanduser(vf)):
    d = json.loads(l)
    if d.get("ans"): M[(d["house"], d["oid"], int(d["i"]))] = float(d["margin"])
S = {(r["house"], r["oid"]): r for r in (json.loads(l) for l in open(os.path.expanduser(sf)))}
G = {}; Z = {}
def gt(h):
    if h not in G:
        g = json.load(open(f"{ROOT}/{h}/gt.json")); G[h] = ({m["t"]: m for m in g["live"]}, {m["oid"]: m for m in g["moves"]})
        Z[h] = np.load(f"{V}/cache/hs2_a_{h}.npz", allow_pickle=True)
    return G[h], Z[h]
tab = collections.Counter(); per = {}
for (h, oid), r in S.items():
    (live, mv), z = gt(h); ts, BX, vocab = z["ts"], z["bx"], list(z["vocab"]); ti = vocab.index(r.get("type") or oid.split("|")[0]) if (r.get("type") or oid.split("|")[0]) in vocab else None
    mt = mv[oid]["t"] if oid in mv else -1
    cnt = collections.Counter()
    for e in r["scored"]:
        i = int(e[0]); t = int(ts[i]); zs = e[1] >= VTH and e[2] >= VTH2; m = M.get((h, oid, i))
        c = live.get(t, {}).get("ctr", {}).get(oid); vis = oid in live.get(t, {}).get("vis", [])
        right = None
        if ti is not None and BX[i, ti][2] > 0 and c is not None: right = np.hypot(float(BX[i, ti][0]) * 768 - c[0], float(BX[i, ti][1]) * 768 - c[1]) < 80
        elif not vis: right = False
        post = t > mt
        if m is None: continue
        mb = "<0" if m < 0 else "0-1" if m < 1 else "1-2" if m < 2 else "≥2"
        tab[("제로샷통과" if zs else "제로샷탈락", "이동후" if post else "이동전", "맞는상자" if right else ("딴상자" if right is False else "미상"), mb)] += 1
        if post and right and not zs: cnt[mb] += 1
    per[(h, oid)] = cnt
print("프레임 표 (제로샷, 시점, 상자, LoRA 마진 구간) → 수")
for k in sorted(tab): print("  %-8s %-5s %-5s %-4s %4d" % (*k, tab[k]))
def rate(zs, right):
    tot = sum(v for k, v in tab.items() if k[0] == zs and k[2] == right); yes = sum(v for k, v in tab.items() if k[0] == zs and k[2] == right and k[3] != "<0")
    hi = sum(v for k, v in tab.items() if k[0] == zs and k[2] == right and k[3] in ("1-2", "≥2"))
    return tot, yes / max(tot, 1), hi / max(tot, 1)
for zs in ("제로샷통과", "제로샷탈락"):
    for right in ("맞는상자", "딴상자"):
        t, y, h = rate(zs, right); print("  %s·%s: n=%d · LoRA 예(≥0) %.2f · 고마진(≥1) %.2f" % (zs, right, t, y, h))
print("물체별: 이동 후 맞는 상자인데 제로샷 탈락 프레임의 LoRA 마진 구간 수")
for k, c in per.items():
    if sum(c.values()): print("  %-11s %-16s %s" % (k[0], k[1][:16], dict(c)))
# OR 규칙 파일: 제로샷 탈락 & LoRA 마진 ≥ X 인 항목의 점수를 통과값으로 덮어쓴다 (대상 물체만)
for X in (0, 1, 2):
    n = 0
    with open(os.path.expanduser(f"{outp}_or{X}.jsonl"), "w") as fo:
        for l in open(os.path.expanduser(cf)):
            d = json.loads(l); k = (d["house"], d["oid"])
            if k in S:
                new = []
                for e in d["scored"]:
                    i = int(e[0]); zs = e[1] >= VTH and e[2] >= VTH2; m = M.get((k[0], k[1], i))
                    if not zs and m is not None and m >= X: new.append([i, 9.9, 9.9]); n += 1
                    else: new.append(e)
                d["scored"] = new
            fo.write(json.dumps(d) + "\n")
    print("→ %s_or%d.jsonl (제로샷 면제 %d장)" % (outp, X, n))
