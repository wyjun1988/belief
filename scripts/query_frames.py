#!/usr/bin/env python3
"""평가기가 포즈를 실제로 쓰는 live 프레임만 뽑는다 — 1fps 전부를 이어 붙일 이유가 없다(사용자 지적 2026-09-04).
  ② 재관측: 검증기가 채점한 후보 프레임(통과분 + 타겟당 상위 N)
  ③ 부재: 이동 뒤 원래 방을 향한 프레임은 검출 캐시에 없으므로, 여기서는 '검증 채점 프레임' 집합으로 근사한다
     (평가기의 부재 판정도 같은 후보 프레임의 패치 위치를 쓴다).
    THOR_ROOT=data/hssd20S2 A3_PREFIX=~/khcache/bench-v2/cache/hs2_a_ VERIFY_JSONL=… python scripts/query_frames.py --out q.json
출력: {"house_0000": [t, t, …], …}  (t = live 프레임 번호). vggt_reloc/cut3r_reloc --query-frames q.json 이 읽는다.
"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=6, help="타겟당 s_ab 상위 N (통과분과 합집합)")
a = ap.parse_args()
ROOT = os.environ.get("THOR_ROOT", "data/hssd20S2"); A3P = os.environ["A3_PREFIX"]; VJ = os.environ["VERIFY_JSONL"]
VTH = float(os.environ.get("VERIFY_TH", "0")); VTH2 = float(os.environ.get("VERIFY_TH2", "-1e9"))
vsc = {}
for l in open(VJ):
    d = json.loads(l); vsc.setdefault(d["house"], []).append(d)
out = {}; tot = 0
for hd in sorted(glob.glob(os.path.join(ROOT, "house_*"))):
    hn = os.path.basename(os.path.realpath(hd)); fa = A3P + hn + ".npz"
    if not os.path.exists(fa) or hn not in vsc: continue
    ts = np.load(fa, allow_pickle=True)["ts"]; sel = set()
    for d in vsc[hn]:
        sc = d["scored"]
        passed = [e for e in sc if e[1] >= VTH and (len(e) < 3 or e[2] >= VTH2)]
        top = sorted(sc, key=lambda e: -e[1])[:a.top]
        for e in passed + top:
            i = int(e[0])
            if 0 <= i < len(ts): sel.add(int(ts[i]))
    out[hn] = sorted(sel); tot += len(sel)
    print("%s 질의 프레임 %d (타겟 %d)" % (hn, len(sel), len(vsc[hn])), flush=True)
json.dump(out, open(a.out, "w")); print("→ %s · 채 %d · 프레임 %d" % (a.out, len(out), tot))
