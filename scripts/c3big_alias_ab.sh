#!/usr/bin/env bash
# c3big 별칭 A/B (9-65) — 옛 M2 캐시에 stand 열만 이식해 백엔드 차(CUDA↔MPS)를 배제하고, 두 팔을 같은 판정기(all_full)로 비교한다.
set -u; cd ~/work/khronos; export KMP_DUPLICATE_LIB_OK=TRUE
K=$HOME/kx-venv/bin/python; MLX=$HOME/mlx-venv/bin/python
C3=$HOME/khcache/bench-c3big; CA=$HOME/khcache/bench-c3big-alias; R=$HOME/khcache/rtx_9_65
mkdir -p $CA/cache $CA/scores
echo "=== 1. 캐시 이식 (옛 캐시 + stand 열) $(date +%H:%M) ==="
$K - <<'PY'
import numpy as np, os, glob
R=os.path.expanduser("~/khcache/rtx_9_65/cache"); O=os.path.expanduser("~/khcache/bench-c3big/cache"); A=os.path.expanduser("~/khcache/bench-c3big-alias/cache")
n=0
for f in sorted(glob.glob(O+"/hs2_a_*.npz")):
    h=os.path.basename(f); b=dict(np.load(f, allow_pickle=True)); a=np.load(R+"/"+h, allow_pickle=True)
    v=list(b["vocab"]); ti=v.index("stand"); nT=int(b["nT"])
    b["s"]=b["s"].copy(); b["s"][:,ti]=a["s"][:,ti]
    b["p"]=b["p"].copy(); b["p"][:,ti]=a["p"][:,ti]
    if "bx" in b and ti < nT: b["bx"]=b["bx"].copy(); b["bx"][:,ti]=a["bx"][:,ti]
    np.savez_compressed(A+"/"+h, **b); n+=1
print("  이식 %d채 (stand 열만)" % n)
PY
for f in $C3/cache/hs2_q_*.npz $C3/cache/hs2_x_*.npz; do ln -sf $f $CA/cache/; done
echo "=== 2. stand 행만 다시 검증 (MLX) $(date +%H:%M) ==="
rm -f $CA/scores/t1_stand.jsonl
THOR_ROOT=data/hssd_c3big A3_PREFIX=$CA/cache/hs2_a_ QC_PREFIX=$C3/cache/hs2_q_ ALL_TARGETS=1 FLOOR=0.8 MAXWALK=40 ONLY_TYPES=stand \
  OUT_JSONL=$CA/scores/t1_stand.jsonl $MLX -u scripts/exp_t1_verify_mlx.py > $CA/verify_stand.log 2>&1
echo "  stand 행 $(wc -l < $CA/scores/t1_stand.jsonl)"
$K - <<'PY'
import json, os
C3=os.path.expanduser("~/khcache/bench-c3big/scores"); CA=os.path.expanduser("~/khcache/bench-c3big-alias/scores")
new={(d["house"],d["oid"]):l for l in open(CA+"/t1_stand.jsonl") for d in [json.loads(l)]}
n=0
with open(CA+"/t1_all_alias.jsonl","w") as fo:
    for l in open(C3+"/t1_all.jsonl"):
        d=json.loads(l); k=(d["house"],d["oid"])
        if k in new: fo.write(new[k] if new[k].endswith("\n") else new[k]+"\n"); n+=1
        else: fo.write(l)
print("  검증 파일 병합: stand %d행 교체" % n)
PY
echo "=== 3. 새 stand 프레임의 판정기 마진 (all_full · M2) $(date +%H:%M) ==="
$K - <<'PY'
import json, os, numpy as np
CA=os.path.expanduser("~/khcache/bench-c3big-alias"); M=os.path.expanduser("~/khcache/rtx_9_55/adopt_margin_all_full_c3big_ALL_0921.jsonl")
mg=set((d["house"],d["oid"],int(d["t"])) for d in (json.loads(l) for l in open(M)))
Z={}; n=0; nf=0
with open(CA+"/scores/t1_stand_gap.jsonl","w") as fo:
    for l in open(CA+"/scores/t1_stand.jsonl"):
        d=json.loads(l); h=d["house"]
        if h not in Z: Z[h]=np.load(f"{CA}/cache/hs2_a_{h}.npz", allow_pickle=True)["ts"]
        keep=[e for e in d["scored"] if e[1]>=2.069 and e[2]>=0.887 and (h,d["oid"],int(Z[h][int(e[0])])) not in mg]
        if keep: d["scored"]=keep; fo.write(json.dumps(d)+"\n"); n+=1; nf+=len(keep)
print("  마진 필요한 프레임 %d (물체 %d)" % (nf, n))
PY
if [ -s $CA/scores/t1_stand_gap.jsonl ]; then
  BACKEND=hf MODEL=Qwen/Qwen3.5-4B ADAPTER=$HOME/khcache/h100_0917/out/lora_adopt_all_full_4b DEVICE=mps USE_CTX=0 \
    VERIFY_JSONL=$CA/scores/t1_stand_gap.jsonl A3_PREFIX=$CA/cache/hs2_a_ THOR_ROOT=data/hssd_c3big \
    OUT_JSONL=$CA/scores/t1_stand_gap_out.jsonl VERDICT_JSONL=$CA/scores/margin_stand_gap.jsonl $MLX -u scripts/lora_adopt_infer.py > $CA/margin_gap.log 2>&1
fi
cat $HOME/khcache/rtx_9_55/adopt_margin_all_full_c3big_ALL_0921.jsonl $CA/scores/margin_stand_gap.jsonl 2>/dev/null > $CA/scores/margin_all_alias.jsonl
echo "  마진 $(wc -l < $CA/scores/margin_all_alias.jsonl)줄"
echo "=== 4. 두 팔 필터 (같은 판정기 · 마진 없으면 버림) $(date +%H:%M) ==="
$K scripts/apply_adopt_margin.py $HOME/khcache/rtx_9_55/adopt_margin_all_full_c3big_ALL_0921.jsonl 0 $CA/scores/filter_old.jsonl --verify $C3/scores/t1_all.jsonl --cache-prefix $C3/cache/hs2_a_ --missing drop | tail -2 | cut -c1-140
$K scripts/apply_adopt_margin.py $CA/scores/margin_all_alias.jsonl 0 $CA/scores/filter_alias.jsonl --verify $CA/scores/t1_all_alias.jsonl --cache-prefix $CA/cache/hs2_a_ --missing drop | tail -2 | cut -c1-140
echo "=== 5. 초기맵 이식 (챔피언 initmap_owl + stand 만 별칭판) $(date +%H:%M) ==="
$K - <<'PY'
import json, glob, os
R=os.path.expanduser("~/khcache/rtx_9_65"); n=0; ns=0
for hd in sorted(glob.glob("data/hssd_c3big/house_*")):
    hn=os.path.basename(hd); a=os.path.join(hd,"initmap_owl.json"); b=os.path.join(R,hn,"initmap_owl_alias.json")
    if not (os.path.exists(a) and os.path.exists(b)): continue
    A=json.load(open(a)); B=json.load(open(b)); add=[e for e in B if e["type"]=="stand"]
    json.dump([e for e in A if e["type"]!="stand"]+add, open(os.path.join(hd,"initmap_owl_merge.json"),"w")); n+=1; ns+=len(add)
print("  병합 초기맵 %d채 · stand %d개" % (n, ns))
PY
echo "=== 6. 벤치 $(date +%H:%M) ==="
source scripts/champion_bench.sh
run_c3big c3_옛_allfull VERIFY_JSONL=$CA/scores/filter_old.jsonl
run_c3big c3_별칭_allfull VERIFY_JSONL=$CA/scores/filter_alias.jsonl INITMAP_FILE=initmap_owl_merge.json A3_PREFIX=$CA/cache/hs2_a_
run_c3big c3_챔피언
echo "C3_ALIAS_DONE $(date +%H:%M)"
