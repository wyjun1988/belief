#!/usr/bin/env bash
# stand 프레임의 채택 판정 마진을 **새 상자(별칭 캐시)로** 다시 매긴다 (2026-09-23).
# 왜: 마진은 (집, 물체, 프레임 t) 로 키가 걸려 있어 상자가 바뀌어도 옛 상자로 매긴 값이 그대로 쓰였다.
#     별칭은 stand 상자를 옮기므로 stand 프레임 마진이 전부 낡았다(c3big 은 "마진 필요 0" 으로 보였지만 실은 낡은 값).
set -u; cd ~/work/khronos; export KMP_DUPLICATE_LIB_OK=TRUE
K=$HOME/kx-venv/bin/python; MLX=$HOME/mlx-venv/bin/python; AD=$HOME/khcache/h100_0917/out/lora_adopt_all_full_4b
one(){ nm=$1; VF=$2; A3=$3; ROOT=$4; OLDM=$5; OUTD=$6
  echo "=== $nm: stand 행 추출 $(date +%H:%M) ==="
  $K - "$VF" "$OUTD/t1_stand_only.jsonl" <<'PY'
import json, sys
n=0; nf=0
with open(sys.argv[2],"w") as fo:
    for l in open(sys.argv[1]):
        d=json.loads(l)
        if d["oid"].split("|")[0]=="stand":
            d["scored"]=[e for e in d["scored"] if e[1]>=2.069 and e[2]>=0.887]
            if d["scored"]: fo.write(json.dumps(d)+"\n"); n+=1; nf+=len(d["scored"])
print("  stand 물체 %d · 제로샷 통과 프레임 %d" % (n, nf))
PY
  BACKEND=hf MODEL=Qwen/Qwen3.5-4B ADAPTER=$AD DEVICE=mps USE_CTX=0 VERIFY_JSONL=$OUTD/t1_stand_only.jsonl A3_PREFIX=$A3 THOR_ROOT=$ROOT \
    OUT_JSONL=$OUTD/t1_stand_only_out.jsonl VERDICT_JSONL=$OUTD/margin_stand_newbox.jsonl $MLX -u scripts/lora_adopt_infer.py 2>&1 | grep -E "ADOPT_INFER_DONE" | cut -c1-120
  cat $OLDM $OUTD/margin_stand_newbox.jsonl > $OUTD/margin_all_standfresh.jsonl      # 뒤에 온 값(새 상자)이 이긴다
  echo "  마진 $(wc -l < $OUTD/margin_all_standfresh.jsonl)줄"; }
V=$HOME/khcache/bench-v2full; VA=$HOME/khcache/bench-v2alias; C3=$HOME/khcache/bench-c3big; CA=$HOME/khcache/bench-c3big-alias
one HSSD $VA/scores/t1_floor0.8_d40.jsonl $VA/cache/hs2_a_ data/hssd_v2 $VA/scores/adopt_margin_all_full_plusgap.jsonl $VA/scores
$K scripts/apply_adopt_margin.py $VA/scores/margin_all_standfresh.jsonl 0 $VA/scores/t1_champ_allfull_v3.jsonl --verify $VA/scores/t1_floor0.8_d40.jsonl --cache-prefix $VA/cache/hs2_a_ --missing drop | tail -1 | cut -c1-120
one c3big $CA/scores/t1_all_alias.jsonl $CA/cache/hs2_a_ data/hssd_c3big $CA/scores/margin_all_alias.jsonl $CA/scores
$K scripts/apply_adopt_margin.py $CA/scores/margin_all_standfresh.jsonl 0 $CA/scores/filter_alias_v2.jsonl --verify $CA/scores/t1_all_alias.jsonl --cache-prefix $CA/cache/hs2_a_ --missing drop | tail -1 | cut -c1-120
echo "=== 벤치 $(date +%H:%M) ==="
source scripts/champion_bench.sh
run_hssd_old HSSD_옛챔피언
run_hssd HSSD_병합_낡은마진
run_hssd HSSD_병합_새마진 VERIFY_JSONL=$VA/scores/t1_champ_allfull_v3.jsonl
run_c3big c3_옛_allfull VERIFY_JSONL=$CA/scores/filter_old.jsonl
run_c3big c3_별칭_새마진 VERIFY_JSONL=$CA/scores/filter_alias_v2.jsonl INITMAP_FILE=initmap_owl_merge.json A3_PREFIX=$CA/cache/hs2_a_
echo "STAND_REFRESH_DONE $(date +%H:%M)"
