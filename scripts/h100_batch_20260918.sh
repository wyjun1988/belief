#!/usr/bin/env bash
# H100 9-50 (2026-09-18) — **효과를 하나씩 분리하는** 설계.
#   cd ~/khronos && git pull && bash scripts/h100_batch_20260918.sh
#
# 규칙 3개 (이게 지켜져야 표를 읽을 수 있다):
#   1) 각 판은 기준(B0)에서 **딱 한 가지만** 바꾼다. 조합은 맨 뒤에 따로, 이름에 표시한다.
#   2) 모든 판을 **같은 고정 검증셋**(--val-data $KH/val_fixed, 1,787행)으로 채점한다.
#      판마다 학습 데이터가 달라 자기 val 로 재면 숫자를 비교할 수 없다(9-48 에서 실제로 그랬다).
#   3) 씨앗 고정, 1 에폭 고정 — 다르게 하는 판은 그게 그 판의 변수다.
set -u; cd "$(dirname "$0")/.."
KH=${KH:-$HOME/khcache}; DRIVE=${DRIVE:-$HOME}; M=Qwen/Qwen3.5-4B
G0=${G0:-0}; G1=${G1:-1}; OUT=$KH/out_9_50; mkdir -p $OUT
SUM=$KH/h100_summary_0918.txt; : > $SUM
log(){ echo "[$(date +%H:%M)] $*" | tee -a $SUM; }
export PYTHONUNBUFFERED=1
log "=== 0. 재료 ==="
for Z in lora_adopt_real val_fixed lora_adopt_hn lora_adopt_c2 lora_adopt_v2 lora_presence_v2 adopt_infer_v2; do
  [ -d $KH/$Z ] && continue
  for p in "$DRIVE/$Z.zip" "$DRIVE/${Z}_0917.zip" packs/$Z.zip; do [ -f "$p" ] && { unzip -q "$p" -d $KH/ && log "풀림 $Z"; break; }; done
  [ -d $KH/$Z ] || log "⚠ $Z 없음"
done
python scripts/lora_presence_train.py --help 2>&1 | grep -q -- "--val-data" || { log "✗ git pull 안 됨 (--val-data 없음)"; exit 1; }
log "=== 0.2 OmniGibson 학습셋 (없으면 만든다 — 9-39/40 에서 이 기계에 만들었던 것) ==="
if [ ! -d $KH/lora_adopt_og ] && [ -d data/hssd_og ]; then
  python scripts/lora_adopt_data.py data/hssd_og $KH/lora_adopt_og --val-houses 6 2>&1 | tail -1 | tee -a $SUM
fi
if [ ! -d $KH/lora_presence_og ] && [ -d data/hssd_og ]; then
  python scripts/lora_presence_data.py data/hssd_og $KH/lora_presence_og --k 4 --per-target 3 --val-houses 6 2>&1 | tail -1 | tee -a $SUM
fi
for Z in lora_adopt_og lora_presence_og; do
  [ -d $KH/$Z ] || log "ℹ $Z 없음 — **그대로 진행한다**. 모든 판이 똑같이 빠지므로 판 사이 비교는 그대로 유효하다(기준선 절대값만 조금 내려간다)."
done
log "=== 0.5 학습셋 합치기 (집 단위 분리 — v2/c2 는 같은 장면이다) ==="
rm -rf $KH/lora_adopt_all $KH/lora_adopt_allhn $KH/lora_adopt_allreal $KH/lora_presence_mix2 $KH/lora_mt
python scripts/lora_merge_sets.py $KH/lora_adopt_all     $KH/lora_adopt_v2 $KH/lora_adopt_og $KH/lora_adopt_c2 2>&1 | tail -1 | tee -a $SUM
python scripts/lora_merge_sets.py $KH/lora_adopt_allhn   $KH/lora_adopt_v2 $KH/lora_adopt_og $KH/lora_adopt_c2 $KH/lora_adopt_hn 2>&1 | tail -1 | tee -a $SUM
python scripts/lora_merge_sets.py $KH/lora_adopt_allreal $KH/lora_adopt_v2 $KH/lora_adopt_og $KH/lora_adopt_c2 $KH/lora_adopt_real 2>&1 | tail -1 | tee -a $SUM
python scripts/lora_merge_sets.py $KH/lora_presence_mix2 $KH/lora_presence_v2 $KH/lora_presence_og 2>&1 | tail -1 | tee -a $SUM
# belief 는 이번 배치에서 뺀다 — 새 과제와 새 데이터를 한꺼번에 넣으면 무엇이 효과인지 못 가른다(9-51 로 미룸).
python scripts/lora_multitask_merge.py $KH/lora_mt adopt=$KH/lora_adopt_allreal presence=$KH/lora_presence_mix2 2>&1 | tail -1 | tee -a $SUM

VF=$KH/val_fixed
run(){ # run <GPU> <이름> <과제> <데이터> [추가인자…]
  G=$1; NM=$2; T=$3; D=$4; shift 4; O=$OUT/$NM
  [ -s $O/adapter_model.safetensors ] && { log "skip $NM"; return; }
  [ -d $D ] || { log "⚠ 데이터 없음 $D → $NM 건너뜀"; return; }
  CUDA_VISIBLE_DEVICES=$G python scripts/lora_presence_train.py --task $T --data $D --val-data $VF \
    --model $M --out $O --epochs 1 --eval-every 200 --seed 0 "$@" > $OUT/$NM.log 2>&1
  grep -aE "trainable params|고정 검증셋|EVAL\[final\]|EVALGRP\[final\]|LORA_TRAIN_DONE|OutOfMemory|Traceback" $OUT/$NM.log | tail -4 | sed "s|^|  [$NM] |" | tee -a $SUM; }

log "=== 1. 한 가지씩만 바꾼 판 (효과 분리) ==="
#     이름            과제   데이터                       바꾼 것 하나
( run $G0 A0_base      adopt $KH/lora_adopt_all     --targets narrow                    # 기준
  run $G0 A1_r32       adopt $KH/lora_adopt_all     --targets narrow --r 32 --alpha 64  # 용량
  run $G0 A2_hn        adopt $KH/lora_adopt_allhn   --targets narrow                    # 시뮬 어려운음성
  run $G0 A3_real      adopt $KH/lora_adopt_allreal --targets narrow                    # ★ 실사 추가
) > /dev/null 2>&1 &
( run $G1 A4_vision    adopt $KH/lora_adopt_all     --targets full --grad-ckpt          # 비전 타워
  run $G1 A5_lr3       adopt $KH/lora_adopt_all     --targets narrow --lr 3e-5          # 학습률
  run $G1 P0_presence  presence $KH/lora_presence_mix2 --targets narrow                 # 부재 단독 기준
) > /dev/null 2>&1 &
wait
log "=== 2. 조합 (1단계에서 이득 본 것만 합친 판) ==="
( run $G0 C1_real_r32  adopt $KH/lora_adopt_allreal --targets narrow --r 32 --alpha 64
  run $G0 C2_multi     multi $KH/lora_mt            --targets narrow                    # 채택+부재 한 어댑터 (과제 합본 효과만)
) > /dev/null 2>&1 &
( run $G1 C3_multi_r32 multi $KH/lora_mt            --targets narrow --r 32 --alpha 64   # 채택+부재 (belief 없음)
  run $G1 C4_real_vis  adopt $KH/lora_adopt_allreal --targets full --grad-ckpt
) > /dev/null 2>&1 &
wait
log "=== 3. 133채 실측 추론 (최종 판정 재료 — M2 는 어댑터당 3.5시간이라 못 돈다) ==="
for NM in A0_base A1_r32 A2_hn A3_real A4_vision A5_lr3 C1_real_r32 C2_multi C3_multi_r32 C4_real_vis; do
  A=$OUT/$NM; V=$KH/adopt_margin_$NM.jsonl
  [ -s $A/adapter_model.safetensors ] || continue
  [ -s $V ] && { log "skip 추론 $NM"; continue; }
  PACK=$KH/adopt_infer_v2 BACKEND=hf DEVICE=cuda MODEL=$M ADAPTER=$A VERDICT_JSONL=$V VERIFY_JSONL=/dev/null A3_PREFIX=/tmp/ \
    python scripts/lora_adopt_infer.py > $OUT/infer_$NM.log 2>&1
  log "추론 $NM → $(wc -l < $V 2>/dev/null || echo 0)건"; done
log "=== 4. 묶기 ==="
cd $KH && tar czf h100_9_50_results.tar.gz h100_summary_0918.txt adopt_margin_*.jsonl out_9_50/ 2>/dev/null
log "완료 → $KH/h100_9_50_results.tar.gz (요약 + 마진 + 어댑터 + **로그 전부**)"
