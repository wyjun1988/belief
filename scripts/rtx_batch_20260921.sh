#!/usr/bin/env bash
# 9-51 (2026-09-21) RTX PRO 6000 — H100 볼륨 복구 전까지 9-50 의 핵심 3판을 여기서 다시 학습하고 133채 마진까지.
#   cd ~/khronos && git pull && bash scripts/rtx_batch_20260921.sh
# 순서(사용자 우선순위): A3_real → A1_r32 → A4_vision. 판 하나 끝날 때마다 rtx_9_51_<판>.tar.gz 를 만든다 → 그때그때 Drive 에.
# 원칙: set -e 없음 · 산출 있으면 건너뜀 · 로그 통째로 · 모든 판은 고정 검증셋(val_fixed)으로 채점.
set -u; cd "$(dirname "$0")/.."
KH=${KH:-$HOME/khcache}; DRIVE=${DRIVE:-$HOME}; M=Qwen/Qwen3.5-4B; OUT=$KH/out_9_51; mkdir -p $OUT
SUM=$KH/rtx_summary_0921.txt; : > $SUM
log(){ echo "[$(date +%H:%M)] $*" | tee -a $SUM; }
export PYTHONUNBUFFERED=1
log "=== 0. 재료 ==="
if [ ! -d $KH/lora_adopt_real ] || [ ! -d $KH/val_fixed ] || [ ! -d $KH/adopt_infer_v2 ]; then
  T=$(ls "$DRIVE"/h100_data_20260918.tar.gz packs/h100_data_20260918.tar.gz 2>/dev/null | head -1)
  [ -n "$T" ] && { tar xzf "$T" -C $KH/ && log "풀림 $(basename $T) (7개 셋)"; } || log "✗ h100_data_20260918.tar.gz 없음 — Drive 에서 받아 ~/ 에 두고 다시"
fi
for D in lora_adopt_v2 lora_adopt_c2 lora_adopt_real lora_presence_v2 val_fixed adopt_infer_v2; do [ -d $KH/$D ] && log "  있음 $D ($(wc -l < $KH/$D/train.jsonl 2>/dev/null || echo -)행)" || log "  ✗ 없음 $D"; done
# OmniGibson 셋: 9-39/40 에서 이 RTX 에 만들었던 것. 있으면 H100 의 A0_base(이미 133채 실측 있음)와 같은 재료라 그 기준선을 그대로 쓴다.
OG=1; for D in lora_adopt_og; do [ -d $KH/$D ] && log "  있음 $D" || { OG=0; log "  ✗ 없음 $D"; }; done
if [ $OG = 0 ] && [ -d data/hssd_og ]; then python scripts/lora_adopt_data.py data/hssd_og $KH/lora_adopt_og --val-houses 6 2>&1 | tail -1 | tee -a $SUM; [ -d $KH/lora_adopt_og ] && OG=1; fi
[ $OG = 1 ] || log "ℹ OG 셋 없음 — v2+c2(+real) 로 진행. 이 경우 H100 A0_base 와 재료가 달라지므로 **A0_base 도 여기서 다시 학습**해 대조한다."
python scripts/lora_presence_train.py --help 2>&1 | grep -q -- "--val-data" || { log "✗ git pull 안 됨 (--val-data 없음)"; exit 1; }
log "=== 0.5 합본 (집 단위 누수 차단) ==="
OGS=""; [ $OG = 1 ] && OGS="$KH/lora_adopt_og"
rm -rf $KH/lora_adopt_all $KH/lora_adopt_allreal
python scripts/lora_merge_sets.py $KH/lora_adopt_all     $KH/lora_adopt_v2 $OGS $KH/lora_adopt_c2 2>&1 | tail -1 | tee -a $SUM
python scripts/lora_merge_sets.py $KH/lora_adopt_allreal $KH/lora_adopt_v2 $OGS $KH/lora_adopt_c2 $KH/lora_adopt_real 2>&1 | tail -1 | tee -a $SUM
VF=$KH/val_fixed
train(){ NM=$1; D=$2; shift 2; O=$OUT/$NM
  [ -s $O/adapter_model.safetensors ] && { log "skip 학습 $NM"; return; }
  python scripts/lora_presence_train.py --task adopt --data $D --val-data $VF --model $M --out $O --epochs 1 --eval-every 200 --seed 0 "$@" > $OUT/$NM.log 2>&1
  grep -aE "trainable params|고정 검증셋|EVALGRP\[final\]|LORA_TRAIN_DONE|OutOfMemory|Traceback" $OUT/$NM.log | tail -4 | sed "s|^|  [$NM] |" | tee -a $SUM; }
margin(){ NM=$1; A=$OUT/$NM; V=$KH/adopt_margin_$NM.jsonl
  [ -s $A/adapter_model.safetensors ] || { log "✗ $NM 어댑터 없음 — 마진 생략"; return; }
  [ -s $V ] && { log "skip 마진 $NM"; return; }
  PACK=$KH/adopt_infer_v2 BACKEND=hf DEVICE=cuda MODEL=$M ADAPTER=$A VERDICT_JSONL=$V VERIFY_JSONL=/dev/null A3_PREFIX=/tmp/ \
    python scripts/lora_adopt_infer.py > $OUT/infer_$NM.log 2>&1
  log "마진 $NM → $(wc -l < $V 2>/dev/null || echo 0)건 (6996 이어야)"; }
bundle(){ NM=$1; (cd $KH && tar czf rtx_9_51_$NM.tar.gz rtx_summary_0921.txt adopt_margin_$NM.jsonl out_9_51/$NM/ out_9_51/$NM.log out_9_51/infer_$NM.log 2>/dev/null) && log "★ 묶음 $KH/rtx_9_51_$NM.tar.gz → Drive 에 올려주세요"; }
log "=== 1. A3_real (★ 실사) ==="
train A3_real $KH/lora_adopt_allreal --targets narrow; margin A3_real; bundle A3_real
if [ $OG = 0 ]; then log "=== 1b. A0_base (OG 없어 대조군 재학습) ==="; train A0_base $KH/lora_adopt_all --targets narrow; margin A0_base; bundle A0_base; fi
log "=== 2. A1_r32 (용량) ==="
train A1_r32 $KH/lora_adopt_all --targets narrow --r 32 --alpha 64; margin A1_r32; bundle A1_r32
log "=== 3. A4_vision (비전 타워 · 96GB 안에 들게 grad-ckpt + img-max) ==="
train A4_vision $KH/lora_adopt_all --targets full --grad-ckpt --img-max 640; margin A4_vision; bundle A4_vision
log "=== 4. 여유가 있으면: C4_real_vis (실사+비전 — 챔피언 후계 후보) ==="
train C4_real_vis $KH/lora_adopt_allreal --targets full --grad-ckpt --img-max 640; margin C4_real_vis; bundle C4_real_vis
log "RTX_9_51_DONE"
