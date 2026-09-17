#!/usr/bin/env bash
# H100 밤샘 배치 (9-48, 2026-09-17). 자원 여유가 있어 **학습 8판 + 추론 전량**을 한 번에 돌린다.
#   cd ~/khronos && git pull && bash scripts/h100_batch_20260917.sh
# 원칙: set -e 없음(한 판이 죽어도 나머지 진행) · 산출 있으면 건너뜀 · 로그는 grep 없이 통째로 파일에.
# 목표: ② 채택률 병목(이동 후 참 유지 0.62)을 깨는 조합 찾기. 판정은 M2 가 133채 실측으로 한다.
set -u; cd "$(dirname "$0")/.."
KH=${KH:-$HOME/khcache}; DRIVE=${DRIVE:-$HOME}; M4=Qwen/Qwen3.5-4B; M9=Qwen/Qwen3.5-9B
G0=${G0:-0}; G1=${G1:-1}; OUT=$KH/out_0917; mkdir -p $OUT
SUM=$KH/h100_summary_0917.txt; : > $SUM
log(){ echo "[$(date +%H:%M)] $*" | tee -a $SUM; }
step(){ log "=== $1 ==="; }
export PYTHONUNBUFFERED=1
step "0. 재료"
unz(){ [ -d "$KH/$1" ] && return; for z in "$DRIVE/$1.zip" "$DRIVE/$2" packs/$1.zip; do [ -n "${z:-}" ] && [ -f "$z" ] && { unzip -q "$z" -d "$KH/" && log "풀림 $1"; return; }; done; log "⚠ $1 없음"; }
unz lora_adopt_hn ""; unz lora_adopt_c2 lora_adopt_c2_0917.zip; unz adopt_infer_v2 ""; unz oracle_pack_v2b ""
python -c "import peft" 2>/dev/null || pip install -q peft
mk(){ D=$KH/$1; shift; mkdir -p $D/images; : > $D/train.jsonl; : > $D/val.jsonl
  for s in "$@"; do [ -d $KH/$s ] || { log "⚠ $s 없음"; continue; }
    ln -sfn $KH/$s/images/* $D/images/ 2>/dev/null; cat $KH/$s/train.jsonl >> $D/train.jsonl; cat $KH/$s/val.jsonl >> $D/val.jsonl; done
  log "$1: train $(wc -l < $D/train.jsonl) · val $(wc -l < $D/val.jsonl)"; }
mk lora_adopt_all  lora_adopt_v2 lora_adopt_og lora_adopt_c2
mk lora_adopt_allhn lora_adopt_v2 lora_adopt_og lora_adopt_c2 lora_adopt_hn
mk lora_presence_mix2 lora_presence_v2 lora_presence_og
python scripts/lora_presence_train.py --help 2>&1 | grep -q -- "--targets" || { log "✗ git pull 안 됨 (--targets 없음)"; exit 1; }

train(){ # train <GPU> <task> <data> <이름> [인자…]
  G=$1; T=$2; D=$3; NM=$4; shift 4; O=$OUT/$NM
  [ -s $O/adapter_model.safetensors ] && { log "skip $NM"; return; }
  [ -d $D ] || { log "⚠ 데이터 없음 $D → $NM 건너뜀"; return; }
  CUDA_VISIBLE_DEVICES=$G python scripts/lora_presence_train.py --task $T --data $D --model $M4 --out $O --epochs 1 --eval-every 200 "$@" > $OUT/$NM.log 2>&1
  grep -aE "trainable params|EVAL\[final\]|LORA_TRAIN_DONE|OutOfMemory|Traceback" $OUT/$NM.log | tail -4 | sed "s|^|  [$NM] |" | tee -a $SUM; }

step "1. 채택 — ② 병목을 깨는 조합 8판 (GPU$G0 / GPU$G1 병렬)"
( train $G0 adopt $KH/lora_adopt_hn    adopt_hn          --targets narrow --val-max 1081
  train $G0 adopt $KH/lora_adopt_allhn adopt_allhn       --targets narrow --val-max 2400
  train $G0 adopt $KH/lora_adopt_v2    adopt_v2_lr3      --targets narrow --lr 3e-5 --eval-every 100 --val-max 863
  train $G0 adopt $KH/lora_adopt_allhn adopt_allhn_lr3   --targets narrow --lr 3e-5 --val-max 2400 ) > /dev/null 2>&1 &
( train $G1 adopt $KH/lora_adopt_allhn adopt_allhn_r32   --targets narrow --r 32 --alpha 64 --val-max 2400
  train $G1 adopt $KH/lora_adopt_allhn adopt_allhn_e2    --targets narrow --epochs 2 --val-max 2400
  train $G1 adopt $KH/lora_adopt_hn    adopt_hn_full     --targets full --grad-ckpt --val-max 1081
  train $G1 presence $KH/lora_presence_mix2 pres_mix_lr3 --lr 3e-5 --val-max 1103 ) > /dev/null 2>&1 &
wait
step "2. 채택 추론 — 133채 실측 묶음 6,996건, 어댑터별 마진 (M2 가 이 파일로 최종 점수를 낸다)"
for NM in adopt_hn adopt_allhn adopt_v2_lr3 adopt_allhn_lr3 adopt_allhn_r32 adopt_allhn_e2 adopt_hn_full; do
  A=$OUT/$NM; V=$KH/adopt_margin_$NM.jsonl
  [ -s $A/adapter_model.safetensors ] || continue
  [ -s $V ] && { log "skip 추론 $NM"; continue; }
  PACK=$KH/adopt_infer_v2 BACKEND=hf DEVICE=cuda MODEL=$M4 ADAPTER=$A VERDICT_JSONL=$V VERIFY_JSONL=/dev/null A3_PREFIX=/tmp/ \
    python scripts/lora_adopt_infer.py > $OUT/infer_$NM.log 2>&1
  log "추론 $NM → $(wc -l < $V 2>/dev/null || echo 0)건"; done
step "3. 부재 — 오라클 고정 잣대 87건 (기준 0.897)"
for NM in pres_mix_lr3; do
  A=$OUT/$NM; [ -s $A/adapter_model.safetensors ] || continue
  BACKEND=hf MODEL=$M4 ADAPTER=$A SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$KH/oracle_$NM.jsonl \
    python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback" | sed "s|^|  [$NM] |" | tee -a $SUM; done
step "4. 묶기"
cd $KH && tar czf h100_0917_results.tar.gz h100_summary_0917.txt adopt_margin_*.jsonl oracle_*.jsonl \
  $(cd $KH && ls -d out_0917/*/ 2>/dev/null | sed 's|/$||') out_0917/*.log 2>/dev/null
log "완료 → $KH/h100_0917_results.tar.gz (어댑터 + 마진 + 로그) 를 Drive 에"
