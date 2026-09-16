#!/usr/bin/env bash
# 프로 6000 저녁 배치 (9-45, 2026-09-16). 재시작 가능 — 산출이 있으면 건너뜀. 두 GPU 병렬.
#   cd ~/khronos && git pull && bash scripts/rtx_batch_20260916.sh
# 질문 셋: (A) LoRA 를 모델 전체(비전 포함)에 붙이면 얼마나 오르나  (B) ② 중심 데이터(c2set)를 더하면 이동 후 판정이 살아나나
#          (C) 부재 판정도 full 로 오르나.  기준선: 채택 혼합 narrow 0.918(HSSD)/0.837(OG) · 기록자리 narrow+balance 0.810 · 부재 혼합 0.961
set -u; cd "$(dirname "$0")/.."
KH=${KH:-$HOME/khcache}; DRIVE=${DRIVE:-$HOME}; M4=Qwen/Qwen3.5-4B; G0=${G0:-0}; G1=${G1:-1}
SUM=$KH/rtx_batch_summary_0916.txt; : > $SUM; log() { echo "[$(date +%H:%M)] $*" | tee -a $SUM; }
step() { log "=== $1 ==="; }
unz() { [ -d "$KH/$1" ] && return; for z in "$DRIVE/$1.zip" "packs/$1.zip"; do [ -f "$z" ] && { unzip -q "$z" -d "$KH/" && log "풀림 $1"; return; }; done; log "⚠ $1.zip 없음"; }
export PYTHONUNBUFFERED=1
step "0. 재료"; for z in lora_adopt_v2 lora_adopt_og lora_adopt_c2 lora_place_v2 lora_presence_v2 lora_presence_og; do unz $z; done
[ -d $KH/lora_adopt_c2 ] || { [ -f "$DRIVE/lora_adopt_c2.zip" ] && unzip -q "$DRIVE/lora_adopt_c2.zip" -d $KH/ && log "풀림 lora_adopt_c2"; }
python -c "import peft" 2>/dev/null || pip install -q peft
# 합본: 채택 v2+og+c2 (이동 후 표본 강화), 부재 v2+og
mk() { D=$KH/$1; shift; mkdir -p $D/images; : > $D/train.jsonl; : > $D/val.jsonl
  for s in "$@"; do [ -d $KH/$s ] || { log "⚠ $s 없음 → 합본에서 빠짐"; continue; }
    ln -sfn $KH/$s/images/* $D/images/ 2>/dev/null; cat $KH/$s/train.jsonl >> $D/train.jsonl; cat $KH/$s/val.jsonl >> $D/val.jsonl; done
  log "$1: train $(wc -l < $D/train.jsonl) · val $(wc -l < $D/val.jsonl)"; }
mk lora_adopt_all lora_adopt_v2 lora_adopt_og lora_adopt_c2
mk lora_presence_mix2 lora_presence_v2 lora_presence_og
# 사전 점검: 트레이너 argparse 가 살아 있나 (9-35~9-44 사이 HEAD 가 즉사하던 버그)
python scripts/lora_presence_train.py --help 2>&1 | grep -q -- "--targets" || { log "✗ 트레이너에 --targets 없음 — git pull 이 안 됐다"; exit 1; }

train() { # train <GPU> <task> <data> <out> [추가인자...] — 산출 있으면 건너뜀, 로그는 통째로 파일에
  G=$1; T=$2; D=$3; O=$4; shift 4; [ -s $O/adapter_model.safetensors ] && { log "skip $O"; return; }
  CUDA_VISIBLE_DEVICES=$G python scripts/lora_presence_train.py --task $T --data $D --model $M4 --out $O --epochs 1 --eval-every 200 "$@" > $O.log 2>&1
  grep -aE "trainable params|EVAL\[final\]|LORA_TRAIN_DONE|Traceback|Error" $O.log | tail -4 | sed "s|^|  [$(basename $O)] |" | tee -a $SUM; }
step "1. (A) 기록자리 · 채택 — targets llm / full   [GPU$G0]  (B) 채택 all-data narrow/full  [GPU$G1]"
( train $G0 place  $KH/lora_place_v2  $KH/lora_place_llm_4b   --balance --targets llm  --val-max 1301
  train $G0 place  $KH/lora_place_v2  $KH/lora_place_full_4b  --balance --targets full --val-max 1301
  train $G0 adopt  $KH/lora_adopt_v2  $KH/lora_adopt_full_4b  --targets full --val-max 863 ) &
( train $G1 adopt  $KH/lora_adopt_all $KH/lora_adopt_all_4b      --targets narrow --val-max 2000
  train $G1 adopt  $KH/lora_adopt_all $KH/lora_adopt_all_full_4b --targets full   --val-max 2000
  train $G1 presence $KH/lora_presence_mix2 $KH/lora_pres_full_4b --targets full --val-max 1103 ) &
wait
step "2. 도메인별 채점 — 채택 어댑터 4종 × val 3종 (HSSD / OG / c2set)"
for A in lora_adopt_mix_4b lora_adopt_full_4b lora_adopt_all_4b lora_adopt_all_full_4b; do
  [ -s $KH/$A/adapter_model.safetensors ] || { log "skip eval $A"; continue; }
  for D in lora_adopt_v2 lora_adopt_og lora_adopt_c2; do [ -d $KH/$D ] || continue
    CUDA_VISIBLE_DEVICES=$G0 python scripts/lora_presence_train.py --task adopt --data $KH/$D --model $M4 --out /tmp/x --eval-only --adapter $KH/$A --val-max 2000 2>&1 \
      | grep -aE "EVAL\[val\] AUC|Traceback" | sed "s|^|  [$A → ${D#lora_adopt_}] |" | tee -a $SUM; done; done
step "3. 오라클 고정 잣대 — 부재 full 어댑터 (기준 4B 133채 0.897 · 9B 파일럿 0.897)"
[ -s $KH/lora_pres_full_4b/adapter_model.safetensors ] && [ -d $KH/oracle_pack_v2b ] && \
  CUDA_VISIBLE_DEVICES=$G0 BACKEND=hf MODEL=$M4 ADAPTER=$KH/lora_pres_full_4b SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$KH/oracle_presfull_B.jsonl \
  python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback" | tee -a $SUM
step "4. 묶기"
tar czf $KH/rtx_0916_results.tar.gz -C $KH rtx_batch_summary_0916.txt $(cd $KH && ls -d lora_place_llm_4b lora_place_full_4b lora_adopt_full_4b lora_adopt_all_4b lora_adopt_all_full_4b lora_pres_full_4b *.log oracle_presfull_B.jsonl 2>/dev/null) 2>/dev/null
log "완료 → $KH/rtx_0916_results.tar.gz 를 Drive 에"
