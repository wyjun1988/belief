#!/usr/bin/env bash
# RTX PRO 6000 일괄 실행 (2026-09-13) — 밀린 과제를 한 번에: 9-24 ② 재식별(9B) → 9-23 오라클 27B → 9-29 LoRA(4B, 이어서 9B) + 학습 뒤 오라클 → 9-28(a) OG 같은 방 보강 PnP
# 단계마다 산출물이 있으면 건너뛴다(재실행 안전). 실패해도 다음 단계로 간다. 끝나면 요약 + 결과 tar 를 만든다.
#   cd ~/work/khronos && git pull && nohup bash scripts/rtx_batch_20260913.sh > ~/rtx_batch.log 2>&1 &
#   DRY=1 이면 각 학습·프로브를 소량(MAX_OBJ/--max-steps)으로만 돌려 파이프라인을 점검한다.
set -u; cd "$(dirname "$0")/.."
KH=${KH:-$HOME/khcache}; DRIVE=${DRIVE:-$HOME}                     # KH: 묶음을 푸는 곳 · DRIVE: Drive 에서 받은 zip 이 있는 곳
OG_ROOT=${OG_ROOT:-/mnt/ssd2/wooyeol/work/og4v}; OG_BENCH=${OG_BENCH:-$HOME/khcache/bench-og24_th006_20260910}
M4=${M4:-Qwen/Qwen3.5-4B}; M9=${M9:-Qwen/Qwen3.5-9B}; M27=${M27:-Qwen/Qwen3.5-27B}; DRY=${DRY:-0}
SUM=$KH/rtx_batch_summary_$(date +%m%d).txt; : > $SUM; log() { echo "[$(date +%H:%M)] $*" | tee -a $SUM; }
step() { log "=== $1 ==="; }
unz() { [ -d "$KH/$1" ] || { [ -f "$DRIVE/$1.zip" ] && unzip -q "$DRIVE/$1.zip" -d "$KH/" && log "풀림 $1" || log "⚠ $1.zip 없음 ($DRIVE)"; }; }
export PYTHONUNBUFFERED=1
step "0. 재료"; for z in oracle_pack_v2b reid_pack_v2b lora_pack_pilot lora_presence_v2; do unz $z; done
[ -d "$KH/lora_presence_v2" ] || { [ -f "$DRIVE/lora_pack_v2.zip" ] && unzip -q "$DRIVE/lora_pack_v2.zip" -d "$KH/" && log "풀림 lora_pack_v2 → lora_presence_v2"; }
python -c "import peft" 2>/dev/null || pip install -q peft && log "peft OK"
DRYO=""; [ "$DRY" = 1 ] && DRYO="MAX_OBJ=5"

step "1. 9-24 ② 인스턴스 재식별 9B (서문 1/0)"
for P in 1 0; do o=$KH/reid_9b_pre$P.jsonl; [ -s $o ] && { log "skip $o"; continue; }
  env $DRYO BACKEND=hf MODEL=$M9 SELECT_IN=$KH/reid_pack_v2b PREAMBLE=$P OUT_JSONL=$o python scripts/oracle_reid_probe.py 2>&1 | grep -aE "REID_DONE|Traceback|Error" | tail -2 | tee -a $SUM; done

step "2. 9-23 오라클 B 27B"
if python - <<PY 2>/dev/null
import sys; from huggingface_hub import scan_cache_dir
ok = any("$M27".split('/')[-1] in r.repo_id for r in scan_cache_dir().repos); sys.exit(0 if ok else 1)
PY
then o=$KH/oracle_27b_B.jsonl; [ -s $o ] && log "skip $o" || env $DRYO BACKEND=hf MODEL=$M27 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM
else log "27B 캐시 없음 → 건너뜀 (M27=$M27)"; fi

step "3. 9-29 LoRA 자리 존재 판정 — 4B"
D=$KH/lora_presence_v2; [ "$DRY" = 1 ] && D=$KH/lora_pack_pilot
if [ -d $D ]; then
  A4=$KH/lora_presence_4b; MS=""; [ "$DRY" = 1 ] && MS="--max-steps 20 --eval-every 10"
  [ -f $A4/adapter_config.json ] && log "skip 학습(어댑터 있음) $A4" || python scripts/lora_presence_train.py --data $D --model $M4 --out $A4 --epochs 2 --eval-every 200 --val-max 300 $MS 2>&1 | grep -aE "^EVAL|LORA_TRAIN_DONE|Traceback|trainable" | tee -a $SUM
  o=$KH/oracle_4b_lora_B.jsonl; [ -s $o ] && log "skip $o" || env $DRYO BACKEND=hf MODEL=$M4 ADAPTER=$A4 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM
  o=$KH/oracle_4b_zs_B.jsonl; [ -s $o ] && log "skip $o" || env $DRYO BACKEND=hf MODEL=$M4 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM   # 같은 4B 제로샷(대조)
  step "3b. LoRA 9B (1 에폭)"
  A9=$KH/lora_presence_9b; [ -f $A9/adapter_config.json ] && log "skip $A9" || python scripts/lora_presence_train.py --data $D --model $M9 --out $A9 --epochs 1 --eval-every 200 --val-max 300 $MS 2>&1 | grep -aE "^EVAL|LORA_TRAIN_DONE|Traceback|trainable" | tee -a $SUM
  o=$KH/oracle_9b_lora_B.jsonl; [ -s $o ] && log "skip $o" || env $DRYO BACKEND=hf MODEL=$M9 ADAPTER=$A9 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM
else log "⚠ 학습셋 없음 $D"; fi

step "4. 9-28(a) OG 24채 같은 방 보강 PnP (인라이어 50)"
if [ -d "$OG_ROOT" ] && [ -d "$OG_BENCH" ]; then
  [ -s $OG_BENCH/pnp/pose_all_room.jsonl ] && log "skip pose_all_room" || OUT=$OG_ROOT BENCH_DIR=$OG_BENCH MIRROR=0 PAR=3 bash scripts/og_pnp_room.sh 2>&1 | grep -aE "POSE_CHECK|POSE_JSONL|Traceback" | tee -a $SUM
else log "⚠ OG 경로 없음 (OG_ROOT=$OG_ROOT OG_BENCH=$OG_BENCH) — 환경변수로 지정"; fi

step "5. 요약·묶기"
tar -czf $KH/rtx_batch_$(date +%m%d)_results.tar.gz -C $KH $(cd $KH && ls reid_9b_pre*.jsonl oracle_27b_B.jsonl oracle_4b_lora_B.jsonl oracle_4b_zs_B.jsonl oracle_9b_lora_B.jsonl 2>/dev/null) $(cd $KH && ls -d lora_presence_4b lora_presence_9b 2>/dev/null) rtx_batch_summary_$(date +%m%d).txt 2>/dev/null && log "결과 tar: $KH/rtx_batch_$(date +%m%d)_results.tar.gz (Drive 에 올릴 것)"
log "RTX_BATCH_DONE"; echo; cat $SUM
