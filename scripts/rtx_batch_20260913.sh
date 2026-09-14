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
unz() { [ -d "$KH/$1" ] && return; for z in "$DRIVE/$1.zip" "packs/$1.zip"; do [ -f "$z" ] && { unzip -q "$z" -d "$KH/" && log "풀림 $1 ($z)"; return; }; done; log "⚠ $1.zip 없음 ($DRIVE · packs/)"; }
export PYTHONUNBUFFERED=1
step "0. 재료"; for z in oracle_pack_v2b reid_pack_v2b lora_pack_pilot lora_presence_v2; do unz $z; done
[ -d "$KH/lora_presence_v2" ] || { for z in "$DRIVE/lora_pack_v2.zip" packs/lora_pack_v2.zip; do [ -f "$z" ] && { unzip -q "$z" -d "$KH/" && log "풀림 lora_pack_v2 → lora_presence_v2"; break; }; done; }
[ -d "$KH/lora_presence_v2" ] || log "⚠ lora_pack_v2.zip(233MB) 없음 — Drive 에서 받아 \$DRIVE 에 두면 133채 LoRA, 없으면 파일럿 268샘플로 대체"
python -c "import peft" 2>/dev/null || pip install -q peft && log "peft OK"
DRYO=""; [ "$DRY" = 1 ] && DRYO="MAX_OBJ=5"
# 두 GPU 를 다 쓰려면 무거운 판(27B)만 다른 카드로 보낸다. 단일 GPU 면 G27=$G 로 두면 된다.
G=${G:-1}; G27=${G27:-0}
log "GPU 배치: 27B→cuda:$G27 · 9B/4B→cuda:$G  (단일 GPU 라면 G27=$G 로 실행)"

step "0.5 OG 24채 사슬 재실행 (STEP 5~10) — 캐시 오염·GT 채점 수정판 (§166-58·60)"
# 임베딩(지문 검증·데이터셋별 캐시)·mappts·검증기 전 타겟(ALL_TARGETS=1)·seq 데이터셋별 — 이전 OG 표(9-12~9-18·9-27)는 이 수정 전 것이라 보류.
if [ -d "$OG_ROOT" ] && [ -d "$OG_BENCH" ]; then
  if [ -s $OG_BENCH/scores/t1_all_done.flag ]; then log "skip OG 재실행(flag)"; else
    OUT=$OG_ROOT BENCH_DIR=$OG_BENCH STEP=5 TH=0.06 TOPK=5 MIN_INLIERS=50 ALL_TARGETS=1 bash scripts/nogt_chain_og.sh > $KH/og24_rerun_$(date +%m%d).log 2>&1
    grep -aE "전체 GT|라이브 PnP|최종 답|^\s*(①|②|③)[^ ]* +n=|Traceback" $KH/og24_rerun_$(date +%m%d).log | tail -12 | tee -a $SUM
    python scripts/pose_check_conventions.py $OG_ROOT $OG_BENCH/pnp/pose_all.jsonl 2>/dev/null | head -2 | tee -a $SUM
    grep -q "최종 답" $KH/og24_rerun_$(date +%m%d).log && date > $OG_BENCH/scores/t1_all_done.flag; fi
else log "⚠ OG 경로 없음 → OG 재실행 건너뜀 (OG_ROOT=$OG_ROOT OG_BENCH=$OG_BENCH)"; fi

step "0.6 OG 벤치만 재실행 — 기하 노브 신기본값 (§166-65: C0_MIN=2 · C0_MAXD=2.5)"
# 0.5 를 이미 돌렸다면 사슬 전체가 아니라 10단계(벤치)만 다시 돌린다 — 몇 분이면 끝난다.
# HSSD 133채에서 이 두 값이 ① 거짓 채택 0.226 → 0.030, 총점 0.646 → 0.697 이었다. 종전 노브와 나란히 찍는다.
if [ -d "$OG_ROOT" ] && [ -d "$OG_BENCH" ]; then
  for KNOB in "C0_MIN=2 C0_MAXD=2.5" "C0_MIN=1 C0_MAXD=0"; do
    log "-- OG 벤치 [$KNOB]"
    env $KNOB OUT=$OG_ROOT BENCH_DIR=$OG_BENCH STEP=10 ALL_TARGETS=1 bash scripts/nogt_chain_og.sh 2>&1 \
      | grep -aE "최종 답|^[[:space:]]*(①|②|③확인기회O|④)[^ ]*[[:space:]]+n=" | sed "s|^|  [$KNOB] |" | tee -a $SUM
  done
else log "⚠ OG 경로 없음 → 0.6 건너뜀"; fi

step "1. 9-24 ② 인스턴스 재식별 9B (서문 1/0)"
for P in 1 0; do o=$KH/reid_9b_pre$P.jsonl; [ -s $o ] && { log "skip $o"; continue; }
  env CUDA_VISIBLE_DEVICES=$G $DRYO BACKEND=hf MODEL=$M9 SELECT_IN=$KH/reid_pack_v2b PREAMBLE=$P OUT_JSONL=$o python scripts/oracle_reid_probe.py 2>&1 | grep -aE "REID_DONE|Traceback|Error" | tail -2 | tee -a $SUM; done

step "2. 9-23 오라클 B 27B"
if python - <<PY 2>/dev/null
import sys; from huggingface_hub import scan_cache_dir
ok = any("$M27".split('/')[-1] in r.repo_id for r in scan_cache_dir().repos); sys.exit(0 if ok else 1)
PY
then o=$KH/oracle_27b_B.jsonl; [ -s $o ] && log "skip $o" || env CUDA_VISIBLE_DEVICES=$G27 $DRYO BACKEND=hf MODEL=$M27 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM
else log "27B 캐시 없음 → 건너뜀 (M27=$M27)"; fi

step "3. 9-29 LoRA 자리 존재 판정 — 4B"
D=$KH/lora_presence_v2; { [ "$DRY" = 1 ] || [ ! -d $D ]; } && D=$KH/lora_pack_pilot; log "LoRA 데이터: $D"
if [ -d $D ]; then
  A4=$KH/lora_presence_4b; MS=""; [ "$DRY" = 1 ] && MS="--max-steps 20 --eval-every 10"
  [ -f $A4/adapter_config.json ] && log "skip 학습(어댑터 있음) $A4" || python scripts/lora_presence_train.py --data $D --model $M4 --out $A4 --epochs 2 --eval-every 200 --val-max 300 $MS 2>&1 | grep -aE "^EVAL|LORA_TRAIN_DONE|Traceback|trainable" | tee -a $SUM
  o=$KH/oracle_4b_lora_B.jsonl; [ -s $o ] && log "skip $o" || env CUDA_VISIBLE_DEVICES=$G $DRYO BACKEND=hf MODEL=$M4 ADAPTER=$A4 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM
  o=$KH/oracle_4b_zs_B.jsonl; [ -s $o ] && log "skip $o" || env CUDA_VISIBLE_DEVICES=$G $DRYO BACKEND=hf MODEL=$M4 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM   # 같은 4B 제로샷(대조)
  step "3b. LoRA 9B (1 에폭)"
  A9=$KH/lora_presence_9b; [ -f $A9/adapter_config.json ] && log "skip $A9" || python scripts/lora_presence_train.py --data $D --model $M9 --out $A9 --epochs 1 --eval-every 200 --val-max 300 $MS 2>&1 | grep -aE "^EVAL|LORA_TRAIN_DONE|Traceback|trainable" | tee -a $SUM
  o=$KH/oracle_9b_lora_B.jsonl; [ -s $o ] && log "skip $o" || env CUDA_VISIBLE_DEVICES=$G $DRYO BACKEND=hf MODEL=$M9 ADAPTER=$A9 SELECT_IN=$KH/oracle_pack_v2b VARIANT=B OUT_JSONL=$o python scripts/oracle_vlm_probe.py 2>&1 | grep -aE "ORACLE_DONE|Traceback|Error" | tail -2 | tee -a $SUM
else log "⚠ 학습셋 없음 $D"; fi

step "4. 9-28(a) OG 24채 같은 방 보강 PnP (인라이어 50)"
if [ -d "$OG_ROOT" ] && [ -d "$OG_BENCH" ]; then
  [ -s $OG_BENCH/pnp/pose_all_room.jsonl ] && log "skip pose_all_room" || OUT=$OG_ROOT BENCH_DIR=$OG_BENCH MIRROR=0 PAR=3 bash scripts/og_pnp_room.sh 2>&1 | grep -aE "POSE_CHECK|POSE_JSONL|Traceback" | tee -a $SUM
else log "⚠ OG 경로 없음 (OG_ROOT=$OG_ROOT OG_BENCH=$OG_BENCH) — 환경변수로 지정"; fi

step "5. 요약·묶기"
tar -czf $KH/rtx_batch_$(date +%m%d)_results.tar.gz -C $KH $(cd $KH && ls reid_9b_pre*.jsonl oracle_27b_B.jsonl oracle_4b_lora_B.jsonl oracle_4b_zs_B.jsonl oracle_9b_lora_B.jsonl 2>/dev/null) $(cd $KH && ls -d lora_presence_4b lora_presence_9b 2>/dev/null) rtx_batch_summary_$(date +%m%d).txt 2>/dev/null && log "결과 tar: $KH/rtx_batch_$(date +%m%d)_results.tar.gz (Drive 에 올릴 것)"
log "RTX_BATCH_DONE"; echo; cat $SUM
