#!/usr/bin/env bash
# H100 배치 (2026-09-23) — 학습 4건(문맥 2팔 + 부재 2팔)·추론 6건, 약 5시간. 순서: 9-60b(장면그래프 문맥 ctx1) → 9-64(부재 판정기 재학습)
#   cd <khronos> && git pull && KC=<khcache 경로> bash scripts/batch/h100_0923.sh          # 실행
#   DRY=1 KC=<khcache 경로> bash scripts/batch/h100_0923.sh                                 # 입력만 점검
# 필요한 새 입력(드라이브): rtx_9_61_adopt_pack_c2set_fix.tar.gz · rtx_9_64_pres.tar.gz  → $KC 에 두기만 하면 스크립트가 푼다.
# 끝나면 $KC/out_0923/h100_0923_all.tar.gz 하나만 드라이브에. 어댑터 본체는 넣지 않는다(크다) — 판정 결과가 좋으면 따로 요청한다.
set -u; cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"; export KMP_DUPLICATE_LIB_OK=TRUE
KC=${KC:?KC=khcache 경로 필요}; PY=${PY:-python}; DRY=${DRY:-0}; O=$KC/out_0923; mkdir -p $O $KC/out_9_60 $KC/out_9_64
log(){ echo "[$(date +%H:%M)] $*" | tee -a $O/h100_0923_summary.txt; }
need(){ for p in "$@"; do [ -e "$p" ] || { log "  ✗ 입력 없음: $p"; return 1; }; done; }
untar(){ [ -e "$2" ] || { [ -f "$KC/$1" ] && tar xzf "$KC/$1" -C "$KC" && log "  풀었다 $1"; }; }
log "=== H100 배치 시작 · KC=$KC · DRY=$DRY ==="

# ── 9-60b 장면그래프 문맥 A/B — 두 팔을 **이 실행에서 같은 데이터로 다시** 학습한다 ──
# 9/23 정정: 9-60 지시의 합본 명령이 인자 순서가 틀렸다(출력이 첫 인자인데 --out 을 붙였다 → 그대로면 실패).
#   ctx0 이 어떤 데이터로 학습됐는지 확인할 수 없으므로 ctx0·ctx1 을 새 디렉터리(out_9_60b)에서 함께 학습한다.
t960b(){ log "── 9-60b 장면그래프 문맥 A/B"; untar rtx_9_60_ctx.tar.gz $KC/rtx_9_60_ctx; untar rtx_9_61_adopt_pack_c2set_fix.tar.gz $KC/adopt_pack_c2set_fix
  need $KC/lora_adopt_v2/images $KC/lora_adopt_c2/images $KC/lora_adopt_og/images $KC/val_fixed/val.jsonl \
       $KC/rtx_9_60_ctx/lora_adopt_v2/train.jsonl $KC/adopt_pack_all/items.jsonl $KC/adopt_pack_c2set_fix/items.jsonl || return
  [ $DRY = 1 ] && { log "  (DRY) 입력 OK"; return; }
  # 1) 문맥 jsonl 을 팩에 넣는다(이미지는 그대로) — 원본은 .noctx 로 한 번만 보관
  for d in lora_adopt_v2 lora_adopt_c2 lora_adopt_real; do for sp in train val; do
    [ -f $KC/$d/$sp.jsonl.noctx ] || cp $KC/$d/$sp.jsonl $KC/$d/$sp.jsonl.noctx 2>/dev/null
    cp $KC/rtx_9_60_ctx/$d/$sp.jsonl $KC/$d/$sp.jsonl 2>/dev/null; done; done
  [ -f $KC/val_fixed/val.jsonl.noctx ] || cp $KC/val_fixed/val.jsonl $KC/val_fixed/val.jsonl.noctx; cp $KC/rtx_9_60_ctx/val_fixed/val.jsonl $KC/val_fixed/val.jsonl
  # 2) 집 단위 합본 (출력이 첫 인자)
  rm -rf $KC/lora_adopt_all_ctx
  $PY scripts/lora_merge_sets.py $KC/lora_adopt_all_ctx $KC/lora_adopt_v2 $KC/lora_adopt_c2 $KC/lora_adopt_og | tail -1 | tee -a $O/h100_0923_summary.txt
  n=$(grep -c '"ctx": "Around' $KC/lora_adopt_all_ctx/train.jsonl); log "  합본 문맥 있는 행 $n"; [ "$n" -gt 1000 ] || { log "  ✗ 문맥 행 부족 — 중단"; return; }
  # 3) 두 팔 학습 (데이터·시드 동일, USE_CTX 만 다름)
  mkdir -p $KC/out_9_60b
  for CTX in 0 1; do
    [ -s $KC/out_9_60b/adopt_ctx${CTX}/adapter_config.json ] && { log "  ctx${CTX} 어댑터 이미 있음"; continue; }
    USE_CTX=${CTX} $PY scripts/lora_presence_train.py --task adopt --data $KC/lora_adopt_all_ctx --val-data $KC/val_fixed --targets full --grad-ckpt --img-max 448 \
      --out $KC/out_9_60b/adopt_ctx${CTX} > $O/adopt_ctx${CTX}_train.log 2>&1
    log "  ctx${CTX} 학습: $(grep -aE 'LORA_TRAIN_DONE|EVALGRP\[final\]' $O/adopt_ctx${CTX}_train.log | tail -2 | tr '\n' ' ' | cut -c1-200)"; done
  # 4) 추론: 133채 · c2set × 두 팔 (items 는 문맥판)
  cmp -s $KC/rtx_9_60_ctx/items/items_hssd133.jsonl $KC/adopt_pack_all/items.jsonl || { [ -f $KC/adopt_pack_all/items.jsonl.noctx ] || cp $KC/adopt_pack_all/items.jsonl $KC/adopt_pack_all/items.jsonl.noctx; cp $KC/rtx_9_60_ctx/items/items_hssd133.jsonl $KC/adopt_pack_all/items.jsonl; }
  for SET in hssd133:adopt_pack_all c2set:adopt_pack_c2set_fix; do NM=${SET%%:*}; PK=${SET#*:}
    for CTX in 0 1; do F=$O/adopt_margin_${NM}_ctx${CTX}_0923.jsonl; [ -s $F ] && continue
      PACK=$KC/$PK ADAPTER=$KC/out_9_60b/adopt_ctx${CTX} BACKEND=hf MODEL=Qwen/Qwen3.5-4B DEVICE=cuda USE_CTX=${CTX} \
        VERDICT_JSONL=$F $PY scripts/lora_adopt_infer.py > $O/infer_${NM}_ctx${CTX}_0923.log 2>&1
      log "  ${NM} ctx${CTX}: $(tail -1 $O/infer_${NM}_ctx${CTX}_0923.log | cut -c1-100)"; done; done; }

# ── 9-64 ③ 부재 판정기 재학습 2팔 → c3big 타임라인 판정 ──
t964(){ log "── 9-64 부재 판정기 재학습"; untar rtx_9_64_pres.tar.gz $KC/rtx_9_64_pres
  need $KC/lora_presence_v2/images $KC/rtx_9_64_pres/lora_presence_v2_ynbal/train.jsonl $KC/rtx_9_64_pres/lora_presence_v2_bin/train.jsonl \
       $KC/rtx_9_63_tl/pack/timeline_prep_sel.jsonl || return
  [ $DRY = 1 ] && { log "  (DRY) 입력 OK"; return; }
  for d in lora_presence_v2_ynbal lora_presence_v2_bin; do ln -sfn $KC/lora_presence_v2/images $KC/rtx_9_64_pres/$d/images; done
  # P1: yes·no 를 같은 수로(no 복제), unsure 는 원래 187 그대로 · P2: unsure 없이 yes·no 만 + --balance
  [ -s $KC/out_9_64/pres_ynbal/adapter_config.json ] || USE_CTX=0 $PY scripts/lora_presence_train.py --task presence --data $KC/rtx_9_64_pres/lora_presence_v2_ynbal \
    --targets narrow --out $KC/out_9_64/pres_ynbal > $O/pres_ynbal_train.log 2>&1
  [ -s $KC/out_9_64/pres_bin/adapter_config.json ] || USE_CTX=0 $PY scripts/lora_presence_train.py --task presence --data $KC/rtx_9_64_pres/lora_presence_v2_bin \
    --targets narrow --balance --out $KC/out_9_64/pres_bin > $O/pres_bin_train.log 2>&1
  for AD in pres_ynbal pres_bin; do
    log "  $AD 학습: $(grep -aE 'LORA_TRAIN_DONE|EVAL\[final\]' $O/${AD}_train.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
    F=$O/timeline_9_64_${AD}.jsonl; [ -s $F ] && continue
    BACKEND=hf MODEL=Qwen/Qwen3.5-4B ADAPTER=$KC/out_9_64/$AD THOR_ROOT=$KC/rtx_9_63_tl/pack PREP_JSONL=$KC/rtx_9_63_tl/pack/timeline_prep_sel.jsonl \
      MODE=simple PREFILL=1 OUT_JSONL=$F $PY -u scripts/timeline_verdict_mlx.py > $O/tl_9_64_${AD}.log 2>&1
    log "  $AD 판정 분포: $($PY -c "import json,collections,sys; print(dict(collections.Counter(json.loads(l).get('at_spot') for l in open('$F'))))" 2>/dev/null)"
  done; }

[ "${SKIP_960B:-0}" = 1 ] && log "── 9-60b 건너뜀(SKIP_960B=1)" || t960b; t964
[ $DRY = 1 ] && { log "=== DRY 끝 — ✗ 줄이 없으면 DRY=0 으로 실행 ==="; exit 0; }
( cd $O && tar czf h100_0923_all.tar.gz h100_0923_summary.txt *.log *.jsonl 2>/dev/null )
log "=== 끝 → $O/h100_0923_all.tar.gz 를 드라이브에 ==="
