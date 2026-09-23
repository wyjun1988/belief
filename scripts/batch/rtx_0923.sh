#!/usr/bin/env bash
# 프로6000 배치 (2026-09-23) — 추론만, 약 1~1.5시간. 순서: 9-61 → 9-59(나머지) → 9-58
#   cd ~/work/khronos && git pull && KC=~/work/khcache bash scripts/batch/rtx_0923.sh          # 실행
#   DRY=1 KC=~/work/khcache bash scripts/batch/rtx_0923.sh                                      # 입력만 점검(GPU 안 씀)
# 끝나면 $KC/out_0923/rtx_0923_all.tar.gz 하나만 드라이브에 올리면 된다. 작업별 묶음도 각자 남는다(중간에 끊겨도 됨).
set -u; cd "$(git rev-parse --show-toplevel 2>/dev/null || echo ~/work/khronos)"; export KMP_DUPLICATE_LIB_OK=TRUE
KC=${KC:-$HOME/work/khcache}; PY=${PY:-python}; DRY=${DRY:-0}; O=$KC/out_0923; mkdir -p $O
log(){ echo "[$(date +%H:%M)] $*" | tee -a $O/rtx_0923_summary.txt; }
need(){ for p in "$@"; do [ -e "$p" ] || { log "  ✗ 입력 없음: $p"; return 1; }; done; }
untar(){ [ -e "$2" ] || { [ -f "$KC/$1" ] && tar xzf "$KC/$1" -C "$KC" && log "  풀었다 $1"; }; }
log "=== 프로6000 배치 시작 · KC=$KC · DRY=$DRY ==="

# ── 9-61 c2set 채택 판정 재측정 (챔피언 어댑터 · 문맥 없이) ──
t961(){ log "── 9-61 c2set 채택 판정"; untar rtx_9_61_adopt_pack_c2set_fix.tar.gz $KC/adopt_pack_c2set_fix
  need $KC/adopt_pack_c2set_fix/items.jsonl $KC/lora_adopt_all_full_4b/adapter_config.json || return
  OUTF=$O/adopt_margin_c2set_fix_0923.jsonl; [ -s $OUTF ] && [ $(wc -l < $OUTF) -ge 8520 ] && { log "  이미 있음(건너뜀)"; return; }
  [ $DRY = 1 ] && { log "  (DRY) 입력 OK"; return; }
  PACK=$KC/adopt_pack_c2set_fix ADAPTER=$KC/lora_adopt_all_full_4b BACKEND=hf MODEL=Qwen/Qwen3.5-4B DEVICE=cuda USE_CTX=0 \
    VERDICT_JSONL=$OUTF $PY scripts/lora_adopt_infer.py > $O/infer_c2set_fix_0923.log 2>&1
  log "  $(tail -1 $O/infer_c2set_fix_0923.log | cut -c1-120)"
  tar czf $O/rtx_9_61_result.tar.gz -C $O adopt_margin_c2set_fix_0923.jsonl infer_c2set_fix_0923.log; }

# ── 9-59 나머지: 새 시뮬 5차분 앵커 등록부 + 기록자리 마진 (앞 단계 결과는 $KC/ns5_out · 초기맵은 묶음 안) ──
t959(){ log "── 9-59 새 시뮬 5차분 기록자리 마진"; R=$KC/rtx_9_59_ns5/newsim5; B=$KC/ns5_out
  need $R/house_0000/initmap_owl.json $KC/rtx_9_59_ns5/lora_place_bal_4b/adapter_config.json $B/cache || return
  [ -s $B/place_margin.jsonl ] && { log "  이미 있음(건너뜀)"; } || {
    [ $DRY = 1 ] && { log "  (DRY) 입력 OK"; return; }
    THOR_ROOT=$R INITMAP_FILE=initmap_owl.json QUERY_ONLY=1 $PY -u scripts/anchor_registry.py > $O/ns5_anchor_registry.log 2>&1
    log "  등록부 $(tail -1 $O/ns5_anchor_registry.log | cut -c1-80)"
    THOR_ROOT=$R INITMAP_FILE=initmap_owl.json ADAPTER=$KC/rtx_9_59_ns5/lora_place_bal_4b DEVICE=cuda OUT_JSONL=$B/place_margin.jsonl \
      $PY -u scripts/lora_place_infer.py > $O/ns5_place_infer.log 2>&1
    log "  마진 $(wc -l < $B/place_margin.jsonl 2>/dev/null || echo 0)줄"; }
  ( cd $B && tar czf $O/rtx_9_59_ns5_out.tar.gz cache scores place_margin.jsonl 2>/dev/null )
  ( cd $R && tar czf $O/rtx_9_59_ns5_initmap.tar.gz house_*/initmap_owl.json house_*/initmap_owl_raw.json 2>/dev/null )
  log "  캐시 $(ls $B/cache/hs2_a_*.npz 2>/dev/null | wc -l)채 · 초기맵 $(ls $R/house_*/initmap_owl.json 2>/dev/null | wc -l)채 묶음"; }

# ── 9-58 큰 검출기(OWLv2-large)로 소형 물체 재검출 (9-54 묶음 재사용) ──
t958(){ log "── 9-58 OWLv2-large"; untar rtx_9_54_tile.tar.gz $KC/rtx_9_54_tile
  need $KC/rtx_9_54_tile/bench-v2full/rows_DG_진단.jsonl || return
  [ -s $O/owl_large_0923.log ] && grep -q OWL_TILE_PILOT_DONE $O/owl_large_0923.log && { log "  이미 있음(건너뜀)"; return; }
  [ $DRY = 1 ] && { log "  (DRY) 입력 OK"; return; }
  OWL_MODEL=google/owlv2-large-patch14-ensemble BENCH_DIR=$KC/rtx_9_54_tile/bench-v2full THOR_ROOT=$KC/rtx_9_54_tile/data/hssd_v2 \
    $PY scripts/owl_tile_pilot.py > $O/owl_large_0923.log 2>&1
  log "  $(grep -E '합계|물체 단위' $O/owl_large_0923.log | tr '\n' ' ' | cut -c1-160)"; }

t961; t959; t958
[ $DRY = 1 ] && { log "=== DRY 끝 — ✗ 줄이 없으면 DRY=0 으로 실행 ==="; exit 0; }
( cd $O && tar czf rtx_0923_all.tar.gz rtx_0923_summary.txt *.log *.jsonl rtx_9_59_ns5_out.tar.gz rtx_9_59_ns5_initmap.tar.gz 2>/dev/null )
log "=== 끝 → $O/rtx_0923_all.tar.gz 를 드라이브에 ==="
