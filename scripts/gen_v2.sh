#!/usr/bin/env bash
# v2 데이터 생성 (2026-09-09): 유령 수정판 생성기 + ③ 목적지 사전확률 대본 + 확장 MOVABLE.
# 종전 115채 벤치는 (a) 렌더에 없는 유령 45%, (b) ③ 목적지를 far_low(사전확률 반대)로 뽑아 belief 측정 불가였다(§166-25·§166-34).
#   FROM=0 TO=134 OUT=data/hssd_v2 CASE3=0.5 bash scripts/gen_v2.sh
set -u; cd "$(dirname "$0")/.."
SCENES=${SCENES:-docs/bench/hssd_v2_scenes.txt}; FROM=${FROM:-0}; TO=${TO:-999}; OUT=${OUT:-data/hssd_v2}; SEED0=${SEED0:-7000}
HSSD_DATASET=${HSSD_DATASET:-$HOME/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json}; HAB=${HAB:-$HOME/miniforge3/envs/hab/bin/python}
export KMP_DUPLICATE_LIB_OK=TRUE; mkdir -p "$OUT"; i=0; n_ok=0
for SC in $(cat "$SCENES"); do
  if [ $i -ge $FROM ] && [ $i -lt $TO ]; then
    H="$OUT/house_$(printf %04d $i)"
    if [ ! -f "$H/gt.json" ]; then
      echo "=== $H ← $SC $(date +%H:%M) ==="
      $HAB -u scripts/hab_episode.py --scene "$SC" --dataset "$HSSD_DATASET" --move data/hssd_move.json \
        --frames ${FRAMES:-1200} --moves ${MOVES:-8} --case3 ${CASE3:-0.5} --far 0.0 --evidence 3:1.4 --dwell 0 --pace 0.25 --turn 0.5 --scan 35 --max-turn 999 \
        --map-travel 0.35 --map-sites 1 --map-step 60 --c3-check-dist ${C3DIST:-1.2} --seed $((SEED0 + i)) --out "$H" 2>&1 \
        | grep -aE "이동 후보|③ 자격|이동 계획|이동 기록|이동 취소|핸들 없는|Traceback|Error" || echo "  $SC 실패"
      [ -f "$H/gt.json" ] || { echo "  ⚠ gt.json 없음 → 삭제"; rm -rf "$H"; }
    fi
    [ -f "$H/gt.json" ] && n_ok=$((n_ok+1))
  fi; i=$((i+1))
done
echo "GEN_V2_DONE $(date +%H:%M) · gt.json $n_ok 채"
