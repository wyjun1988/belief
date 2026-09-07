#!/usr/bin/env bash
# 두 번째 에피소드 — 이동이 1건 이상인 집만 같은 장면·다른 시드(+1000)로 다시 생성 (집 번호 +100, OUT2). ②③ 표본 배가(규칙 불변, 집 단위 상관 있음).
#   OUT=data/hssd90_c4 OUT2=data/hssd90_c4e2 FROM=30 TO=60 OFFSET=31 bash scripts/gen_c4_ep2.sh
set -u; cd "$(dirname "$0")/.."
SCENES=${SCENES:-docs/bench/hssd60_c4_scenes.txt}; FROM=${FROM:-0}; TO=${TO:-60}; OFFSET=${OFFSET:-31}; OUT=${OUT:-data/hssd90_c4}; OUT2=${OUT2:-data/hssd90_c4e2}
HSSD_DATASET=${HSSD_DATASET:-$HOME/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json}; HAB=${HAB:-$HOME/miniforge3/envs/hab/bin/python}; K=${PY:-$HOME/kx-venv/bin/python}
export KMP_DUPLICATE_LIB_OK=TRUE; mkdir -p "$OUT2"; i=0
for SC in $(cat "$SCENES"); do
  if [ $i -ge $FROM ] && [ $i -lt $TO ]; then
    H1="$OUT/house_$(printf %04d $((OFFSET + i)))"; H2="$OUT2/house_$(printf %04d $((OFFSET + i + 100)))"
    NMV=$($K -c "import json; print(len(json.load(open('$H1/gt.json'))['moves']))" 2>/dev/null || echo 0)
    if [ "$NMV" -ge 1 ] && [ ! -f "$H2/gt.json" ]; then
      echo "=== $H2 ← $SC (ep1 이동 $NMV) $(date +%H:%M) ==="
      $HAB -u scripts/hab_episode.py --scene "$SC" --dataset "$HSSD_DATASET" --move data/hssd_move.json \
        --frames 1200 --moves 8 --case3 ${CASE3:-1.0} --far 0.0 --evidence 3:1.4 --dwell 0 --pace 0.25 --turn 0.5 --scan 35 --max-turn 999 \
        --map-travel 0.35 --map-sites 1 --map-step 60 --seed $((OFFSET + i + 1200)) --out "$H2" 2>&1 \
        | grep -aE "이동 계획|이동 기록|이동 취소|Traceback|Error" || echo "  $SC 실패"
      [ -f "$H2/gt.json" ] || { echo "  ⚠ gt.json 없음 → 삭제: $H2"; rm -rf "$H2"; }   # 생성 중 죽은 집은 남기지 않는다(체인이 그 집에서 죽는다)
    fi
  fi; i=$((i+1))
done
echo GEN_C4_EP2_DONE
