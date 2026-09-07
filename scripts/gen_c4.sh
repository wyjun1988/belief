#!/usr/bin/env bash
# hssd90_c4 생성 — ③ 대본 6차 파라미터 그대로(bench_v2_chain.sh 1단계), 집만 늘린다(집 번호 31~90). 성긴 지도(0.35 m·지점 1·60°) 를 생성 시 바로.
#   SCENES=docs/bench/hssd60_c4_scenes.txt FROM=0 TO=30 OFFSET=31 OUT=data/hssd90_c4 bash scripts/gen_c4.sh   # 장면 목록의 [FROM,TO) 를 house_(OFFSET+i)
set -u; cd "$(dirname "$0")/.."
SCENES=${SCENES:-docs/bench/hssd60_c4_scenes.txt}; FROM=${FROM:-0}; TO=${TO:-60}; OFFSET=${OFFSET:-31}; OUT=${OUT:-data/hssd90_c4}
HSSD_DATASET=${HSSD_DATASET:-$HOME/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json}; HAB=${HAB:-$HOME/miniforge3/envs/hab/bin/python}
export KMP_DUPLICATE_LIB_OK=TRUE; mkdir -p "$OUT"
i=0
for SC in $(cat "$SCENES"); do
  if [ $i -ge $FROM ] && [ $i -lt $TO ]; then
    H="$OUT/house_$(printf %04d $((OFFSET + i)))"
    if [ -f "$H/gt.json" ]; then echo "  $H 있음 — 건너뜀"; else
      echo "=== $H ← $SC $(date +%H:%M) ==="
      $HAB -u scripts/hab_episode.py --scene "$SC" --dataset "$HSSD_DATASET" --move data/hssd_move.json \
        --frames 1200 --moves 8 --case3 0.5 --far 0.0 --evidence 3:1.4 --dwell 0 --pace 0.25 --turn 0.5 --scan 35 --max-turn 999 \
        --map-travel 0.35 --map-sites 1 --map-step 60 --seed $((OFFSET + i + 200)) --out "$H" 2>&1 \
        | grep -aE "이동 후보|③ 자격|이동 계획|이동 기록|이동 취소|증인 렌더|매핑 포즈|Traceback|Error" || echo "  $SC 실패"
      [ -f "$H/gt.json" ] || { echo "  ⚠ gt.json 없음 → 삭제: $H"; rm -rf "$H"; }   # 생성 중 죽은 집은 남기지 않는다(체인이 그 집에서 죽는다)
    fi
  fi
  i=$((i+1))
done
echo GEN_C4_DONE
