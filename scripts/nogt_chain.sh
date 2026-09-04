#!/bin/bash
# GT 제거 사슬 — 생성된 데이터셋을 받아 사다리에서 GT 를 전부 걷어낸 벤치까지 간다.
#   OUT=data/hssd20_c3 SCENES=docs/bench/hssd20_scenes.txt SEED0=700 BENCH_DIR=~/khcache/bench-h20c3 bash scripts/nogt_chain.sh
# 단계: 1 재매핑(이동 프레임) → 2 앵커캐시 → 3 SfM(라벨 정렬·DA 척도) → 4 초기맵(SfM 포즈+SfM 점 깊이)
#        → 5 검증점수 → 6 거리(DA+SfM 점) → 7 벤치(위치·포즈·카메라방 SfM · 거리 DA · 초기맵 검출)
# 남는 GT: 경우 분류(①②③④ 라벨)뿐 — 그건 정답지이지 시스템 입력이 아니다.
set -u
OUT=${OUT:?데이터셋 디렉터리}; SCENES=${SCENES:?장면 목록}; SEED0=${SEED0:-700}
B=${BENCH_DIR:?벤치 디렉터리}; SFM=${SFM:-$HOME/khcache/sfm-nogt}
HAB=${HAB:-$HOME/miniforge3/envs/hab/bin/python}; PY=${PY:-$HOME/kx-venv/bin/python}
MLX=${MLX:-$HOME/mlx-venv/bin/python}
HSSD_DATASET=${HSSD_DATASET:-$HOME/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json}
TRAVEL=${TRAVEL:-0.35}; PAR=${PAR:-3}; THREADS=${THREADS:-3}; STEP=${STEP:-1}
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")/.."; mkdir -p $SFM $B/scores

if [ "$STEP" -le 1 ]; then
echo "=== 1. 재매핑 (이동 프레임 ${TRAVEL}m) ==="
i=0
for SC in $(cat "$SCENES"); do
  H="$OUT/house_$(printf %04d $i)"; i=$((i+1))
  [ -d "$H" ] || continue
  rm -f "$H"/map/*.jpg                     # 지점 수가 줄면 옛 프레임이 남는다 — gt["map"] 은 통째로 갈린다
  $HAB -u scripts/hab_episode.py --scene "$SC" --dataset "$HSSD_DATASET" --remap --map-travel $TRAVEL \
    --seed $((i-1+SEED0)) --out "$H" 2>&1 | grep -aE "매핑 포즈|remap 완료" | sed "s|^|  $(basename $H) |"
done
fi

if [ "$STEP" -le 2 ]; then
echo "=== 2. 앵커 캐시 재생성 (지도가 바뀌었으므로) ==="
rm -f $B/cache/hs2_a_*.npz $B/cache/hs2_x_*.npz
THOR_ROOT=$OUT CACHE_PREFIX=$B/cache/hs2_a_ BOXES=1 $PY -u scripts/exp_anchowl.py 4 2>&1 | tail -2
THOR_ROOT=$OUT ACACHE_PREFIX=$B/cache/hs2_x_ STRIDE=4 $PY -u scripts/exp_anchor_exemplar.py 2>&1 | tail -2
fi

if [ "$STEP" -le 3 ]; then
echo "=== 3. SfM 재국소화 (라벨 정렬·DA 척도 — GT 0) ==="
N=$(ls -d $OUT/house_* | wc -l | tr -d ' ')
ROOT=$OUT HOUSES=$N PAR=$PAR THREADS=$THREADS SFM=$SFM PY=$PY \
  EXTRA="--align sites --scale da --fast" OUT=$SFM/pose_all.jsonl bash scripts/sfm_batch.sh 2>&1 | tail -40
fi

if [ "$STEP" -le 4 ]; then
echo "=== 4. 초기맵 (SfM 포즈 + SfM 점 깊이) ==="
THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ INITMAP_GEO=1 INITMAP_INST=1 \
  MAP_POSE_DIR=$SFM MAP_DEPTH=pts MAP_POINTS=1 MAP_PROP=0 $PY -u scripts/build_initmap.py 2>&1 | tail -6
fi

if [ "$STEP" -le 5 ]; then
echo "=== 5. 검증 점수 (지도 앵커가 바뀌었으므로 다시) ==="
THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ FLOOR=0.8 MAXWALK=40 \
  OUT_JSONL=$B/scores/t1_floor0.8_d40.jsonl $MLX -u scripts/exp_t1_verify_mlx.py 2>&1 | tail -2
fi

if [ "$STEP" -le 6 ]; then
echo "=== 6. 거리 (DA-V2 + SfM 점 앵커 — GT 0) ==="
THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ \
  SCORES=$B/scores/t1_floor0.8_d40.jsonl SFM_POINTS=$SFM POSE_JSONL=$SFM/pose_all.jsonl \
  OUT_JSONL=$B/scores/geo_depth_nogt.jsonl $PY -u scripts/geo_depth.py 2>&1 | tail -4
fi

echo "=== 7. 벤치 (GT 0 — 경우 분류만 GT) ==="
BENCH_DIR=$B THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ \
  VERIFY_JSONL=$B/scores/t1_floor0.8_d40.jsonl POSE_JSONL=$SFM/pose_all.jsonl \
  GEO_DEPTH=$B/scores/geo_depth_nogt.jsonl ROOM_GROUPS=1 PY=$PY bash scripts/bench.sh 2>&1 \
  | grep -aE "재료 사다리|GT 재료|최종 답|①이동없음  |②재촬영 |③|④|기준선" | grep -avE "^\s+③.*\[" | cut -c1-140
echo NOGT_CHAIN_DONE
