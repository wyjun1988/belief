#!/usr/bin/env bash
# 무GT 사슬 v2 (2026-09-06) — SfM 없이: 라벨 검사 → 자가검사 → 캐시 → 방 그룹 → 임베딩 카메라방 → 초기맵(GT 지도 포즈·DA) → 검증 → 거리(DA) → 앵커 PnP(GT 스캔 지도) → 벤치 D
#   OUT=data/hssd90_c4 BENCH_DIR=~/khcache/bench-h90c4 STEP=1 bash scripts/nogt_chain_v2.sh      # STEP 부터 재개
# 남는 GT: 스캔 단계(매핑워크 포즈·평면도·방 라벨)와 경우 라벨. 1fps 구간은 검출·임베딩·PnP·DA·VLM 만.
set -u; cd "$(dirname "$0")/.."
OUT=${OUT:?데이터셋}; B=${BENCH_DIR:?벤치 디렉터리}; STEP=${STEP:-1}; PAR=${PAR:-3}
K=${PY:-$HOME/kx-venv/bin/python}; MLX=${MLX:-$HOME/mlx-venv/bin/python}; HAB=${HAB:-$HOME/miniforge3/envs/hab/bin/python}
HSSD_DATASET=${HSSD_DATASET:-$HOME/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json}; SCENES=${SCENES:-docs/bench/hssd60_c4_scenes.txt}; OFFSET=${OFFSET:-31}
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4
mkdir -p $B/cache $B/scores $B/pnp/logs
HOUSES=$(ls -d $OUT/house_* | wc -l | tr -d ' '); echo "사슬 v2 · $OUT ($HOUSES채) · $B · STEP $STEP · $(date +%H:%M)"
[ $STEP -le 1 ] && { echo "=== 1. 매핑워크 라벨 검사(위치 기준) ==="; $K scripts/fix_map_room_labels.py $OUT 2>&1 | grep -v Warn | tail -2; }
[ $STEP -le 2 ] && { echo "=== 2. 생성기 자가검사 ==="; N=20 $K scripts/gen_selfcheck.py $OUT/house_* 2>&1 | grep -v Warn | tail -2; }
[ $STEP -le 3 ] && { echo "=== 3. 캐시 (앵커 OWL · 이미지질의 · exemplar) $(date +%H:%M) ==="
  THOR_ROOT=$OUT CACHE_PREFIX=$B/cache/hs2_a_ BOXES=1 $K -u scripts/exp_anchowl.py 4 2>&1 | tail -1
  THOR_ROOT=$OUT QCACHE_PREFIX=$B/cache/hs2_q_ STRIDE=4 $K -u scripts/exp_imgq.py 2>&1 | tail -1
  THOR_ROOT=$OUT ACACHE_PREFIX=$B/cache/hs2_x_ STRIDE=4 $K -u scripts/exp_anchor_exemplar.py 2>&1 | tail -1; }
[ $STEP -le 4 ] && { echo "=== 4. 열린 공간 방 그룹 $(date +%H:%M) ==="; i=0
  for SC in $(cat $SCENES); do H=$OUT/house_$(printf %04d $((OFFSET + i))); i=$((i+1)); [ -d $H ] || continue; [ -f $H/room_groups.json ] && continue
    $HAB scripts/room_groups.py --scene "$SC" --dataset $HSSD_DATASET --house $H 2>&1 | grep -aE "^house_" | cut -c1-120; done; }
[ $STEP -le 5 ] && { echo "=== 5. 임베딩 카메라방 (CLIP 노드 + Viterbi) $(date +%H:%M) ==="
  THOR_ROOT=$OUT HOUSES=$HOUSES MODEL=clip EMIT=max OUT_JSONL=$B/scores/room_embed_clip.jsonl $K -u scripts/room_embed.py 2>&1 | grep -aE "전체 GT|Traceback" | cut -c1-160; }
[ $STEP -le 6 ] && { echo "=== 6. 초기맵 (GT 지도 포즈 · DA×0.468 · 검출) $(date +%H:%M) ==="
  THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ INITMAP_GEO=1 INITMAP_INST=1 MAP_DEPTH=da MAP_POINTS=0 MAP_PROP=0 $K -u scripts/build_initmap.py 2>&1 | grep -aE "완료|Traceback" | tail -1; }
[ $STEP -le 7 ] && { echo "=== 7. 검증 점수 (mlx Qwen) $(date +%H:%M) ==="
  THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ FLOOR=0.8 MAXWALK=40 OUT_JSONL=$B/scores/t1_floor0.8_d40.jsonl $MLX -u scripts/exp_t1_verify_mlx.py 2>&1 | tail -1
  echo "  점수 $(wc -l < $B/scores/t1_floor0.8_d40.jsonl)줄"; }
[ $STEP -le 8 ] && { echo "=== 8. 거리 (DA) $(date +%H:%M) ==="
  THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ SCORES=$B/scores/t1_floor0.8_d40.jsonl OUT_JSONL=$B/scores/geo_depth_nogt.jsonl $K -u scripts/geo_depth.py 2>&1 | tail -2; }
[ $STEP -le 9 ] && { echo "=== 9. 앵커 프레임 PnP (GT 스캔 포즈 지도 · CLIP 검색 · SIFT 직접 매칭) $(date +%H:%M) ==="
  for H in $OUT/house_*; do hn=$(basename $H); [ -d data/seq/c4_$hn ] || $K scripts/hssd_to_seq_reloc.py $H data/seq/c4_$hn 2>&1 | grep -v Warn | tail -1; done
  $K - <<PY
import numpy as np, json, glob, os
out = {}
for f in sorted(glob.glob("$B/cache/hs2_a_house_*.npz")): out[os.path.basename(f)[6:-4]] = [int(t) for t in np.load(f, allow_pickle=True)["ts"]]
json.dump(out, open("$B/q_anchors.json", "w")); print("앵커 목록 %d채 %d장" % (len(out), sum(len(v) for v in out.values())))
PY
  one() { hn=$1; S=data/seq/c4_$hn; W=$HOME/khcache/hloc-c4/$hn; NM=$($K -c "import json; print(json.load(open('$S/camera_info.json'))['n_map'])" 2>/dev/null)
    $K -u scripts/reloc_hloc.py $S --scan-end $NM --live-step 1 --work $W --map gt --embed clip --topk 5 --threads 3 --live-list $B/q_anchors.json --house-name $hn \
      --pose-out $B/pnp/pose_$hn.jsonl --hssd-mirror 1 --min-inliers 50 > $B/pnp/logs/$hn.log 2>&1
    echo "  $hn $(grep -aE '라이브 PnP' $B/pnp/logs/$hn.log | sed -E 's/^\[ *[0-9]+s\] //; s/ · 장당.*//' | cut -c1-150)"; }
  export -f one; export K B
  ls -d $OUT/house_* | xargs -n1 basename | xargs -P $PAR -I{} bash -c 'one {}'
  cat $B/pnp/pose_house_*.jsonl > $B/pnp/pose_all.jsonl; echo "  POSE_JSONL $(wc -l < $B/pnp/pose_all.jsonl)줄"; }
[ $STEP -le 10 ] && { echo "=== 10. 벤치 D (GT 0) $(date +%H:%M) ==="
  env BENCH_DIR=$B THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ VERIFY_JSONL=$B/scores/t1_floor0.8_d40.jsonl \
    GEO_DEPTH=$B/scores/geo_depth_nogt.jsonl ROOM_GROUPS=1 PY=$K POSE_JSONL=$B/pnp/pose_all.jsonl ROOM_JSONL=$B/scores/room_embed_clip.jsonl ROWS_OUT=$B/rows_D.jsonl \
    bash scripts/bench.sh > $B/bench_D_full.log 2>&1
  grep -aE "재료 사다리|GT 재료|최종 답|^\s*(①|②|③|④)[^ ]* +n=" $B/bench_D_full.log | cut -c1-200; }
echo "NOGT_CHAIN_V2_DONE $(date +%H:%M)"
