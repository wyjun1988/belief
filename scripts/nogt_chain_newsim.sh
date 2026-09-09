#!/usr/bin/env bash
# 새 시뮬레이터(AIUE, newsim_to_gt.py 출력) 용 무GT 사슬 — nogt_chain_og.sh 와 같되 M2 에서 돈다(검증기 mlx Qwen, kx-venv). 2026-09-09
#   OUT=/mnt/ssd2/wooyeol/work/og4v BENCH_DIR=~/khcache/bench-og4v STEP=3 bash scripts/nogt_chain_og.sh
#   지도 포즈 출처: MAP_POSE_DIR 비우면 GT 스캔 포즈(gt.json map apos/yaw) · CUT3R 포즈로 하려면 MAP_POSE_DIR=~/khcache/cut3r_v (<dir>/<house>/map_pose_<house>.jsonl, 아래 9-b 참고)
#   차이: 1·2·4 단계 없음(HSSD 라벨 수정·자가검사·habitat 방 그룹) · 초점거리 FRAME_FX(scene_meta.intrinsics) · 기울기 TILT=0 · 미러 MIRROR(기본 0) · 검증기 = HF Qwen(exp_t1_verify_pipeline) · 사전확률 OG 어휘 없음(→ 균등, belief 무의미)
set -u; cd "$(dirname "$0")/.."
OUT=${OUT:-data/newsim}; B=${BENCH_DIR:-$HOME/khcache/bench-newsim}; STEP=${STEP:-3}; PAR=${PAR:-2}; K=${PY:-$HOME/kx-venv/bin/python}; MIRROR=${MIRROR:-0}
MLX=${MLX:-$HOME/mlx-venv/bin/python}
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
H0=$(ls -d $OUT/house_* | head -1); read FW FX <<< "$($K -c "import json,sys; i=json.load(open('$H0/gt.json')).get('scene_meta',{}).get('intrinsics') or {}; print(i.get('W',1280), i.get('fx', i.get('W',1280)/2))")"
export FRAME_W=$FW FRAME_FX=$FX TILT=${TILT:-0}
mkdir -p $B/cache $B/scores $B/pnp/logs; HOUSES=$(ls -d $OUT/house_* | wc -l | tr -d ' ')
echo "OG 사슬 · $OUT ($HOUSES채) · $B · STEP $STEP · W $FRAME_W fx $FRAME_FX tilt $TILT mirror $MIRROR · $(date +%H:%M)"
[ $STEP -le 3 ] && { echo "=== 3. 캐시 (앵커 OWL · 이미지질의 · exemplar) $(date +%H:%M) ==="
  THOR_ROOT=$OUT CACHE_PREFIX=$B/cache/hs2_a_ BOXES=1 $K -u scripts/exp_anchowl.py 4 2>&1 | tail -1
  THOR_ROOT=$OUT QCACHE_PREFIX=$B/cache/hs2_q_ STRIDE=4 $K -u scripts/exp_imgq.py 2>&1 | tail -1
  THOR_ROOT=$OUT ACACHE_PREFIX=$B/cache/hs2_x_ STRIDE=4 $K -u scripts/exp_anchor_exemplar.py 2>&1 | tail -1; }
[ $STEP -le 5 ] && { echo "=== 5. 임베딩 카메라방 (CLIP 노드 + Viterbi) $(date +%H:%M) ==="
  THOR_ROOT=$OUT HOUSES=$HOUSES MODEL=clip EMIT=max OUT_JSONL=$B/scores/room_embed_clip.jsonl $K -u scripts/room_embed.py 2>&1 | grep -aE "전체 GT|Traceback" | cut -c1-160; }
[ $STEP -le 6 ] && { echo "=== 6. 초기맵 (지도 포즈 ${MAP_POSE_DIR:-GT} · DA 자가보정 · 검출) $(date +%H:%M) ==="
  # 스캔 삼각측량 점(DA 척도 자가보정)은 9단계 뒤에야 생기므로 STEP 6 첫 실행은 DA_K 상수(0.5, OG 는 미측정)로. 9단계 뒤 `STEP=6 REBUILD=1` 로 다시 돌리면 점·자가보정을 쓴다.
  if [ "${REBUILD:-0}" = 1 ] && [ -d "$HOME/khcache/mappts" ]; then _MP="MAP_POINTS=1 MAP_POSE_DIR=$HOME/khcache/mappts DA_K=auto"; else _MP="MAP_POINTS=0 DA_K=${DA_K:-0.5}"; fi
  [ -n "${MAP_POSE_DIR:-}" ] && _MP="$_MP MAP_POSE_DIR=$MAP_POSE_DIR MAP_PROP=1"
  env THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ INITMAP_GEO=1 INITMAP_INST=1 MAP_DEPTH=da INITMAP_RAW=1 $_MP $K -u scripts/build_initmap.py 2>&1 | grep -aE "완료|방배정|자가보정|SfM 대체|Traceback" | tail -$HOUSES
  $K scripts/recluster_initmap.py $OUT --out initmap_owl_rc.json --rank max --th 0.12; }
[ $STEP -le 7 ] && [ "${SKIP_VERIFY:-0}" = 1 ] && { echo "=== 7. 검증 생략(SKIP_VERIFY=1) ==="; : > $B/scores/t1_floor0.8_d40.jsonl; }
[ $STEP -le 7 ] && [ "${SKIP_VERIFY:-0}" != 1 ] && { echo "=== 7. 검증 점수 (mlx Qwen) $(date +%H:%M) ==="
  THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ FLOOR=0.8 MAXWALK=40 OUT_JSONL=$B/scores/t1_floor0.8_d40.jsonl $MLX -u scripts/exp_t1_verify_mlx.py 2>&1 | tail -1
  echo "  점수 $(wc -l < $B/scores/t1_floor0.8_d40.jsonl)줄"; }
[ $STEP -le 8 ] && { echo "=== 8. 거리 (DA) $(date +%H:%M) ==="
  THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ SCORES=$B/scores/t1_floor0.8_d40.jsonl OUT_JSONL=$B/scores/geo_depth_nogt.jsonl $K -u scripts/geo_depth.py 2>&1 | tail -2; }
[ $STEP -le 9 ] && { echo "=== 9. 앵커 프레임 PnP (스캔 포즈 지도 · CLIP 검색 · SIFT) $(date +%H:%M) ==="
  for H in $OUT/house_*; do hn=$(basename $H); [ -d data/seq/newsim_$hn ] || $K scripts/hssd_to_seq_reloc.py $H data/seq/newsim_$hn --mirror $MIRROR 2>&1 | grep -v Warn | tail -1; done
  $K - <<PY
import numpy as np, json, glob, os
out = {}
for f in sorted(glob.glob("$B/cache/hs2_a_house_*.npz")): out[os.path.basename(f)[6:-4]] = [int(t) for t in np.load(f, allow_pickle=True)["ts"]]
json.dump(out, open("$B/q_anchors.json", "w")); print("앵커 목록 %d채 %d장" % (len(out), sum(len(v) for v in out.values())))
PY
  one() { hn=$1; S=data/seq/newsim_$hn; W=$HOME/khcache/hloc-newsim/$hn; NM=$($K -c "import json; print(json.load(open('$S/camera_info.json'))['n_map'])" 2>/dev/null)
    $K -u scripts/reloc_hloc.py $S --scan-end $NM --live-step 1 --work $W --map gt --embed clip --topk 5 --threads 4 --live-list $B/q_anchors.json --house-name $hn \
      --pose-out $B/pnp/pose_$hn.jsonl --hssd-mirror $MIRROR --min-inliers 50 > $B/pnp/logs/$hn.log 2>&1
    echo "  $hn $(grep -aE 'GT 포즈 삼각측량|라이브 PnP' $B/pnp/logs/$hn.log | sed -E 's/^\[ *[0-9]+s\] //; s/ · 장당.*//' | tr '\n' ' ' | cut -c1-200)"
    :; }
  export -f one; export K B MIRROR
  ls -d $OUT/house_* | xargs -n1 basename | xargs -P $PAR -I{} bash -c 'one {}'
  cat $B/pnp/pose_house_*.jsonl > $B/pnp/pose_all.jsonl; echo "  POSE_JSONL $(wc -l < $B/pnp/pose_all.jsonl)줄"
  echo "  ⚠️ '삼각측량 지도' 줄의 재투영 오차가 2 px 를 넘으면 MIRROR 를 바꿔(0↔1) 9단계만 다시 (좌표 손 규약)."
  for H in $OUT/house_*; do hn=$(basename $H); $K scripts/export_map_points.py $H --work $HOME/khcache/hloc-newsim/$hn --seq data/seq/newsim_$hn --out $HOME/khcache/mappts/$hn 2>&1 | grep -v Warn | tail -1; done; }
[ $STEP -le 10 ] && [ "${SKIP_ABSV:-0}" != 1 ] && { echo "=== 9-b. 부재 검증기 (mlx, 포즈+CLIP 문맥) $(date +%H:%M) ==="
  THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ ROOM_JSONL=$B/scores/room_embed_clip.jsonl INITMAP_FILE=initmap_owl_rc.json POSE_JSONL=$B/pnp/pose_all.jsonl RETR=both TOPN=8 OUT_JSONL=$B/scores/abs_verify_rc_both.jsonl $MLX -u scripts/abs_verify_mlx.py 2>&1 | grep -aE "→|Traceback" | tail -2; }
[ $STEP -le 10 ] && { echo "=== 10. 벤치 D (GT 0) $(date +%H:%M) ==="
  env BENCH_DIR=$B THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ VERIFY_JSONL=$B/scores/t1_floor0.8_d40.jsonl \
    GEO_DEPTH=$B/scores/geo_depth_nogt.jsonl ROOM_GROUPS=0 PY=$K POSE_JSONL=$B/pnp/pose_all.jsonl ROOM_JSONL=$B/scores/room_embed_clip.jsonl ROWS_OUT=$B/rows_D.jsonl \
    INITMAP_FILE=${INITMAP_FILE:-initmap_owl_rc.json} ROI_DIST=${ROI_DIST:-0} ROI_BOX=${ROI_BOX:-0} PHANTOM_JSON= ABS_VERIFY_JSONL=$B/scores/abs_verify_rc_both.jsonl ABSV_MOB=0.1 ABSV_GEOONLY=1 ABSV_TP=0 ABSV_EARLY_MIN=1.0 ABS_OVER_C0=0 ROWS_OUT=$B/rows_D.jsonl bash scripts/bench.sh > $B/bench_D_full.log 2>&1
  grep -aE "재료 사다리|GT 재료|최종 답|belief 순위|^\s*(①|②|③|④)[^ ]* +n=" $B/bench_D_full.log | cut -c1-200; }
echo "NOGT_CHAIN_NEWSIM_DONE $(date +%H:%M)"
