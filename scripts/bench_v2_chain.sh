#!/usr/bin/env bash
# bench-v2 전체 체인 (이식 가능판) — 생성 → 기하 자가검사 게이트 → 캐시 → 초기맵 → [캘리브] → 채점 → 평가 2회
#
#   Linux/CUDA (4090·RTX PRO 6000):
#     HSSD_DATASET=~/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json \
#     HAB=~/miniforge3/envs/hab/bin/python PY=~/kx-venv/bin/python SCORER=hf CALIB=1 \
#     bash scripts/bench_v2_chain.sh
#   Apple Silicon: SCORER=mlx (기본 자동 판별). 로그는 stdout 에, 완료 표식 BENCH_V2_DONE.
#
# 채점기가 다르면(mlx 4B vs HF 9B) 검증기 문턱은 **그 머신에서 다시 잰다** (§89: CALIB=1).
# 데이터·설정은 같으므로 나머지 수치는 머신 간 비교 가능하다.
set -e
cd "$(dirname "$0")/.."
HSSD_DATASET=${HSSD_DATASET:-$HOME/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json}
HAB=${HAB:-$HOME/miniforge3/envs/hab/bin/python}
PY=${PY:-$HOME/kx-venv/bin/python}
OUT=${OUT:-data/hssd20S2}
B2=${BENCH_DIR:-$HOME/khcache/bench-v2}
SCENES=${SCENES:-docs/bench/hssd20_scenes.txt}
FRAMES=${FRAMES:-1200}
if [ -z "$SCORER" ]; then
  if [ "$(uname)" = Darwin ] && [ -x "$HOME/mlx-venv/bin/python" ]; then SCORER=mlx; else SCORER=hf; fi
fi
MLX=${MLX:-$HOME/mlx-venv/bin/python}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
mkdir -p "$B2/cache" "$B2/scores"
[ -f "$HSSD_DATASET" ] || { echo "HSSD_DATASET 없음: $HSSD_DATASET (HF hssd-hab 내려받기)"; exit 2; }
[ -f data/hssd_move.json ] || { echo "data/hssd_move.json 없음"; exit 2; }

# ── 1. 생성 (기하 게이트·증인 렌더 포함) ──
if [ "${SKIP_GEN:-0}" != 1 ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"; i=0
  for SC in $(cat "$SCENES"); do
    $HAB -u scripts/hab_episode.py --scene "$SC" --dataset "$HSSD_DATASET" --move data/hssd_move.json \
      --frames "$FRAMES" --moves 8 --case3 "${CASE3:-0.5}" --far "${FAR:-0.0}" --seed $((i+200)) --out "$OUT/house_$(printf %04d $i)" 2>&1 \
      | grep -E "이동 후보|이동 기록|이동 취소|증인 렌더" || echo "  $SC 실패"
    i=$((i+1))
  done
fi
echo "=== 자가검사 (기하 게이트) ==="
N=20 $PY scripts/gen_selfcheck.py "$OUT"/house_* 2>/dev/null || { echo BENCH_V2_FAIL_SELFCHECK; exit 1; }

# ── 2. 캐시 (박스 포함) · 초기맵 ──
THOR_ROOT=$OUT CACHE_PREFIX=$B2/cache/hs2_a_ BOXES=1 $PY -u scripts/exp_anchowl.py 4
THOR_ROOT=$OUT QCACHE_PREFIX=$B2/cache/hs2_q_ STRIDE=4 $PY -u scripts/exp_imgq.py
THOR_ROOT=$OUT ACACHE_PREFIX=$B2/cache/hs2_x_ STRIDE=4 $PY -u scripts/exp_anchor_exemplar.py
$PY -c "import numpy as np,glob; z=np.load(sorted(glob.glob('$B2/cache/hs2_a_*.npz'))[0]); print('bx 저장:', 'bx' in z.files)"
THOR_ROOT=$OUT A3_PREFIX=$B2/cache/hs2_a_ INITMAP_GEO=1 INITMAP_INST=1 $PY -u scripts/build_initmap.py | tail -3

# ── 3. 검증기 문턱 캘리브 (채점기가 바뀌면 필수) ──
if [ "${CALIB:-0}" = 1 ]; then
  THOR_ROOT=$OUT OUT=$B2/pairs N=500 $PY scripts/make_sim_pairs.py
  if [ "$SCORER" = mlx ]; then
    PAIRS=$B2/pairs OUT_JSONL=$B2/scores/calib.jsonl $MLX scripts/exp_vlm_verify3_mlx.py
  else
    MODEL=$MODEL PAIRS=$B2/pairs OUT_JSONL=$B2/scores/calib.jsonl $PY scripts/exp_vlm_verify3.py
  fi
  read VERIFY_TH VERIFY_TH2 <<<"$($PY - <<PYC
import json, numpy as np
r=[json.loads(l) for l in open("$B2/scores/calib.jsonl")]
neg=[x for x in r if not x["truth"]]
print(round(float(np.quantile([x["s_ab"] for x in neg],0.95)),3), round(float(np.quantile([x["s_ac"] for x in neg],0.90)),3))
PYC
)"
  export VERIFY_TH VERIFY_TH2
  echo "캘리브 → VERIFY_TH=$VERIFY_TH (s_ab 기각0.95) · VERIFY_TH2=$VERIFY_TH2 (s_ac 기각0.90)"
  $PY scripts/rtx7_sweep.py $B2/scores/calib.jsonl --pick 0.95 | head -6
fi
export VERIFY_TH=${VERIFY_TH:-2.069} VERIFY_TH2=${VERIFY_TH2:-0.887}

score() {  # $1=FLOOR $2=MAXWALK $3=out
  if [ "$SCORER" = mlx ]; then
    THOR_ROOT=$OUT A3_PREFIX=$B2/cache/hs2_a_ QC_PREFIX=$B2/cache/hs2_q_ OUT_JSONL=$3 FLOOR=$1 MAXWALK=$2 $MLX -u scripts/exp_t1_verify_mlx.py | tail -1
  else
    THOR_ROOT=$OUT A3_PREFIX=$B2/cache/hs2_a_ QC_PREFIX=$B2/cache/hs2_q_ OUT_JSONL=$3 FLOOR=$1 MAXWALK=$2 MODEL=$MODEL $PY -u scripts/exp_t1_verify_pipeline.py | tail -1
  fi
}
bench() {  # $1=scores jsonl  $2=label
  echo "=== bench-v2 $2 ==="
  BENCH_DIR=$B2 THOR_ROOT=$OUT A3_PREFIX=$B2/cache/hs2_a_ QC_PREFIX=$B2/cache/hs2_q_ AX_PREFIX=$B2/cache/hs2_x_ \
    VERIFY_JSONL=$1 PY=$PY bash scripts/bench.sh
}
# ── 4. 챔피언 (floor 0.8 · depth 40) ──
score 0.8 40 "$B2/scores/t1_floor0.8_d40.jsonl"
(shasum -a 256 "$OUT"/house_*/gt.json "$OUT"/house_*/initmap_owl.json; ls "$B2/cache") > "$B2/MANIFEST.sha256"
bench "$B2/scores/t1_floor0.8_d40.jsonl" "run0 챔피언 (SCORER=$SCORER)"
# ── 5. 단일 노브: 후보 문턱 제거 (§124) ──
score 0.0 160 "$B2/scores/t1_floor0.0_d160.jsonl"
bench "$B2/scores/t1_floor0.0_d160.jsonl" "run1 단일노브 VERIFY_JSONL=floor0.0_d160"
echo BENCH_V2_DONE   # (BENCH_DIR/OUT 로 v3 이상도 같은 스크립트)
