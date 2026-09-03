#!/usr/bin/env bash
# 20-house og20 generation: SPEC 4-4 walking realism, --evidence 3, 3400 frames.
# Scenes are the seven that og_scene_screen.py passed (4-8 indoor rooms with a kitchen
# and a bathroom).  --case4 > 0 only where the scene has rooms outside the episode
# scope to carry an object into (handover section 12, option 1).
#
# Resume is by EPISODE_OK in the per-house log, not by gt.json: a gate failure also
# writes gt.json, so keying on the file would silently accept failed houses.
set -u
WORK=/mnt/ssd2/wooyeol/work
PY=$WORK/behavior1k_latest/conda_envs/behavior51/bin/python
GEN=$WORK/belief_main_20260902/scripts/og_episode.py
OUT=$WORK/og20
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_GPU_ID=1 OMNIGIBSON_HEADLESS=1 PYTHONNOUSERSITE=1
export OMNIGIBSON_DATA_PATH=$WORK/behavior1k_latest/og_data_2026
export OMNIGIBSON_APPDATA_PATH=$WORK/behavior1k_latest/appdata_og51
mkdir -p "$OUT"

PLAN="Rs_int 0 0
Pomaria_1_int 0 0
Merom_1_int 0 0
Wainscott_0_int 0 2
Ihlen_1_int 0 2
Wainscott_0_garden 0 2
house_single_floor 0 2
Rs_int 1 0
Pomaria_1_int 1 0
Merom_1_int 1 0
Wainscott_0_int 1 2
Ihlen_1_int 1 2
Wainscott_0_garden 1 2
house_single_floor 1 2
Rs_int 2 0
Pomaria_1_int 2 0
Merom_1_int 2 0
Wainscott_0_int 2 2
Ihlen_1_int 2 2
house_single_floor 2 2"

H=0
while read -r S SEED C4; do
  [ -z "${S:-}" ] && continue
  TAG=$(printf 'house_%04d' "$H")
  LOG="$OUT/$TAG.log"
  if grep -q EPISODE_OK "$LOG" 2>/dev/null; then
    echo "SKIP $TAG ($S seed $SEED) already OK"
    H=$((H + 1)); continue
  fi
  rm -rf "$OUT/$TAG"
  echo "=== $TAG  $S  seed=$SEED  case4=$C4 ==="
  timeout 7200 "$PY" -u "$GEN" --scene "$S" --out "$OUT/$TAG" --house "$H" --seed "$SEED" \
    --frames 3400 --props 24 --case1 12 --case2 5 --case3 5 --case4 "$C4" \
    --spare 5 --evidence 3:1.4 --map-sites 3 --map-step 45 \
    --dwell 30 --pace 0.25 --turn 0.35 --scan 35 --max-turn 25 \
    > "$LOG" 2>&1
  RC=$?
  OK=$(grep -c EPISODE_OK "$LOG")
  FAIL=$(grep -o 'GATE FAILED.*' "$LOG" | head -1 | cut -c1-170)
  echo "HOUSE $TAG $S seed=$SEED rc=$RC ok=$OK $FAIL"
  H=$((H + 1))
done <<< "$PLAN"
echo GEN20_COMPLETE
