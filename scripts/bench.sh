#!/usr/bin/env bash
# bench-v1 러너 (AUDIT_20260902 조치2) — 동결 데이터 + 동결 챔피언 설정.
# 바꾸고 싶은 노브 **하나만** env 로 덮어쓴다:   VERIFY_JSONL=... bash scripts/bench.sh
# 데이터·캐시·점수는 $BENCH_DIR 아래 동결본을 쓴다 (/tmp 아님). docs/BENCH_V1.md 참조.
set -e
cd "$(dirname "$0")/.."
B=${BENCH_DIR:-$HOME/khcache/bench-v1}
export THOR_ROOT=${THOR_ROOT:-data/hssd20S}
export A3_PREFIX=${A3_PREFIX:-$B/cache/hsc_a_}
export QC_PREFIX=${QC_PREFIX:-$B/cache/hsc_q_}
export AX_PREFIX=${AX_PREFIX:-$B/cache/hsc_x_}
export FRAME_W=${FRAME_W:-768}
export SG_INIT=${SG_INIT:-hybrid}
export LOC_GEO=${LOC_GEO:-1}
export LOC_YAW_GT=${LOC_YAW_GT:-1}      # ⚠️ GT 포즈 — 실물 사슬(투표 yaw) 복원 전까지
export LOC_TRI=${LOC_TRI:-1}
export ABS_TH=${ABS_TH:-0.100}
export VERIFY_JSONL=${VERIFY_JSONL:-$B/scores/t1_floor0.8_d40.jsonl}
export VERIFY_TH=${VERIFY_TH:-2.069}
export VERIFY_TH2=${VERIFY_TH2:-0.887}
export C0_MIN=${C0_MIN:-1}
export PRIOR_JSON=${PRIOR_JSON:-data/hssd_move.json}   # HSSD 어휘 (thor_prior 는 전부 미등록이었다)
PY=${PY:-$HOME/kx-venv/bin/python}
echo "bench-v1 · 덮어쓴 노브: $(env | grep -E '^(VERIFY_JSONL|VERIFY_TH|VERIFY_TH2|C0_MIN|ABS_TH|LOC_YAW_GT|LOC_TRI|SG_INIT|GEO_DEPTH|PRIOR_JSON)=' | tr '\n' ' ')"
$PY scripts/eval_online.py 2>/dev/null | grep -vE "^\s*$"
