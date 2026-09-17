#!/usr/bin/env bash
# 부재 검증기·타임라인을 **벤치가 고른 기록 인스턴스** 위에서 돌린다 (2026-09-18, §166-83).
#
# 왜: 종전에는 검증기(abs_verify_mlx.py)와 타임라인(timeline_prep.py)이 초기맵 **첫 인스턴스**를
#     독립적으로 골라, 벤치가 답으로 쓰는 인스턴스와 **다른 자리**를 봤다. c3big 에서 ③ 미인계
#     14건 중 8건이 그 자리에서 3.8~12.9 m 어긋나 "거기 있다"는 오답을 냈다.
# 방법: 벤치를 한 번 돌려 REC_DUMP 로 (집·물체·기록방·자리) 를 뽑고, 그 파일을 INST_SEL_JSONL 로
#     검증기·타임라인에 먹인다. c3big 69채 실측: 총 0.623→0.628 · ① 0.797→0.802 · ③인계 0.61→0.63
#     (모든 축에서 공짜).
#
#   OUT=data/hssd_c3big BENCH_DIR=~/khcache/bench-c3big EMB_CACHE=~/khcache/room_embed-hssd_c3big \
#     INITMAP_FILE=initmap_owl.json POSE_JSONL=.../pose_all.jsonl VERIFY_JSONL=.../t1_m0.jsonl \
#     bash scripts/absv_sel.sh
# → $B/rec_sel.jsonl · $B/scores/abs_verify_sel.jsonl · $B/scores/timeline_sel.jsonl
set -u; cd "$(dirname "$0")/.."
OUT=${OUT:?데이터셋}; B=${BENCH_DIR:?벤치 디렉터리}
K=${PY:-$HOME/kx-venv/bin/python}; MLX=${MLXPY:-$HOME/mlx-venv/bin/python}
IM=${INITMAP_FILE:-initmap_owl.json}; PJ=${POSE_JSONL:?포즈}; VJ=${VERIFY_JSONL:?검증점수}
RG=${ABS_ROOMGATE:-either}; EC=${EMB_CACHE:-$B/emb}; TL=${DO_TIMELINE:-1}; LOS=${LOS:-0}
echo "=== absv_sel 1/3 기록 선택 덤프 $(date +%H:%M) ==="
rm -f $B/rec_sel.jsonl
env ${EXTRA:-FOO=1} BENCH_DIR=$B THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ \
  GEO_DEPTH=$B/scores/geo_depth_all.jsonl ROOM_GROUPS=1 PY=$K ROOM_JSONL=$B/scores/room_embed_clip.jsonl \
  INITMAP_FILE=$IM POSE_JSONL=$PJ VERIFY_JSONL=$VJ ROI_DIST=0 ROI_BOX=0 REC_DUMP=$B/rec_sel.jsonl \
  bash scripts/bench.sh > $B/absv_sel_dump.log 2>&1
echo "  덤프 $(wc -l < $B/rec_sel.jsonl 2>/dev/null || echo 0)행 · 자리 있는 행 $(grep -c '"rec_pos": \[' $B/rec_sel.jsonl 2>/dev/null || echo 0)"
[ -s $B/rec_sel.jsonl ] || { echo "  ✗ 덤프 비었다 — 중단"; grep -aE "Traceback" -A3 $B/absv_sel_dump.log | tail -5; exit 1; }
echo "=== absv_sel 2/3 부재 검증기 $(date +%H:%M) ==="
rm -f $B/scores/abs_verify_sel.jsonl
INST_SEL_JSONL=$B/rec_sel.jsonl ABS_ROOMGATE=$RG EMB_CACHE=$EC THOR_ROOT=$OUT \
  A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ ROOM_JSONL=$B/scores/room_embed_clip.jsonl \
  INITMAP_FILE=$IM POSE_JSONL=$PJ RETR=${RETR:-both} TOPN=${TOPN:-8} \
  OUT_JSONL=$B/scores/abs_verify_sel.jsonl $MLX -u scripts/abs_verify_mlx.py > $B/abs_verify_sel.log 2>&1
echo "  부재 $(wc -l < $B/scores/abs_verify_sel.jsonl 2>/dev/null || echo 0)줄 · $(grep -aE 'Traceback' $B/abs_verify_sel.log | tail -1 | cut -c1-80)"
[ "$TL" = "1" ] || { echo "ABSV_SEL_DONE $(date +%H:%M)"; exit 0; }
echo "=== absv_sel 3/3 타임라인 LOS$LOS $(date +%H:%M) ==="
INST_SEL_JSONL=$B/rec_sel.jsonl THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ \
  ROOM_JSONL=$B/scores/room_embed_clip.jsonl POSE_JSONL=$PJ INITMAP_FILE=$IM LOS=$LOS \
  OUT_JSONL=$B/scores/timeline_prep_sel.jsonl $K -u scripts/timeline_prep.py > $B/tl_prep_sel.log 2>&1
rm -f $B/scores/timeline_sel.jsonl
THOR_ROOT=$OUT PREP_JSONL=$B/scores/timeline_prep_sel.jsonl OUT_JSONL=$B/scores/timeline_sel.jsonl \
  MODE=simple PREFILL=1 $MLX -u scripts/timeline_verdict_mlx.py > $B/tl_verdict_sel.log 2>&1
echo "  타임라인 $(wc -l < $B/scores/timeline_sel.jsonl 2>/dev/null || echo 0)줄"
echo "ABSV_SEL_DONE $(date +%H:%M)"
