#!/usr/bin/env bash
# 데이터셋별 챔피언 벤치 구성 — 한 곳에 고정 (2026-09-23).
#
# 왜: 챔피언 노브가 스크래치패드 스크립트(maxd_sweep.sh 등) 안에만 있어, 새 실험마다 그 머리를 sed 로 복사했다.
#     복사가 실행문까지 딸려 오거나(함정 9) 스크래치패드가 사라지면 재현이 안 된다. 요약 줄도 ① 채택 전부를
#     "거짓채택" 으로 세고 있었다(bench_rep.py 로 교체).
#
# 사용: source scripts/champion_bench.sh; run_hssd <tag> [KEY=VAL ...]   # KEY=VAL 이 기본값을 덮는다
#       run_c3big / run_c2set / run_newsim4 도 같은 모양. 행 파일 = $BENCH_DIR/rows_CH_<tag>.jsonl
cd ~/work/khronos; export KMP_DUPLICATE_LIB_OK=TRUE
K=${K:-$HOME/kx-venv/bin/python}
_rep(){ [ -s "$2" ] && $K scripts/bench_rep.py "$1" "$2" || { echo "   ✗ $1"; grep -aE "Traceback|✗" -A2 "${2%.jsonl}.log" 2>/dev/null | tail -3 | cut -c1-140; }; }
_bench(){ tag=$1; BD=$2; shift 2; rows=$BD/rows_CH_$tag.jsonl; rm -f $rows
  env "$@" PY=$K ROWS_OUT=$rows bash scripts/bench.sh > $BD/rows_CH_$tag.log 2>&1; _rep "$tag" "$rows"; }

# HSSD 133채 — 2026-09-23 챔피언: 별칭 캐시 + 옛 초기맵에 stand 만 이식(initmap_k8_merge) + 판정기 공백 메운 필터
V=$HOME/khcache/bench-v2full; VA=$HOME/khcache/bench-v2alias
run_hssd(){ tag=$1; shift; _bench "$tag" $V \
  INITMAP_FILE=initmap_k8_merge.json INST_ANG=10 ABSV_MOB=0.15 ABSV_GEOONLY=1 ABSV_TP=0 ABSV_EARLY_MIN=1.0 ABS_OVER_C0=0 ROI_DIST=0 ROI_BOX=0 C0_MAXD=4.0 \
  POSE_JSONL=$V/pnp/pose_all_relax.jsonl ABS_VERIFY_JSONL=$V/scores/abs_verify_room_either.jsonl ABS_ROOMGATE=either \
  BENCH_DIR=$V THOR_ROOT=data/hssd_v2 A3_PREFIX=$VA/cache/hs2_a_ QC_PREFIX=$VA/cache/hs2_q_ AX_PREFIX=$VA/cache/hs2_x_ \
  VERIFY_JSONL=$VA/scores/t1_champ_allfull_v2.jsonl GEO_DEPTH=$VA/scores/geo_depth_nogt.jsonl ROOM_GROUPS=1 ROOM_JSONL=$V/scores/room_embed_clip.jsonl "$@"; }
# HSSD 옛 챔피언(9/21) — 비교 기준
run_hssd_old(){ tag=$1; shift; run_hssd "$tag" INITMAP_FILE=initmap_k8.json A3_PREFIX=$V/cache/hs2_a_ QC_PREFIX=$V/cache/hs2_q_ AX_PREFIX=$V/cache/hs2_x_ \
  VERIFY_JSONL=$V/scores/t1_champ_allfull.jsonl GEO_DEPTH=$V/scores/geo_depth_all.jsonl "$@"; }

# c3big 69채 — ③ 정식 출처 (옛 포즈 · 제로샷 부재 · 타임라인 2차 의견)
C3=$HOME/khcache/bench-c3big
run_c3big(){ tag=$1; shift; _bench "$tag" $C3 \
  INITMAP_FILE=initmap_owl.json INST_ANG=10 ABSV_MOB=0.15 ABSV_TP=0 ABS_OVER_C0=0 ROI_DIST=0 ROI_BOX=0 ABS_ROOMGATE=either ABSV_GEOONLY=0 ABSV_EARLY_MIN=1.0 \
  POSE_JSONL=$C3/pnp/pose_all.jsonl ABS_VERIFY_JSONL=$C3/scores/abs_verify_sel.jsonl BUNDLE_JSONL=$C3/scores/timeline_sel.jsonl BUNDLE_MODE=abs BUNDLE_TH=70 BUNDLE_MIN_SPOT=2 \
  BENCH_DIR=$C3 THOR_ROOT=data/hssd_c3big A3_PREFIX=$C3/cache/hs2_a_ QC_PREFIX=$C3/cache/hs2_q_ AX_PREFIX=$C3/cache/hs2_x_ \
  VERIFY_JSONL=$C3/scores/t1_m0.jsonl GEO_DEPTH=$C3/scores/geo_depth_all.jsonl ROOM_GROUPS=1 ROOM_JSONL=$C3/scores/room_embed_clip.jsonl "$@"; }

# c2set 69채 — ② 확인용 (9/23 기준선: k8 재군집 + 완화 PnP + 9-61 전 프레임 마진으로 만든 필터, 마진 없으면 버림)
C2=$HOME/khcache/bench-c2set
run_c2set(){ tag=$1; shift; _bench "$tag" $C2 \
  INITMAP_FILE=initmap_k8.json INST_ANG=10 ROI_DIST=0 ROI_BOX=0 C0_MAXD=4.0 POSE_JSONL=$C2/pnp2/pose_all.jsonl \
  BENCH_DIR=$C2 THOR_ROOT=data/hssd_c2set A3_PREFIX=$C2/cache/hs2_a_ QC_PREFIX=$C2/cache/hs2_q_ AX_PREFIX=$C2/cache/hs2_x_ \
  VERIFY_JSONL=$C2/scores/t1_champ_fix.jsonl GEO_DEPTH=$C2/scores/geo_depth_all.jsonl ROOM_GROUPS=1 ROOM_JSONL=$C2/scores/room_embed_clip.jsonl "$@"; }

# 새 시뮬 4차분 10채 — 새 시뮬 이동성 표 · 1280 · 기록자리 판정기 PLACE_W 2
N4=$HOME/khcache/bench-ns4b
run_newsim4(){ tag=$1; shift; _bench "$tag" $N4 \
  PRIOR_JSON=data/newsim_move.json MOVABLE_MIN=0.1 FRAME_W=1280 INITMAP_FILE=initmap_owl.json INST_ANG=10 ROI_DIST=0 ROI_BOX=0 \
  PLACE_JSONL=$HOME/khcache/bench-ns4/place_margin.jsonl PLACE_W=2 VERIFY_JSONL=$N4/scores/t1_floor0.8_d40.jsonl \
  BENCH_DIR=$N4 THOR_ROOT=data/newsim4 A3_PREFIX=$N4/cache/hs2_a_ QC_PREFIX=$N4/cache/hs2_q_ AX_PREFIX=$N4/cache/hs2_x_ \
  GEO_DEPTH=$N4/scores/geo_depth_nogt.jsonl ROOM_GROUPS=1 ROOM_JSONL=$N4/scores/room_embed_clip.jsonl POSE_JSONL=$N4/pnp/pose_all.jsonl "$@"; }

# 새 시뮬 5차분 15채 (3장면 × 5에피) — 4차분 표준 구성과 같음. 캐시·초기맵·기록자리 마진은 프로6000(9-59, CUDA), PnP·검증·거리는 M2
N5=$HOME/khcache/bench-ns5
run_newsim5(){ tag=$1; shift; _bench "$tag" $N5 \
  PRIOR_JSON=data/newsim_move.json MOVABLE_MIN=0.1 FRAME_W=1280 INITMAP_FILE=initmap_owl.json INST_ANG=10 ROI_DIST=0 ROI_BOX=0 \
  PLACE_JSONL=$N5/place_margin.jsonl PLACE_W=2 VERIFY_JSONL=$N5/scores/t1_floor0.8_d40.jsonl \
  BENCH_DIR=$N5 THOR_ROOT=data/newsim5 A3_PREFIX=$N5/cache/hs2_a_ QC_PREFIX=$N5/cache/hs2_q_ AX_PREFIX=$N5/cache/hs2_x_ \
  GEO_DEPTH=$N5/scores/geo_depth_nogt.jsonl ROOM_GROUPS=1 ROOM_JSONL=$N5/scores/room_embed_clip.jsonl POSE_JSONL=$N5/pnp/pose_all.jsonl "$@"; }

