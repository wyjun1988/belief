#!/usr/bin/env bash
# 여러 데이터셋·벤치 디렉터리를 하나로 합쳐 통합 벤치를 돈다 — 집 이름이 겹치지 않아야 한다(house_0000~0030 · 0031~0090 · 0131~0190).
#   COMBO=data/hssd150_all BCOMBO=~/khcache/bench-h150 bash scripts/merge_bench_dirs.sh data/hssd40_c3:~/khcache/bench-h40c3 data/hssd90_c4:~/khcache/bench-h90c4 data/hssd90_c4e2:~/khcache/bench-h90c4e2
# 캐시(hs2_a/q/x_*.npz)는 심볼릭 링크, jsonl(검증·거리·방·PnP)은 이어붙임. 초기맵·room_groups 는 집 디렉터리 안에 있어 그대로 따라온다.
set -u; cd "$(dirname "$0")/.."
COMBO=${COMBO:-data/hssd150_all}; BC=${BCOMBO:-$HOME/khcache/bench-h150}; K=${PY:-$HOME/kx-venv/bin/python}
rm -rf "$COMBO" "$BC"; mkdir -p "$COMBO" "$BC/cache" "$BC/scores" "$BC/pnp"
for pair in "$@"; do D=${pair%%:*}; B=${pair##*:}; B=$(eval echo $B)
  for H in $D/house_*; do ln -s "$(cd $D && pwd)/$(basename $H)" "$COMBO/$(basename $H)"; done
  for f in $B/cache/hs2_*_house_*.npz; do [ -e "$f" ] && ln -s "$f" "$BC/cache/$(basename $f)"; done
  [ -f $B/scores/t1_floor0.8_d40.jsonl ] && cat $B/scores/t1_floor0.8_d40.jsonl >> $BC/scores/t1_floor0.8_d40.jsonl
  [ -f $B/scores/geo_depth_nogt.jsonl ] && cat $B/scores/geo_depth_nogt.jsonl >> $BC/scores/geo_depth_nogt.jsonl
  [ -f $B/scores/room_embed_clip.jsonl ] && cat $B/scores/room_embed_clip.jsonl >> $BC/scores/room_embed_clip.jsonl
  P=$B/pnp/pose_all.jsonl; [ -f $B/pnp_anc/pose_all.jsonl ] && P=$B/pnp_anc/pose_all.jsonl; [ -f $P ] && cat $P >> $BC/pnp/pose_all.jsonl
  echo "  $D ← $(ls -d $D/house_* | wc -l | tr -d ' ')채 · 캐시 $(ls $B/cache/hs2_a_*.npz 2>/dev/null | wc -l | tr -d ' ')"
done
echo "통합: $(ls $COMBO | wc -l | tr -d ' ')채 · 검증 $(wc -l < $BC/scores/t1_floor0.8_d40.jsonl)줄 · 방 $(wc -l < $BC/scores/room_embed_clip.jsonl)줄 · 포즈 $(wc -l < $BC/pnp/pose_all.jsonl)줄"
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
env BENCH_DIR=$BC THOR_ROOT=$COMBO A3_PREFIX=$BC/cache/hs2_a_ QC_PREFIX=$BC/cache/hs2_q_ AX_PREFIX=$BC/cache/hs2_x_ VERIFY_JSONL=$BC/scores/t1_floor0.8_d40.jsonl \
  GEO_DEPTH=$BC/scores/geo_depth_nogt.jsonl ROOM_GROUPS=1 PY=$K POSE_JSONL=$BC/pnp/pose_all.jsonl ROOM_JSONL=$BC/scores/room_embed_clip.jsonl ROWS_OUT=$BC/rows_D.jsonl \
  bash scripts/bench.sh > $BC/bench_D_full.log 2>&1
grep -aE "재료 사다리|GT 재료|최종 답|^\s*(①|②|③|④)[^ ]* +n=" $BC/bench_D_full.log | cut -c1-200
echo MERGE_BENCH_DONE
