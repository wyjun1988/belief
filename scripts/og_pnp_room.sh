#!/usr/bin/env bash
# OG: 같은 방 후보 보강 PnP (top5 + 같은 방 top5, 인라이어 50 유지) — 문턱을 풀지 않고 검색만 넓힌다 (HSSD 파일럿에서 ①② 유지·포즈 +87). 2026-09-11
#   OUT=<OG 루트> BENCH_DIR=<B> [MIRROR=0 PAR=3] bash scripts/og_pnp_room.sh
set -u; cd "$(dirname "$0")/.."; OUT=${OUT:?}; B=${BENCH_DIR:?}; K=${PY:-python}; MIRROR=${MIRROR:-0}; PAR=${PAR:-3}; SEQP=${SEQ_PREFIX:-og_$(basename $OUT)}; mkdir -p $B/roomjson $B/pnp/logs
THOR_ROOT=$OUT ROOM_JSONL=$B/scores/room_embed_clip.jsonl OUT_DIR=$B/roomjson $K scripts/make_room_json.py 2>&1 | tail -1
one() { hn=$1; S=data/seq/${SEQP}_$hn; W=$HOME/khcache/hloc-${SEQP}/$hn; NM=$($K -c "import json; print(json.load(open('$S/camera_info.json'))['n_map'])" 2>/dev/null)
  $K -u scripts/reloc_hloc.py $S --scan-end $NM --live-step 1 --work $W --map gt --embed clip --topk 5 --threads 4 --live-list $B/q_anchors.json --house-name $hn \
    --pose-out $B/pnp/poseroom_$hn.jsonl --hssd-mirror $MIRROR --min-inliers ${MIN_INLIERS:-50} --room-json $B/roomjson/room_$hn.json > $B/pnp/logs/room_$hn.log 2>&1
  echo "  $hn $(grep -aE '방 후보 보강|라이브 PnP' $B/pnp/logs/room_$hn.log | sed -E 's/^\[ *[0-9]+s\] //; s/ · 장당.*//' | tr '\n' ' ' | cut -c1-170)"; }
export -f one; export K B MIRROR SEQP MIN_INLIERS
ls -d $OUT/house_* | xargs -n1 basename | xargs -P $PAR -I{} bash -c 'one {}'
cat $B/pnp/poseroom_house_*.jsonl > $B/pnp/pose_all_room.jsonl; echo "POSE_JSONL room $(wc -l < $B/pnp/pose_all_room.jsonl)줄 (플레인 $(wc -l < $B/pnp/pose_all.jsonl))"
$K scripts/pose_check_conventions.py $OUT $B/pnp/pose_all_room.jsonl | head -3
