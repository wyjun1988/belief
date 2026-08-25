#!/bin/bash
# 4090 한 번에 실행. 목적은 **경우 2(부재 → belief) 표본 확보**다.
#   현재 아이맥 데이터: 이동 45개 중 원래 방 재방문 18개, 앵커 조건 충족 7개 → 통계 불가
#   여기서 노리는 것: 긴 지평 + 많은 이동으로 경우 2 를 수백 개로
#
# 사용:  bash scripts/gpu_4090_run.sh            (기본: 60채 × 3시간)
#        HOUSES=100 HOURS=4 bash scripts/gpu_4090_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

HOUSES=${HOUSES:-60}
HOURS=${HOURS:-3}
MOVES=${MOVES:-30}          # 1시간 8건 → 3시간 30건. 경우 2 표본이 여기서 늘어난다
DWELL=${DWELL:-600}
SEED=${SEED:-101}
OUT=${OUT:-data/thor4}
STRIDE=${STRIDE:-4}         # ⚠️ 프레임을 두고 오므로 여유 있게. 8 로 하면 되돌릴 수 없다
PY=${PY:-python}

echo "=== 사전 점검 ==="
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -m1 deviceName \
  || echo "⚠️ vulkaninfo 없음 — CloudRendering 이 실패하면 이것부터 확인"
$PY -c "import ai2thor, prior; print('ai2thor', ai2thor.__version__)"
test -f data/thor_prior.json || { echo "❌ data/thor_prior.json 없음"; exit 1; }
test -f data/thor_move.json  || { echo "❌ data/thor_move.json 없음";  exit 1; }

echo "=== [1/4] 생성 $HOUSES 채 × ${HOURS}시간 · 이동 $MOVES 건  $(date +%H:%M) ==="
$PY -u scripts/thor_gen2.py --houses "$HOUSES" --hours "$HOURS" --moves "$MOVES" \
  --dwell "$DWELL" --seed "$SEED" --vis-dist 20 --platform cloud \
  --min-rooms 4 --max-rooms 8 --max-nonbath 6 --min-nonbath 2 \
  --prior data/thor_prior.json --move data/thor_move.json --out "$OUT"

echo "=== [2/4] 앵커 캐시 (stride $STRIDE)  $(date +%H:%M) ==="
THOR_ROOT="$OUT" CACHE_PREFIX=/tmp/a3_ $PY -u scripts/exp_anchowl.py "$STRIDE"

echo "=== [3/4] 이미지·글자 질의 캐시  $(date +%H:%M) ==="
THOR_ROOT="$OUT" QCACHE_PREFIX=/tmp/qc_ STRIDE="$STRIDE" $PY -u scripts/exp_imgq.py

echo "=== [3.5/4] 앵커 개체 exemplar 캐시  $(date +%H:%M) ==="
# 앵커를 글자(타입)로만 찾으면 "어느 의자인지" 를 모른다. 그 탓에 배치 증거·2차 검색이
# 모두 막혔다(0.644→0.613, 0.625→0.522). 개체 exemplar 는 재생성 없이 만들 수 있다.
THOR_ROOT="$OUT" ACACHE_PREFIX=/tmp/ax_ STRIDE="$STRIDE" $PY -u scripts/exp_anchor_exemplar.py

echo "=== [4/4] 수확  $(date +%H:%M) ==="
# scene_meta 가 없으면 받은 쪽에서 분석이 불가능하다. 생성 단계가 이미 넣지만
# 옛 데이터를 섞어 돌린 경우를 대비해 한 번 더 채운다(있으면 건너뛴다).
$PY scripts/export_scene_meta.py --root "$OUT" || true
$PY scripts/harvest.py --root "$OUT" --cache /tmp --prefix a3_,qc_,ax_ --out harvest_$(date +%m%d).tar.gz --keep-geom

echo "=== 완료 $(date +%H:%M) ==="
ls -lh harvest_*.tar.gz
echo "이 tar.gz 만 가져오면 된다. 프레임은 서버에 둔다."
