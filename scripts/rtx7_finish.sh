#!/bin/bash
# 생성 완주 감시 → 캐시 4종 → harvest 자동 체인 (RTX PRO 6000)
#
# 생성(thor_gen2)이 도는 중에 걸어두면 완주를 감지해 뒷단계를 이어서 돈다:
#   nohup bash scripts/rtx7_finish.sh > finish.log 2>&1 &
#
# 완주 감지는 gt.json 개수다 — 생성기가 채 처리의 **마지막에** 쓰는 파일이라
# 존재 = 그 채 완료. 프로세스 감시(pgrep)는 자기매칭 교착 전력이 있어 쓰지 않는다.
#
# 중간 점검(완주 전, 지금 완료된 채만으로):
#   HOUSES=60 PREFIX=t7p_ bash scripts/rtx7_finish.sh
#   → 대기 없이 즉시 발화. 접두어를 본판(t7_)과 반드시 분리할 것 —
#     exp_anchowl 어휘가 채 집합에서 유도되므로 부분판·완판 캐시는 호환되지 않는다.
#
# 캐시 4종은 채 단위 스킵(있으면 건너뜀)이라 중단 후 재실행이 안전하다.
# 단 같은 접두어를 두 프로세스에 동시에 주지 말 것 (런북 함정 #5).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-data/thor7}
HOUSES=${HOUSES:-100}
PREFIX=${PREFIX:-t7_}
KX=${KX:-$HOME/kx-venv/bin/python}
CACHE_DIR=${CACHE_DIR:-/tmp}

# ── 1. 완주 대기 (5분 주기). 개수가 오래 안 늘면 생성이 죽은 것 — 로그를 볼 것.
while :; do
  n=$(ls "$ROOT"/house_*/gt.json 2>/dev/null | wc -l)
  echo "[finish] 완료 ${n}/${HOUSES}채  $(date '+%m-%d %H:%M')"
  [ "$n" -ge "$HOUSES" ] && break
  sleep 300
done

# ── 2. 완료 채만 모은 뷰 — 캐시 스크립트는 house_* 를 전부 glob 하므로
#      미완료 채(gt.json 없음)가 섞이면 죽는다. 링크 너머 실제 채 이름이
#      캐시 파일명에 쓰인다(realpath 기반).
ABS=$(cd "$ROOT" && pwd)
VIEW=${ROOT}_${PREFIX%_}view
mkdir -p "$VIEW"
for d in "$ABS"/house_*; do
  [ -f "$d/gt.json" ] && ln -sfn "$d" "$VIEW/$(basename "$d")"
done
echo "[finish] 뷰 $VIEW: $(ls "$VIEW" | wc -l)채"

# ── 3. 캐시 4종 ──
THOR_ROOT=$VIEW CACHE_PREFIX=$CACHE_DIR/${PREFIX}a_            $KX -u scripts/exp_anchowl.py 4
THOR_ROOT=$VIEW QCACHE_PREFIX=$CACHE_DIR/${PREFIX}q_ STRIDE=4  $KX -u scripts/exp_imgq.py
THOR_ROOT=$VIEW ACACHE_PREFIX=$CACHE_DIR/${PREFIX}x_ STRIDE=4  $KX -u scripts/exp_anchor_exemplar.py
THOR_ROOT=$VIEW CLIP_PREFIX=$CACHE_DIR/${PREFIX}c_  STRIDE=4   $KX -u scripts/exp_clip_rooms.py

# ── 4. harvest ──
OUT=harvest_${PREFIX%_}.tar.gz
$KX scripts/harvest.py --root "$VIEW" --cache "$CACHE_DIR" --prefix "$PREFIX" \
  --out "$OUT" --keep-geom
echo "[finish] 끝 — $OUT 을 DriveSyncFiles 로 올릴 것"
