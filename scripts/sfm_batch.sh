#!/bin/bash
# SfM 재국소화 배치 — 채 단위 병렬. 4채 먼저(HOUSES=4), 가능성 보이면 20채.
#   ROOT=data/hssd20S2 HOUSES=4 PAR=3 THREADS=3 EXTRA="--fast" bash scripts/sfm_batch.sh
#   RTX(CUDA): PAR=2 THREADS=8 EXTRA="--gpu --fast" bash scripts/sfm_batch.sh
# 출력: $OUT (합친 POSE_JSONL) · ~/khcache/sfm/<house>/summary_<house>.json · 채별 표
ROOT=${ROOT:-data/hssd20S2}; HOUSES=${HOUSES:-4}; PAR=${PAR:-3}; THREADS=${THREADS:-3}; EXTRA=${EXTRA:-}
PY=${PY:-$HOME/kx-venv/bin/python}; [ -x "$PY" ] || PY=python
SFM=${SFM:-$HOME/khcache/sfm}; OUT=${OUT:-$SFM/pose_$(basename $ROOT)_${HOUSES}.jsonl}
mkdir -p $SFM/logs; export KMP_DUPLICATE_LIB_OK=TRUE PY SFM THREADS EXTRA
run_one() { h=$1; hn=$(basename $h)
  # ⚠️ --work 를 반드시 $SFM 아래로: 없으면 sfm_reloc 기본값 ~/khcache/sfm/<house> 를 **데이터셋이 달라도 공유**해
  # .extracted 마커·rec_* 캐시를 그대로 읽는다. 2026-09-05 sfm-nogt40(19채)·sfm-nogt20(4채)이 9/3 hssd20S2 재구성을
  # 새 GT 에 채점한 것으로 확인됨 — §165. summary 도 $SFM/<house>/ 에서 읽으므로 경로가 이제 일치한다.
  $PY -u scripts/sfm_reloc.py $h --work $SFM/$hn --threads $THREADS $EXTRA --out $SFM/pose_$hn.jsonl > $SFM/logs/$hn.log 2>&1
  tr -d "\000" < $SFM/logs/$hn.log | tr "\r" "\n" | grep -a -E "^\[.*(정렬|커버리지)|실패|Traceback|Error" | tail -3 | sed "s/^/$hn /"; }
export -f run_one
ls -d $ROOT/house_* | head -$HOUSES | xargs -P $PAR -I{} bash -c 'run_one {}'
: > $OUT; for h in $(ls -d $ROOT/house_* | head -$HOUSES); do cat $SFM/pose_$(basename $h).jsonl >> $OUT 2>/dev/null; done
$PY - <<PY
import json, glob, os
S = []
for h in sorted(glob.glob("$ROOT/house_*"))[:$HOUSES]:
    f = os.path.join("$SFM", os.path.basename(h), "summary_%s.json" % os.path.basename(h))
    if os.path.exists(f): S.append(json.load(open(f)))
print("| 채 | map 등록 | live 커버리지 | ATE 중앙 | <0.5m | yaw 중앙 | 카메라방 | sim3 인라이어 | 초 |")
print("|---|---|---|---|---|---|---|---|---|")
for s in S: print("| %s | %d/%d | %.2f | %.2fm | %.2f | %.1f° | %s | %.2f | %d |" % (s["house"], s["map_reg"], s["n_map"], s["cov"], s["ate_med"] or -1, s["ate_lt05"] or 0, s["yaw_med"] or -1, ("%.2f" % s["room_hit"]) if s.get("room_hit") is not None else "—", s.get("sim3_inl", 0), s["sec"]))
import statistics as st
if S: print("합계 %d채 · 커버 %.2f · <0.5m %.2f · 카메라방 %.2f · 채당 %d초" % (len(S), st.mean(s["cov"] for s in S), st.mean(s["ate_lt05"] or 0 for s in S), st.mean(s["room_hit"] or 0 for s in S), st.mean(s["sec"] for s in S)))
PY
echo "→ $OUT"; echo SFM_BATCH_DONE
