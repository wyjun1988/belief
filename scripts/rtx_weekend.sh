#!/bin/bash
# RTX PRO 6000 주말 무인 잡 (2026-09-05~07). 한 번 띄우면 순서대로 돌고 결과를 $RES 한 파일에 모은다.
#   OG4=/mnt/ssd2/.../og4 K=1 CUT3R_ROOT=/path/CUT3R CUT3R_CKPT=/path/cut3r_512_dpt_4_64.pth nohup bash scripts/rtx_weekend.sh > ~/rtx_weekend.log 2>&1 &
# 규칙: GPU0 미접촉(BEHAVIOR). 각 단계는 실패해도 다음으로 넘어가고 결과 파일에 FAIL 을 남긴다. 재실행하면 이미 있는 산출물은 건너뛴다.
set -u
OG4=${OG4:?og4 디렉터리}; K=${K:-1}; RES=${RES:-$HOME/khcache/RTX_WEEKEND_RESULTS.md}; B=${B:-$HOME/khcache/bench-og4}
CUT3R_ROOT=${CUT3R_ROOT:-}; CUT3R_CKPT=${CUT3R_CKPT:-}; VG=${VG:-$HOME/khcache/vggt}; CR=${CR:-$HOME/khcache/cut3r}
mkdir -p $VG $CR $(dirname $RES); cd "$(dirname "$0")/.."
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $RES; }
HOUSES=$(ls -d $OG4/house_* 2>/dev/null); N=$(echo "$HOUSES" | wc -w | tr -d ' ')
say "# RTX 주말 결과 (채 $N) — 각 단계 실패는 FAIL 로 표기"

say "## 1. 평가 체인(생성 생략) — 위치 GT 기준 3+1 (대조)"
if [ ! -f $B/scores/t1_floor0.8_d40.jsonl ]; then
  SKIP_GEN=1 SCORER=hf CALIB=1 OUT=$OG4 BENCH_DIR=$B bash scripts/bench_v2_chain.sh > $B.chain.log 2>&1 || say "FAIL 평가 체인 (로그 $B.chain.log)"
fi
grep -aE "재료 사다리|최종 답|①이동없음  |②재촬영 |③|④" $B.chain.log 2>/dev/null | grep -av "\[" | head -12 >> $RES

say "## 2. 질의 프레임"
THOR_ROOT=$OG4 A3_PREFIX=$B/cache/hs2_a_ VERIFY_JSONL=$B/scores/t1_floor0.8_d40.jsonl python scripts/query_frames.py --out $VG/q.json 2>&1 | tail -3 >> $RES || say "FAIL 질의 프레임"

say "## 3. VGGT — 척도 상수 K (house_0000, --da-k 1 로 GT sim3 스케일 읽기)"
H0=$(echo "$HOUSES" | head -1)
if [ "$K" = "1" ]; then
  python scripts/vggt_reloc.py $H0 --query-frames $VG/q.json --global-live-step 1 --map-max 0 --res 518 --da-k 1 --out $VG/raw_k.jsonl > $VG/k.log 2>&1 || say "FAIL VGGT K 실행 (로그 $VG/k.log)"
  python scripts/sfm_reloc.py $H0 --from-poses $VG/raw_k.jsonl --scale gt --align gt --work $VG/kdiag --out /tmp/k.jsonl 2>&1 | grep -aE "정렬\(" | tee -a $RES
  K=$(python scripts/sfm_reloc.py $H0 --from-poses $VG/raw_k.jsonl --scale gt --align gt --work $VG/kdiag --out /tmp/k.jsonl 2>&1 | grep -aoE "스케일 [0-9.]+" | head -1 | awk '{print $2}')
  [ -z "$K" ] && K=0.468 && say "K 추출 실패 → 0.468 가정"
fi
say "K = $K"

say "## 4. VGGT 4채 — 질의 프레임 전역 통과 → 진단(GT sim3) + 실제(라벨 정렬)"
for h in $HOUSES; do hn=$(basename $h)
  [ -s $VG/raw_$hn.jsonl ] || python scripts/vggt_reloc.py $h --query-frames $VG/q.json --global-live-step 1 --map-max 0 --res 518 --da-k $K --out $VG/raw_$hn.jsonl > $VG/$hn.log 2>&1 || say "FAIL VGGT $hn"
  echo "### $hn" >> $RES; grep -aE "전역 통과|→" $VG/$hn.log | tail -2 >> $RES
  python scripts/sfm_reloc.py $h --from-poses $VG/raw_$hn.jsonl --scale gt --align gt --work $VG/${hn}_gt --out /tmp/x.jsonl 2>&1 | grep -aE "정렬\(|커버리지" | sed 's/^/  [진단] /' >> $RES
  python scripts/sfm_reloc.py $h --from-poses $VG/raw_$hn.jsonl --scale da --align sites --work $VG/$hn --out $VG/pose_$hn.jsonl 2>&1 | grep -aE "라벨 정렬|커버리지" | sed 's/^/  [실제] /' >> $RES
done
cat $VG/pose_house_*.jsonl > $VG/pose_og.jsonl 2>/dev/null
say "## 5. VGGT 위치:SfM 3+1"
SKIP_GEN=1 SCORER=hf CALIB=1 POSE_JSONL=$VG/pose_og.jsonl OUT=$OG4 BENCH_DIR=$B-vggt bash scripts/bench_v2_chain.sh > $B-vggt.log 2>&1 || say "FAIL VGGT 벤치"
grep -aE "재료 사다리|최종 답|①이동없음  |②재촬영 |③|④" $B-vggt.log | grep -av "\[" | head -12 >> $RES

if [ -n "$CUT3R_ROOT" ] && [ -n "$CUT3R_CKPT" ]; then
  say "## 6. CUT3R 4채 (같은 질의 프레임)"
  for h in $HOUSES; do hn=$(basename $h)
    [ -s $CR/raw_$hn.jsonl ] || python scripts/cut3r_reloc.py $h --cut3r-root $CUT3R_ROOT --model-path $CUT3R_CKPT --query-frames $VG/q.json --da-k $K --out $CR/raw_$hn.jsonl > $CR/$hn.log 2>&1 || say "FAIL CUT3R $hn (로그 $CR/$hn.log)"
    echo "### $hn" >> $RES; grep -aE "세션 완료|DA 척도|→|자가검사" $CR/$hn.log | tail -3 >> $RES
    python scripts/sfm_reloc.py $h --from-poses $CR/raw_$hn.jsonl --scale gt --align gt --work $CR/${hn}_gt --out /tmp/x.jsonl 2>&1 | grep -aE "정렬\(|커버리지" | sed 's/^/  [진단] /' >> $RES
    python scripts/sfm_reloc.py $h --from-poses $CR/raw_$hn.jsonl --scale da --align sites --work $CR/$hn --out $CR/pose_$hn.jsonl 2>&1 | grep -aE "라벨 정렬|커버리지" | sed 's/^/  [실제] /' >> $RES
  done
  cat $CR/pose_house_*.jsonl > $CR/pose_og.jsonl 2>/dev/null
  say "## 7. CUT3R 위치:SfM 3+1"
  SKIP_GEN=1 SCORER=hf CALIB=1 POSE_JSONL=$CR/pose_og.jsonl OUT=$OG4 BENCH_DIR=$B-cut3r bash scripts/bench_v2_chain.sh > $B-cut3r.log 2>&1 || say "FAIL CUT3R 벤치"
  grep -aE "재료 사다리|최종 답|①이동없음  |②재촬영 |③|④" $B-cut3r.log | grep -av "\[" | head -12 >> $RES
else
  say "## 6-7. CUT3R 건너뜀 (CUT3R_ROOT/CUT3R_CKPT 미지정)"
fi
say "## 끝 — 이 파일을 docs/RTX_WEEKEND_20260905.md 로 복사해 커밋/푸시 (브랜치 rtx-weekend)"
echo RTX_WEEKEND_DONE | tee -a $RES
