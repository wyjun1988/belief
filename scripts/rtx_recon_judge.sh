#!/bin/bash
# RTX PRO 6000 — OG 4채 학습식 재구성 판정 (VGGT 전체 지도 · CUT3R 스캔 지도). 평가 체인·HSSD 불필요: 집 디렉터리(map/ live/ gt.json)만 있으면 돈다.
#   OG4=/mnt/ssd2/wooyeol/work/og4 CUT3R_ROOT=/path/CUT3R CUT3R_CKPT=/path/CUT3R/src/cut3r_512_dpt_4_64.pth \
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/rtx_recon_judge.sh > ~/rtx_recon.log 2>&1 &
# 규칙: GPU0 미접촉. 단계가 실패해도 다음으로 넘어가고 결과 파일에 FAIL 을 남긴다. 재실행하면 이미 있는 산출물은 건너뛴다.
# 판정: [진단] sim3 정렬 인라이어 ≥0.4 이면 지도가 섰다(<0.2 접힘) · [실제] 카메라방 적중 ≥0.8 이면 채택선. HSSD 기준선은 docs/RTX_TASKS_20260907.md §4.
set -u
OG4=${OG4:?og4 디렉터리 (house_* 포함)}; RES=${RES:-$HOME/khcache/RTX_RECON_RESULTS.md}
VG=${VG:-$HOME/khcache/vggt}; CR=${CR:-$HOME/khcache/cut3r}; CUT3R_ROOT=${CUT3R_ROOT:-}; CUT3R_CKPT=${CUT3R_CKPT:-}
LSTEP=${LSTEP:-10}          # live 프레임 표본 간격 (질의 목록이 없으니 균일 표본: 1200장 → 120장)
RESN=${RESN:-518}           # VGGT 해상도. 96GB 면 전체 지도(300~500장) 518 이 들어간다. OOM 이면 392.
K=${K:-1}                   # 1 = house 첫 채에서 DA 척도 상수를 GT sim3 스케일로 추정, 아니면 고정값(HSSD 0.468)
mkdir -p $VG $CR $(dirname $RES); cd "$(dirname "$0")/.."
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $RES; }
HOUSES=$(ls -d $OG4/house_* 2>/dev/null); N=$(echo "$HOUSES" | wc -w | tr -d ' ')
[ "$N" -ge 1 ] || { echo "집 없음: $OG4"; exit 2; }
say "# RTX 재구성 판정 (채 $N · live 표본 1/$LSTEP · VGGT ${RESN}px) — FAIL 은 단계 실패"
for h in $HOUSES; do hn=$(basename $h); say "  $hn: map $(ls $h/map/*.jpg 2>/dev/null | wc -l | tr -d ' ')장 · live $(ls $h/live/*.jpg 2>/dev/null | wc -l | tr -d ' ')장 · gt.json $( [ -f $h/gt.json ] && echo 있음 || echo 없음 )"; done

say "## 1. VGGT 척도 상수 K"
H0=$(echo "$HOUSES" | head -1); h0=$(basename $H0)
if [ "$K" = "1" ]; then
  [ -s $VG/raw_k_$h0.jsonl ] || python scripts/vggt_reloc.py $H0 --global-live-step $LSTEP --map-max 0 --res $RESN --da-k 1 --out $VG/raw_k_$h0.jsonl > $VG/k_$h0.log 2>&1 || say "FAIL VGGT K 실행 (로그 $VG/k_$h0.log — OOM 이면 RESN=392 로 재실행)"
  python scripts/sfm_reloc.py $H0 --from-poses $VG/raw_k_$h0.jsonl --scale gt --align gt --work $VG/kdiag_$h0 --out /tmp/k.jsonl 2>&1 | grep -aE "정렬\(" | sed 's/^/  /' | tee -a $RES
  K=$(python scripts/sfm_reloc.py $H0 --from-poses $VG/raw_k_$h0.jsonl --scale gt --align gt --work $VG/kdiag_$h0 --out /tmp/k.jsonl 2>&1 | grep -aoE "스케일 [0-9.]+" | head -1 | awk '{print $2}')
  [ -z "$K" ] && K=0.468 && say "  K 추출 실패 → 0.468(HSSD 값) 가정"
fi
say "  K = $K"

say "## 2. VGGT $N채 — 지도 전부 + live 표본 한 통과 → [진단] GT sim3 · [실제] 지점 라벨 정렬"
for h in $HOUSES; do hn=$(basename $h)
  [ -s $VG/raw_$hn.jsonl ] || python scripts/vggt_reloc.py $h --global-live-step $LSTEP --map-max 0 --res $RESN --da-k $K --out $VG/raw_$hn.jsonl > $VG/$hn.log 2>&1 || say "FAIL VGGT $hn (로그 $VG/$hn.log)"
  echo "### VGGT $hn" >> $RES; grep -aE "전역 통과|→" $VG/$hn.log 2>/dev/null | tail -2 | sed 's/^/  /' >> $RES
  python scripts/sfm_reloc.py $h --from-poses $VG/raw_$hn.jsonl --scale gt --align gt --work $VG/${hn}_gt --out /tmp/x.jsonl 2>&1 | grep -aE "정렬\(|커버리지" | sed 's/^/  [진단] /' >> $RES
  python scripts/sfm_reloc.py $h --from-poses $VG/raw_$hn.jsonl --scale da --align sites --work $VG/$hn --out $VG/pose_$hn.jsonl 2>&1 | grep -aE "라벨 정렬|커버리지" | sed 's/^/  [실제] /' >> $RES
done

if [ -n "$CUT3R_ROOT" ] && [ -n "$CUT3R_CKPT" ]; then
  say "## 3. CUT3R $N채 — 지도 전부 → live 표본 순서로 한 세션 (자가검사 포함)"
  for h in $HOUSES; do hn=$(basename $h)
    [ -s $CR/self_$hn.log ] || python scripts/cut3r_reloc.py $h --cut3r-root $CUT3R_ROOT --model-path $CUT3R_CKPT --selfcheck > $CR/self_$hn.log 2>&1 || say "FAIL CUT3R 자가검사 $hn (로그 $CR/self_$hn.log)"
    [ -s $CR/raw_$hn.jsonl ] || python scripts/cut3r_reloc.py $h --cut3r-root $CUT3R_ROOT --model-path $CUT3R_CKPT --live-step $LSTEP --da-k $K --out $CR/raw_$hn.jsonl > $CR/$hn.log 2>&1 || say "FAIL CUT3R $hn (로그 $CR/$hn.log — OOM 이면 --size 224 또는 --max-frames 800)"
    echo "### CUT3R $hn" >> $RES; grep -aE "자가검사|이동 중앙" $CR/self_$hn.log 2>/dev/null | tail -1 | sed 's/^/  /' >> $RES; grep -aE "세션 완료|DA 척도|→" $CR/$hn.log 2>/dev/null | tail -3 | sed 's/^/  /' >> $RES
    python scripts/sfm_reloc.py $h --from-poses $CR/raw_$hn.jsonl --scale gt --align gt --work $CR/${hn}_gt --out /tmp/x.jsonl 2>&1 | grep -aE "정렬\(|커버리지" | sed 's/^/  [진단] /' >> $RES
    python scripts/sfm_reloc.py $h --from-poses $CR/raw_$hn.jsonl --scale da --align sites --work $CR/$hn --out $CR/pose_$hn.jsonl 2>&1 | grep -aE "라벨 정렬|커버리지" | sed 's/^/  [실제] /' >> $RES
  done
else
  say "## 3. CUT3R 건너뜀 (CUT3R_ROOT/CUT3R_CKPT 미지정)"
fi
say "## 끝 — 이 파일을 docs/RTX_RECON_20260907_RESULTS.md 로 복사해 브랜치 rtx-0907 에 커밋·푸시"
echo RTX_RECON_DONE | tee -a $RES
