# RTX 주말 무인 잡 (2026-09-05~07)

주말엔 피드백을 못 받으니 **한 번 띄우면 끝까지 도는 스크립트**로 묶었다. 결과는 파일 하나에 모인다.

## 0. 준비 (금요일 퇴근 전 10분)
```bash
git pull
# CUT3R 을 같이 돌리려면(권장): docs/RTX_CUT3R_SETUP.md 1절대로 설치 + 2-1 자가검사 한 번 (10분)
```
Merom·Wainscott 생성이 끝나면 4채, 아니면 되는 만큼(2채)으로 시작해도 된다 — 스크립트는 있는 채만 돈다.

## 1. 실행 (한 줄)
```bash
OG4=/mnt/ssd2/wooyeol/work/og4_walk K=1 \
CUT3R_ROOT=/path/CUT3R CUT3R_CKPT=/path/CUT3R/src/cut3r_512_dpt_4_64.pth \
nohup bash scripts/rtx_weekend.sh > ~/rtx_weekend.log 2>&1 &
```
- `K=1` 이면 3단계에서 house_0000 으로 척도 상수를 스스로 구한다(GT sim3 스케일). 이미 알면 `K=0.5xx` 로.
- CUT3R 경로를 비우면 6·7단계는 건너뛴다.
- GPU: VGGT·CUT3R 은 기본 CUDA 장치 — **GPU1 만 쓰도록 `CUDA_VISIBLE_DEVICES=1`** 을 앞에 붙여 달라. GPU0 미접촉.
- 재실행 안전: 이미 만든 raw/pose 파일은 건너뛴다. 중간에 죽으면 같은 명령을 다시 띄우면 이어진다.

## 2. 순서와 예상 시간 (4채 기준)
| 단계 | 내용 | 시간 |
|---|---|---|
| 1 | 평가 체인(생성 생략): 캐시·검증 점수·**위치 GT 3+1 표** | 1~2 h |
| 2 | 질의 프레임 목록(채당 수십 장) | 1 min |
| 3 | VGGT 척도 상수 K (house_0000, GT sim3) | 10 min |
| 4 | VGGT 4채: 지도 전부+질의 프레임 한 통과 → 진단/실제 정렬 | 30 min |
| 5 | **VGGT 위치:SfM 3+1 표** | 30 min |
| 6 | CUT3R 4채 (같은 질의 프레임) | 30 min |
| 7 | **CUT3R 위치:SfM 3+1 표** | 30 min |
전부 6시간 안팎. 토요일 아침이면 끝나 있다.

## 3. 결과 읽는 법 (`~/khcache/RTX_WEEKEND_RESULTS.md`)
- 4·6단계 `[진단] sim3 정렬(… 인라이어 X …)`: **X ≥ 0.4 이면 지도가 섰다**, <0.2 면 접힘.
- `[실제] … 카메라방 적중 Y`: **Y ≥ 0.8 이면 채택선**.
- 1·5·7의 3+1 표: 위치 GT vs VGGT vs CUT3R — 같은 채, 같은 표.
- `FAIL` 줄이 있으면 그 단계 로그 경로가 적혀 있다.

## 4. 시간이 남으면 (선택, 월요일 전에 확인 안 해도 됨)
4단계 인라이어가 3채 이상 ≥0.4 이면 **20채 생성**을 시작해 두어도 좋다(0-b·0-e 설정: 1,200~2,000 프레임, `--map-travel 0.35`).
```bash
# 20채 생성만(평가는 월요일에): 기존 20채 생성 명령 + --map-travel 0.35
```

## 5. 월요일에 보내 줄 것
`~/khcache/RTX_WEEKEND_RESULTS.md` 를 `docs/RTX_WEEKEND_20260905_RESULTS.md` 로 저장해 브랜치 `rtx-weekend` 에 커밋·푸시.
(pycolmap 4.2 호환 수정 등 서버 작업본의 diff 도 같은 브랜치에.)
