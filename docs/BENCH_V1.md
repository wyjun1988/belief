# bench-v1 — 동결 벤치마크 (2026-09-02, AUDIT 조치2)

> ⚠️ **bench-v1 은 ② 에 대해 무효** (§125: 이동 물체가 렌더에 없음). 기준선·CI·재료
> 사다리 기록용으로만 남긴다. 고친 생성기로 재생성한 **bench-v2** 가 유효 벤치다
> (같은 20채 · 같은 설정 · `data/hssd20S2` · `~/khcache/bench-v2/`).
> **bench-v2.2** (2026-09-02 오후): §128 의 결함 4종 수정 후 — 맵 재생성(`--remap`)·qc/ax 재캐시·
> 초기맵 재구축·재채점(`t1_floor0.8_d40_v22.jsonl`). v2.0/v2.1 수치는 무효.

**규칙**: 이 위에서만 수치를 낸다. 알고리즘 변경은 **한 번에 한 노브**, `scripts/bench.sh`
로 실행, 결과는 재료 사다리·기준선 3행·CI 와 함께 기록. 데이터를 바꾸면 bench-v2 다.

## 구성 (M1 Max `~/khcache/bench-v1/`)
- 데이터: `data/hssd20S` — HSSD 20채 · 1200프레임 · LLM 시나리오(hssd_move.json) · 매핑워크
  · 이동 62 (②46 ③16). gt.json 해시는 `MANIFEST.sha256` (동결 시점).
- 캐시: `cache/hsc_{a,q,x}_house_%04d.npz` (OWL 박스·이미지질의·exemplar)
- 초기맵: 각 채 `initmap_owl.json` (투영+인스턴스, build_initmap)
- 점수: `scores/t1_floor0.8_d40.jsonl` (mlx 4B, 현행 챔피언) ·
  `scores/t1_floor0.0_d160.jsonl` (§124 A율 공략판, 완료 시 추가)

## 챔피언 설정 (bench.sh 기본값)
SG_INIT=hybrid · LOC_GEO=1 · **LOC_YAW_GT=1(GT 포즈)** · **거리 GT(GEO_DEPTH 없음)** ·
LOC_TRI=1 · ABS_TH=0.100 · VERIFY_TH=2.069 · VERIFY_TH2=0.887 · C0_MIN=1 ·
PRIOR_JSON=data/hssd_move.json

재료 사다리는 eval_online 이 자동으로 찍는다. 현재 챔피언은 **GT 재료 2종(포즈·거리)**
포함이다 — "무GT" 가 아니다.

## 기준선 (자동 출력)
초기맵만 · 최신 강검출(점수 상위 10% 중 최신 프레임의 카메라 방) · 사전확률만.
시스템이 이 셋을 CI 밖에서 이기지 못하면 그 경우는 "미해결" 이다.
