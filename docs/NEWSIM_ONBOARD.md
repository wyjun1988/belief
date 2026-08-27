# 새 시뮬레이터 에피소드 온보딩 절차

새 에피소드 zip 이 DriveSyncFiles 에 오면 이 순서대로. (1개 에피소드 기준 ~30분)

## 0. 좌표 규약 (확정 — 재검증법 포함)

카메라: `location`(cm) + `rotation_pyr_deg` + `fov_deg`, 표준 구면 규약:
`FWD=(cosP·cosY, cosP·sinY, sinP)` · `RIGHT=(−sinY, cosY, 0)` · `UP=R×F`.
frame1 체스판 투영 5px 오차로 검증(`newsim_project.py` selftest).
**exporter 가 바뀌면 selftest 부터** — 회귀·비율 검증은 표본 부족으로 실패했고
"알려진 프레임의 물체 배치와 직접 대조"가 유일하게 결정적이었다.

## 1. 변환

```bash
python scripts/newsim_adapt.py --ep <에피소드dir> --out data/newsim/epN
```
→ 우리 표준 gt.json (방·타겟·이동·live). 이동/방 개수가 scenario 와 맞는지 확인.

## 2. 캐시 (M1 Max)

```bash
THOR_ROOT=data/newsim/epN CACHE_PREFIX=~/khcache/ns_a_ python scripts/exp_anchowl.py 1
THOR_ROOT=data/newsim/epN QCACHE_PREFIX=~/khcache/ns_q_ STRIDE=1 python scripts/exp_imgq.py
THOR_ROOT=data/newsim/epN ACACHE_PREFIX=~/khcache/ns_x_ python scripts/exp_anchor_exemplar.py
```
(에피소드가 짧아 stride 1)

## 3. 검증기 현지 캘리브레이션 (§89 교훈 — 도메인마다 문턱 재설정)

```bash
python scripts/newsim_calib_pairs.py --ep <dir> --out /tmp/nscalib   # 쌍 200 자동
# tar → DriveSyncFiles → H100:
MODEL=Qwen/Qwen3.5-9B PAIRS=/tmp/nscalib OUT_JSONL=nscalib_scores.jsonl \
  python scripts/exp_vlm_verify3.py
# 로컬: 점수 스윕 → 기각 0.98+ 운용점 채택 (§85 방식)
```
⚠️ 쌍의 목격 오라클은 AABB 투영(가림 미처리) — 수용률이 보수적으로 측정된다.

## 4. 평가

```bash
THOR_ROOT=data/newsim/epN A3_PREFIX=... QC_PREFIX=... python scripts/eval_paths.py
... eval_abilities4.py / eval_handoff.py / eval_online.py (SG_INIT=hybrid)
```
방 인지: 에피소드에 매핑워크가 없으면 초반 구간을 노드로 대용(추가 예정).

## 준비 상태 (2026-08-27)

| 항목 | 상태 |
|---|---|
| 어댑터·투영(=bbox/가시성 GT 자급) | ✅ ep1 검증 |
| 캘리브레이션 쌍 생성기 | ✅ ep1 200쌍 — `nscalib_ep1.tar.gz` 드라이브 대기 |
| 스테레오 뎁스 → 초기맵 | 🔄 다음 |
| 생성측 요청 명세 | docs/NEWSIM_EPISODES_20260826.md (bbox GT 요청은 투영 자급으로 해소) |
