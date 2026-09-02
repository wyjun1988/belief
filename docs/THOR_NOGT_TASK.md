# THOR 무GT 재측정 (RTX 노드 작업)

§122 에서 확인: 지금까지 THOR 수치는 대부분 **SG_INIT=gt**(초기 씬그래프가 정답)
위에 있었다. 무GT 수치(0.532)도 `build_initmap` 이 repo 에 없던 시절이라
**어떤 초기맵이었는지 재현 불가**다. 이제 스크립트가 있으므로 정직하게 다시 잰다.

## 실행 (thor7 뷰, 프레임 필요 — RTX 노드에만 있음)

```bash
git pull
V="THOR_ROOT=<thor7 뷰> A3_PREFIX=/tmp/t7_a_ QC_PREFIX=/tmp/t7_q_ AX_PREFIX=/tmp/t7_x_"

# 1) 초기 씬그래프를 검출로 구축 (매핑워크 프레임 사용, GT 미사용)
env $V INITMAP_GEO=1 INITMAP_INST=1 ~/kx-venv/bin/python -u scripts/build_initmap.py

# 2) 무GT 평가 — GT 판과 나란히
for SG in gt hybrid; do echo "===== SG_INIT=$SG ====="
  env $V SG_INIT=$SG FRAME_W=768 LOC_GEO=1 LOC_TRI=1 ABS_TH=0.100 \
    VERIFY_JSONL=t1_scores_t7ac.jsonl VERIFY_TH=2.262 VERIFY_TH2=1.890 C0_MIN=1 \
    GEO_DEPTH=geo_depth_t7c.jsonl \
    ~/kx-venv/bin/python scripts/eval_online.py | grep -E "최종 답|top-1\+2|①|②|③"
done
```

## 보고 형식
`build_initmap` 의 **방배정 정확도**(채별·평균)와, SG_INIT=gt vs hybrid 의
전체·①·②·③ 를 나란히. HSSD 대조군: 방배정 0.779 · 무GT 전체 0.497.

## 왜 중요한가
"우리 시스템이 GT 없이 얼마나 하는가" 가 실제 성능이다. THOR 는 4시간 에피소드라
② 표본이 충분하므로, **무GT 조건에서 ② 가 얼마나 사는지**를 볼 수 있는 유일한 판이다.

## 2026-09-02 추가 — RTX 가 빌 때 같이 잴 것 (프로토콜 v2)
- 위 명령에 `PRIOR_JSON=data/thor_prior.json` 을 명시. 출력에 **재료 사다리·기준선 3행·CI·
  ② 증거 조건부 블록**이 자동으로 붙는다 (docs/EVAL_PROTOCOL_V2.md). 스칼라 ② 대신 그 표를 보고.
- **앵커 exemplar 384 버그 점검**: thor7 t7view 가 768 프레임이면 `exp_anchor_exemplar` 의 옛
  `/384` 인덱싱이 거기서도 틀렸을 수 있다(§128-4). 고친 스크립트로 `ax` 캐시를 다시 만들고
  `LOC_YAW_GT=0`(투표 포즈, 앵커 게이트 ANCH_EX/TY/DP 기본값) 판을 GT 포즈 판과 나란히.
- 옛 THOR 수치(§111~§118)는 이 두 가지 때문에 재현 대상이지 기준선이 아니다.
