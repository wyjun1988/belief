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
