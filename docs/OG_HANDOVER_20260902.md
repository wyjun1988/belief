# OmniGibson 생성기 인수인계 (2026-09-02, rc-rtxpro)

이 문서만 읽고 이어받을 수 있게 쓴다. 원인 분석의 상세 근거는
[OG_DIAG_20260902.md](OG_DIAG_20260902.md), 요구 조건은
[OMNIGIBSON_EPISODE_SPEC.md](OMNIGIBSON_EPISODE_SPEC.md)·
[PORTING_CHECKLIST.md](PORTING_CHECKLIST.md).

## 0. 한 줄 상태

**생성기는 동작하고 게이트를 통과한 판이 있다**(v6 1200프레임 · v8 2000프레임).
지금은 SPEC 4-2(2026-09-02 추가) 8개 항목을 다 물린 판을 반복 실행하며
경우② 표본 수와 궤적 연속성을 마지막으로 조이는 중이다(v10 실행 중, §7).

**최종 데이터는 아직 만들지 않았다.** 20채 생성은 v10 계열이 게이트를 두 번 연속
통과한 뒤에 시작할 것.

---

## 1. 실행 방법

```bash
WORK=/mnt/ssd2/wooyeol/work
PY=$WORK/behavior1k_latest/conda_envs/behavior51/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_GPU_ID=1              # GPU0 은 타인 작업 — 절대 건드리지 않는다
export OMNIGIBSON_HEADLESS=1 PYTHONNOUSERSITE=1
export OMNIGIBSON_DATA_PATH=$WORK/behavior1k_latest/og_data_2026
export OMNIGIBSON_APPDATA_PATH=$WORK/behavior1k_latest/appdata_og51

$PY -u scripts/og_episode.py --scene Rs_int --out $WORK/og20/house_0000 \
    --house 0 --frames 2000 --props 24 --case1 10 --case2 5 --case3 5 --case4 0
```

- `CUDA_VISIBLE_DEVICES` 는 **설정하지 않는다.** 설정하면 ordinal 이 0 으로 remap 되어
  OmniGibson 의 `activeGpu=1`/`cudaDevice=1` 과 충돌해 invalid device ordinal 이 난다.
  로그에서 `--/renderer/activeGpu=1 --/physics/cudaDevice=1` 을 확인할 것.
- 환경변수 이름은 `OMNIGIBSON_DATA_PATH` 다(`..._DATASET_PATH` 아님). 틀리면 robot
  assets version None 으로 즉사한다.
- **종료코드로 성공을 판정하지 마라.** 정상 완주 후에도 종료 시점에 segfault(rc=139)가
  난다(`--/app/fastShutdown=True`). 판정은 표준출력 `EPISODE_OK` 마커로 한다.

검수 영상:
```bash
$PY scripts/og_make_video.py $WORK/og20/house_0000 out.mp4 --fps 10 --map-frames
```
GT 박스·거리·방·yaw·이동 배너·상단뷰 경로를 얹어 준다. 박스가 실물에 안 붙으면
좌표 규약이 깨진 것이므로 눈으로 즉시 걸린다.

---

## 2. 원인 3종 (전부 실측으로 확정, 상세는 OG_DIAG)

뿌리는 하나였다. **카메라가 애초에 수평이 아니었다.** USD 카메라는 local −Z 를 보고
local +Y 가 up 이라, z-up 스테이지에서 `orientation=[0,0,sin(yaw/2),cos(yaw/2)]` 는
world-Z 회전만 곱하므로 영원히 바닥을 본다.

1. **segfault** = 세그멘테이션 annotator 그래프. 150 pose 격리:
   rgb 150/150 · rgb+depth 150/150 · +seg_semantic **6** · +seg_instance **7** ·
   셋 다 **4** · 셋 다+아래보기 **150/150** · 셋 다+get_obs 미호출 **6**.
   → `seg_semantic` 하나로 죽고, `get_obs()` 없이도 죽는다 = 매 render tick 의
   OmniGraph 문제. **세그멘테이션을 쓰지 않는다**(가시성은 레이캐스트).
2. **좌표 규약** = `BASE=(0.5,−0.5,−0.5,0.5)` 로 고침. yaw 오차 50° → **0.01°**.
   `screen_x_sign = −1`(forward=+X, up=+Z → right=−Y 라 기하적 필연).
3. **navmesh** = 배포 `floor_trav_*.png` 는 방마다 섬(Rs_int 12 컴포넌트, 최대 2 방).
   라이브 씬 레이캐스트 그리드로 **5/5 방 연결**. 문을 무조건 열면 **더 나빠진다**
   (문짝이 통로로 스윙: Rs_int 5/5→4/5, Beechwood 5/11→3/11).

---

## 3. 생성기 구조 (`scripts/og_episode.py`)

```
0  env → open_doors_for_clearance(문마다 shut/lower/upper 중 통로가 열리는 쪽)
1  NavGrid(라이브 레이캐스트) → 방 4~8개·주방·화장실 확인, 초과 씬은 restrict()
2  카메라: viewer_camera + depth_linear 만.  ★ 세그멘테이션 modality 추가 금지 ★
3  물체 인벤토리(AABB 중심 = GT pos), 방은 좌표로 판정
4  소품 주입 --props: 씬에 없는 카테고리를 받침면 위에 OnTop 배치
5  매핑워크: 방마다 --map-sites 지점 × 6방향(0/60/…/300°) → map/%04d.jpg
   ★ 반드시 이동 전 ★ (SPEC 4-2.1)
6  대본 편성: case3(수납 우선) · case2(다른 방 받침면) · case1(정적 앵커)
7  Phase A  방별 초기 관측 + 대본 이동 실행
   Phase B  ★ 이동이 실제 적용된 뒤 읽은 좌표로 ★ 재방문 구간 생성
            case2 = --evidence K:D (far D+1.5 → near D 로 진입 = 시선 방향)
   Phase C  남은 예산을 방 간 보행으로 (숨긴 물체에 시선 닿으면 30회 재추첨)
8  감사 → 게이트 → EPISODE_OK
```

Phase B 를 **사후 생성**하는 것이 경우②의 핵심이다. 빌드 시점의 계획된 받침면으로
접근점을 계산하면 대체 타겟·받침면 위 실제 위치와 어긋나 최단 1.66~2.37m 가 되어
"2m 이내" 문턱을 못 넘는다(v4·v5 에서 ② 3~4/8).

---

## 4. 게이트 (`--smoke` 없이 돌리면 강제)

| 게이트 | 의미 |
|---|---|
| `frames ≥ 1200` | SPEC 2 |
| `4 ≤ 방 수 ≤ 8` + 주방·화장실 | 범위 결정 |
| `case1 ≥ 10` · `case2 ≥ 5` · `case3 ≥ 5` | SPEC 1 |
| `yaw.median_abs_error ≤ 0.5°` | SPEC 5 (실측 0.01°) |
| `route_failures == 0` | 텔레포트·직선보간 fallback 이 코드에 **없다** |
| `pos_jumps_over_0p5m == 0` | 연속 보행 |
| `yaw_step > max_turn_deg` 프레임 0 | 단계 이음새 포함 |
| `moving_frame_frac ≥ 0.55` | 제자리 회전이 주가 되지 않게 |

감사에는 이 외에 `field_convention`(SPEC 4-2.4 필드별 폴리곤 포함률),
`polygon_selfcheck`, `depth_vs_projection`, `movable_funnel`, `props`,
`evidence.closest_m`, `event_status`(실패한 대체 타겟까지)가 들어간다.

---

## 5. 판본별 실측 (이게 인수인계의 핵심)

| 판 | 프레임 | ① | ② | ③ | 보행% | yaw° | 점프 | 결과 / 걸린 것 |
|---|---|---|---|---|---|---|---|---|
| v3 | 1200 | 29 | 0 | 0 | 43 | 0.0009 | 0 | 이동 0건 — 프로그램이 잘려 case 가 사라짐(조용한 손실) |
| v4 | 1200 | 22 | 4 | 7 | 51 | 0.0057 | 0 | ② 4<5, 보행 51%<55% |
| v5 | 1200 | 23 | 3 | 6 | 65 | 0.0057 | 0 | ② 3<5 — 접근점을 이동 **전** 좌표로 계산 |
| **v6** | 1200 | 23 | **5** | **5** | 64 | 0.0084 | 0 | **EPISODE_OK** (3단계 도입) |
| v7 | 1200 | — | — | — | — | — | — | 예산 가드 정상 작동: evidence 로 1862 프레임 필요 |
| **v8** | 2000 | 23 | **6** | **7** | 67 | 0.0102 | 0 | **EPISODE_OK** — 단 `field_convention` 이 폴리곤 버그를 노출 |
| v9 | 2000 | 25 | 3 | 5 | 67 | 0.0115 | 1 | 폴리곤 고침 확인(0.013→0.937). ② 3<5, 점프 1건 |
| v10 | 2000 | 실행 중 | | | | | | §7 |

읽는 법: **v6·v8 은 통과한 판이다.** v9 가 후퇴한 것은 새 감사 항목이 이전 판의
숨은 결함(방 폴리곤)을 드러냈기 때문이고, 그걸 고치면서 노출된 잔여 3건을 v10 에서
잡는 중이다. ② 는 v6 5 · v8 6 · v9 3 으로 **판마다 흔들린다** — 그 원인이 §7 의 1 번이다.

---

## 6. 반복해서 밟은 함정 (같은 것을 또 밟지 말 것)

1. **`og.sim.render()`·`step_physics()` 는 `og.sim._objects_to_initialize` 를 비우지
   않는다.** `og.sim.step()` 만 초기화한다. 미초기화 객체는 `states[...]` 접근이 전부
   실패하고(소품 배치 0/24), 제거도 실패해 레지스트리에 남아 이동 롤백용
   `og.sim.dump_state()` 가 `Object must be initialized before dumping state!` 로 죽는다.
2. **`DatasetObject` 는 원점에 스폰된다.** 24개가 서로 관통해 첫 물리 스텝에 폭발하고
   NaN quaternion 이 된다 → add 직후(물리 전) 방별 보행 가능 셀에 분산 스테이징 +
   NaN 스크리닝. `XFormPrim.set_position_orientation` 은 USD 에 직접 쓰므로 초기화
   전에도 동작한다.
3. **`OnTop`/`Inside.set_value` 는 `use_trav_map=False` 로 불러라.** 기본값 True 는
   깨진 배포 trav 맵을 쓰므로 정상 배치를 거절한다.
4. **받침면을 카테고리로 고르지 마라.** 벽 스위치를 받침면으로 골라 회전의자를 그 위에
   올리려 한 적이 있다. 발자국 면적 ≥0.09㎡ · 대각 ≥0.4m · 상면 하향 레이캐스트
   `normal_z > 0.7` · 블랙리스트(switch·mirror·picture…)를 모두 걸어야 한다.
5. **수납 대상은 크기를 검사하라.** 컨테이너 AABB 가 물체 extent 를 3축 모두
   0.06m 이상 넘을 때만. standing_tv 를 캐비닛에 넣으려다 샘플링 실패했다.
6. **계획마다 대체 타겟을 들고 있어라.** 단일 타겟이면 3/4 가 샘플링 실패한다.
7. **증인 렌더 카메라는 물체를 조준해야 한다.** 눈높이 1.5m·pitch 0 으로는 하부장
   안(높이 0.07~0.49m) 물체가 수직 FOV(±19°) 밖이라 1/5 만 성공했다. 증인 렌더는
   진단 이미지이므로 pitch 를 쓰는 것이 맞고, **live 프레임은 yaw 폐쇄를 위해
   pitch 0 을 유지**해야 한다.
8. **방 폴리곤은 보행 가능 셀이 아니라 방 인스턴스 래스터에서 뽑아라.** 보행 셀은
   몸반경 erode + 가구 footprint 제외라 물체가 전부 밖으로 나간다
   (`gt0_pos` 포함률 0.013).
9. **프로그램을 `--frames` 로 뒤에서 자르지 마라.** 대본 이동이 통째로 사라진다
   (v3: planned 3, moves 0). 초과하면 실패로 처리해야 한다.

---

## 7. 지금 테스트 중인 것 (v10)

v9 에서 남은 3건을 측정으로 원인 규명해 고친 판이다. **이 세 가지가 잡혔는지 확인하는
것이 다음 담당자의 첫 일이다.**

| v9 결함 | 측정된 원인 | v10 수정 |
|---|---|---|
| ② 3/7 | `evidence_goals` 허용 창이 `0.6D~1.4D` = 0.9~2.1m 인데 판정 기준은 **≤2.0m**. 실측 최단 2.03·2.08·1.93m 로 자기 창 안에서 기준 초과 | 창을 `[0.6D, min(1.4D, 2.0−0.25)]` = 0.84~1.75m 로 클램프, 가까운 후보부터 정렬, D 기본 1.5→1.4 |
| 점프 1건 (0.509m) | `t=0→1` 한 곳. A* 호길이 리샘플이 `total/round(total/step)` 이라 `--step` 을 넘을 수 있다 | 리샘플 후 인접 간격이 `step*1.05` 를 넘으면 세분화 |
| 수납 증인 1/5 | 물체가 하부장 안 높이 0.07·0.12·0.49m, 상부장 2.05m. 증인 카메라가 pitch 0 이라 44° 아래 = FOV 밖 | 증인 카메라를 `pitch = atan2(d_z, |d_xy|)` 로 조준, 반경 후보 (2.0,1.6,1.3,2.5), 화면 안 여부까지 검사 |

`--spare` 도 3 으로 되돌려(계획 = quota+3) 이동 실패 여유를 늘렸다.

**확인 명령**
```bash
grep -E 'EPISODE_OK|GATE FAILED' $WORK/og_diag_20260902/gen_full10.log
$PY - <<'EOF'
import json
d=json.load(open('$WORK/og20_v10_Rs_house0/audit.json'))
print({k:d[k] for k in ('case1_no_move_revisited','case2_move_reobserved','case3_absent_belief')})
print(d['continuity']); print(d['evidence']); print(d['field_convention'])
EOF
```

**합격선**: `EPISODE_OK` + `case2 ≥ 5` + `pos_jumps_over_0p5m == 0` +
`evidence.closest_m` 전부 ≤1.8 + 수납 이동 `witness_file` 이 대부분 채워짐
(`gen_selfcheck` 의 `geo_ok` 가 True 가 되는 조건).

미달이면 `audit.evidence.closest_m` 와 `event_status` 를 먼저 보라 — 어느 물체가
어느 거리에서 실패했는지 그 두 필드에 다 있다.

---

## 8. 미결 — 결정이 필요한 것

### (a) 경우④(집 밖 반출)

`garage`·`garden` 등 실외는 **범위 밖으로 확정**됐다(2026-09-02). 그 귀결로 ④ 는
in-scope 목적지가 없어 **생성이 불가능**하다. 현재 `--case4 0`,
`audit.case4_in_scope=false` 로 정직하게 기록만 하고 있다.
SPEC 1 의 "④ ≥2건" 은 이 범위에서 충족 불가다. 선택지:

1. **범위 밖 방으로 반출** — 에피소드 범위(4~8방)에 들지 않은 방으로 옮긴다.
   "관측된 집에서 사라졌다" 가 성립하고 실외를 안 쓴다. 그런 방이 있는 씬에서만 가능.
2. **④ 폐기** — ①10 · ②5 · ③5 = 20건/채로 간다. 현재 상태.

### (b) 20채 구성

실외 방 보유 씬을 뺀 뒤 주방·화장실 + 방 4~8 조건을 만족하는 씬:

| 씬 | 방(맵) | 라이브 연결 | 비고 |
|---|---|---|---|
| `Rs_int` | 5 | **5/5** | 1순위. 전 실험의 기준 씬 |
| `Pomaria_1_int` | 7 | 7/7 | 2순위 |
| `Beechwood_0_int` | 11 | 3~10 | `--max-rooms 8` 부분집합 |
| `Wainscott_0_int` | 12 | 5/12 | 부분집합, 연결 재확인 필요 |
| `Merom_1_int` | 12 | 4/12 | 연결 나쁨, 후순위 |

20채는 **씬 3~5개 × 서로 다른 `--seed`** 로 만드는 편이 낫다. 소품 주입이 seed 마다
다른 카테고리·다른 받침면을 고르므로 채마다 타겟 구성이 달라진다. 연결이 나쁜 씬을
억지로 쓰면 방 3~4개 에피소드가 되어 ③ 이 다시 희귀사건이 된다.

`scripts/og_prescreen.py`(Isaac 불필요)와 `scripts/og_navgrid.py`(인자 없이 실행하면
전 씬 라이브 연결성 측정)로 목록을 갱신할 수 있다. **`og_navgrid.py` 전 씬 스윕은
아직 돌리지 않았다** — 20채 확정 전에 돌려 볼 것.

### (c) 소품 주입의 위상

BEHAVIOR 씬 내용만으로는 케이스 예산이 **원리적으로 불가능**하다(주방+화장실+4~8방
씬 14개 중 최선이 유니크·이동가능 물체 9개, 필요량 10+). 그래서 소품을 주입한다.
이것을 벤치 설계로 문서에 명시할지, 아니면 "씬 원본만 사용" 을 유지하고 케이스 목표를
낮출지는 과제 정의의 문제다. 현재는 주입을 쓰고 `gt0[].injected` 로 표시해
사후 필터가 가능하게 해 두었다.

---

## 9. 폐기 대상 (조용히 재사용되면 안 되는 것)

- `og20_smoke*` 300프레임 판 전부 — 카메라가 바닥을 본 top-down
- `og20_v2~v5`, `v7`, `v9` — 게이트 미통과. `v6`·`v8` 은 통과했지만 방 폴리곤 버그가
  있어 최종 데이터로는 쓰지 말 것
- `og51_probe_20260902` 200프레임 입장시험 프레임 — 같은 쿼터니언 버그로 바닥 크롭에
  AUC 를 잰 셈. `og51_rtx7_sweep_20260902.txt` 는 빈 파일 = 판정 미완. **OmniGibson
  입장 시험은 아직 성립하지 않았다**
- `--smoke` 의 direct path fallback — 새 생성기에는 그 경로가 존재하지 않는다
- 하드코딩 `FX = W*17.0/20.995` — 센서에서 읽는다

---

## 10. 다음 단계 (순서대로)

1. v10 합격 확인(§7). 미달이면 `evidence.closest_m`·`event_status` 로 원인 특정.
2. **같은 설정으로 seed 를 바꿔 한 번 더** 통과시켜 재현성 확인(② 가 판마다 흔들렸으므로).
3. `scripts/og_navgrid.py` 전 씬 스윕 → 20채 씬·seed 목록 확정(§8b).
4. `scripts/gen_selfcheck.py` 로 이동 물체 검출 self-test(SPEC 4-2.7).
   수납 이동은 문 열린 상태 증인 크롭을 채점한다.
5. 20채 생성 → 검수 5줄 보고 → `eval_online` 체인.
6. 입장 시험(§9)을 정상 수평 프레임으로 다시 재고 `SIM_SCREENING` 대장을 갱신.


---

## 10. 아이맥 검토 결과 (2026-09-02 저녁, 패치 적용 후)

패치는 `git apply` 로 깨끗하게 들어갔고(신규 9파일, 기존 파일 수정 없음) 전부 컴파일된다.
구조는 SPEC 4-2 를 잘 따랐다: 매핑워크 이동 전, AABB 중심 GT, 레이캐스트 가시성,
받침·증인 게이트, 증거 대본, 폴리곤 자가검사, 필드별 규약 감사.

**고친 것 1 (평가기 호환 — 이게 없으면 우리 쪽 수치가 전부 틀린다)**: yaw 규약.
내부 θ 는 OG 규약(0°=+x, 반시계)이고 `screen_x_sign=-1` 은 그 규약의 표현이다. 우리
평가기는 `bearing=atan2(dx,dz)`·0°=+z·시계 증가·부호 +1 을 가정한다. 미러가 아니라
**ψ = 90° − θ** 로 일치함을 합성 검증(오차 0°)으로 확인하고, 내보내기 직전에 `live`·`map`
의 `yaw` 를 변환(`yaw_og` 보존), `scene_meta.yaw_convention="ours"`, `screen_x_sign=+1`,
그리고 **우리 공식으로 재감사(`yaw_audit_ours`, 0.5° 게이트)** 를 추가했다. 내부 계획기는
그대로 θ 를 쓴다. → 다음 실행에서 `yaw_audit_ours.median_abs_error_deg ≈ 0` 을 확인할 것.

**고친 것 2 (평가기)**: 목적지가 에피소드 방 목록 밖(`outside`·범위 밖 방)이면 ④ 로 채점.

**미결 (a) 경우④**: 선택지 1(범위 밖 방으로 반출)을 권장. 실외를 안 쓰고 "관측된 집에서
사라짐" 이 성립하며, 평가기는 위 수정으로 `tgt ∉ rids` 를 ④ 로 센다. 그런 방이 없는
씬은 `--case4 0` 유지.
**미결 (c) 소품 주입**: 유지. `gt0[].injected` 플래그로 사후 필터 가능하니 벤치 설계로
SPEC 에 명시(“BEHAVIOR 씬 원본만으로는 케이스 예산 불가 — 소품 24개 주입”).

**확인 못 한 것**: 실행 환경(Isaac)이 없어 정적 검토만 했다. `og_pose_verify.py` 는 OG
규약 기준이라 그대로 두었고, 내보낸 gt.json 의 최종 판정은 `yaw_audit_ours` 가 한다.
