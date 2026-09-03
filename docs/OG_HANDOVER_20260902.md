# OmniGibson 생성기 인수인계 (2026-09-02, rc-rtxpro)

이 문서만 읽고 이어받을 수 있게 쓴다. 원인 분석의 상세 근거는
[OG_DIAG_20260902.md](OG_DIAG_20260902.md), 요구 조건은
[OMNIGIBSON_EPISODE_SPEC.md](OMNIGIBSON_EPISODE_SPEC.md)·
[PORTING_CHECKLIST.md](PORTING_CHECKLIST.md).

## 0. 한 줄 상태

**생성기는 SPEC 4-2 를 다 물린 상태로 전 게이트를 통과하며, 재현성과 씬 일반성까지
확인됐다** — `Rs_int` seed 0/1 과 `Pomaria_1_int` seed 0, 세 판 모두 `EPISODE_OK`(§5-2).

**최종 데이터는 아직 만들지 않았다.** 남은 것은 §10 — 전 씬 연결성 스윕으로 20채
씬·seed 목록을 확정하고, `Pomaria_1_int` 처럼 여유가 얇은 씬을 위해 케이스 목표에
마진을 두는 것.

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
    --house 0 --frames 2600 --props 24 --case1 10 --case2 5 --case3 5 --case4 0
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
| `frames ≥ 1200` | SPEC 2 (기본값 2600 — evidence 장치가 대본만 ~2050 프레임) |
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
| v10 | 2000 | 24 | **6** | **8** | 72 | 0.0141 | 1 | evidence 창 수정 적중(최단 2.08→0.66~1.94m), 증인 1/5→6/8. 점프 1건 |
| v11 | 2000 | — | — | — | — | — | — | 예산 가드: 2343 프레임 필요. `smooth_program` 이 `walk_to` 출력을 재분할 |
| v12 | 2000 | — | — | — | — | — | — | 허용치 통일 후 2031 프레임 — 예산 31 프레임 초과 |
| **v13** | **2600** | **23** | **6** | **7** | **74** | **0.0142** | **0** | **EPISODE_OK · 이동 16/16 · max step 0.27m** |

읽는 법: v6·v8 도 게이트는 통과했지만 방 폴리곤 버그를 안고 있었다. v9 가 수치상
후퇴한 것은 새 감사 항목(`field_convention`)이 그 숨은 결함을 드러냈기 때문이고,
고치면서 노출된 잔여 항목을 v10~v12 에서 하나씩 닫아 **v13 이 첫 온전한 합격판**이다.
② 는 v6 5 · v8 6 · v9 3 · v10 6 · v13 6 으로 **판마다 흔들린다** — 그래서 §10 의
1 번(다른 seed 로 재현성 확인)이 20채 생성보다 먼저다.

### 5-2. 재현성·일반성 확인 (§10 의 1~2 완료)

| | `Rs_int` s0 | `Rs_int` s1 | `Pomaria_1_int` s0 |
|---|---|---|---|
| 방(범위) | 5 | 5 | **6** |
| 프레임 | 2600 | 2600 | **2961**(자동 확장) |
| 이동 적용 | 16/16 | 13/16 | 16/16 |
| ① | 23 | 23 | **11** |
| ② | 6 | 6 | **5** |
| ③ | 7 | 6 | 7 |
| 보행 | 0.738 | 0.742 | **0.782** |
| 0.5m 초과 점프 | 0 | 0 | 0 |
| yaw° | 0.0142 | 0.0141 | **0.0073** |
| depth ±0.6m | 0.900 | 0.917 | **0.771** |
| `gt0` 폴리곤 포함 | 0.924 | 0.937 | 0.967 |
| 증인 | 13/16 | 9/13 | 13/16 |

읽는 법:

- **② 가 두 seed 에서 모두 6 으로 안정됐다.** v6~v10 에서 5~6 과 3 사이로 흔들린
  원인(evidence 창이 판정 기준 밖)이 해소된 것이 두 번째 판으로 확인됐다.
  이동 적용 수는 16/16 ↔ 13/16 으로 변동하지만 `--spare 3` 여유분이 흡수한다.
- **`Pomaria_1_int` 는 여유가 얇다** — ① 11(기준 10) · ② 5(기준 5)로 딱 걸쳐 있다.
  20채 생성에서는 `--spare` 를 올리거나 `--case1/--case2` 목표에 마진을 둘 것.
  물체 funnel 이 `non_structural 101 → unique 34 → not_fixed 26 → seen 21 →
  carryable 18`(주입 18)이라 후보 자체는 충분하다 — 얇은 것은 ① 의 "그 방 재방문 2회"
  조건이고, 방이 6 개로 늘어 방당 체류가 분산된 결과다.
- **depth-투영 일치가 씬마다 다르다**(0.90 → 0.77). 큰 가구가 많으면 AABB **중심**을
  투영해 **앞면** 깊이와 비교하는 정상 편향이 커진다. 게이트는 아니지만 도메인 지표로
  기록할 값이다.
- 방 개수가 늘면 대본 길이도 늘어(`Rs_int` 5방 ~2050 → `Pomaria_1_int` 6방 ~2820)
  프레임 예산이 자동 확장된다. `audit.frames_requested` 와 `frames_budget` 을 비교하면
  확장 여부가 보인다.

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
   (v3: planned 3, moves 0). 초과하면 **예산을 늘린다**(`widen_budget`) — 대본 길이는
   방 개수에 비례하므로 씬마다 `--frames` 를 손으로 맞추는 방식은 20채 생성에 맞지 않다.
   `--frames-max-factor`(기본 2.5) 를 넘으면 그때는 실패시킨다.

---

## 7. v9~v13 에서 잡은 것 (측정 → 원인 → 고침)

| v9 결함 | 측정된 원인 | v10 수정 |
|---|---|---|
| ② 3/7 | `evidence_goals` 허용 창이 `0.6D~1.4D` = 0.9~2.1m 인데 판정 기준은 **≤2.0m**. 실측 최단 2.03·2.08·1.93m 로 자기 창 안에서 기준 초과 | 창을 `[0.6D, min(1.4D, 2.0−0.25)]` = 0.84~1.75m 로 클램프, 가까운 후보부터 정렬, D 기본 1.5→1.4 |
| 점프 1건 (0.509m) | `t=0→1` 한 곳. A* 호길이 리샘플이 `total/round(total/step)` 이라 `--step` 을 넘을 수 있다 | 리샘플 후 인접 간격이 `step*1.05` 를 넘으면 세분화 |
| 수납 증인 1/5 | 물체가 하부장 안 높이 0.07·0.12·0.49m, 상부장 2.05m. 증인 카메라가 pitch 0 이라 44° 아래 = FOV 밖 | 증인 카메라를 `pitch = atan2(d_z, |d_xy|)` 로 조준, 반경 후보 (2.0,1.6,1.3,2.5), 화면 안 여부까지 검사 |

v10 에서 셋 다 효과가 확인됐다(② 3→6, 증인 1/5→6/8, 최단 접근 2.08→0.66~1.94m).
남은 0.5m 점프 1 건은 v11~v13 에서 아래처럼 닫았다.

| v10 잔여 | 측정된 원인 | 고침 |
|---|---|---|
| 점프 1 건 (0.509m, `t=0→1`) | `walk_to` 가 자기 leg 를 세분화해도 남았다. 어느 생산 경로가 냈는지 추적 불가 | 생성자별로 잡지 않고 **모든 phase 가 지나가는 `smooth_program()` 에서 위치·heading 두 불변식을 함께 강제**. phase 이음새는 seed 로 덮는다 → v13 max 0.27m, 초과 0 건 |
| 증인 6/8 | 남은 2 건은 하부장 앞 공간이 몸반경 0.25m erode 로 지워져 `ng.snap`(보행 셀만) 이 서지 못함 | 증인 렌더는 걸어갈 필요가 없으므로 **erode 전 free 격자** 후퇴 경로 추가 → 13/16 |
| (v11 회귀) 예산 2343 필요 | `walk_to` 는 `step*1.05` 로 세분화하는데 `smooth_program` 은 `step` 정확값으로 판정해 `ceil(0.2625/0.25)=2`, **모든 보행 프레임 사이에 1 프레임씩 삽입** | 판정 허용치를 `step*1.15` 로 통일 → 2343→2031 |
| (v12) 예산 31 프레임 초과 | evidence 구간 사이 위치 불연속이 이제 실제 보행 프레임으로 채워진다(정당한 증가) | `--frames` 기본 2000→**2600** |

**교훈**: 같은 불변식을 두 곳에서 **서로 다른 허용치로** 판정하면 한쪽이 다른 쪽의
출력을 무한히 쪼갠다. 불변식은 한 지점에서, 한 허용치로 강제할 것.

`--spare` 는 3(계획 = quota+3)으로 이동 실패 여유를 둔다.

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

**v13 실측 (합격)**
```
EPISODE_OK · frames 2600 · moves 16/16 · ① 23 · ② 6 · ③ 7(수납 8)
continuity  moving 0.738 · walked 446.7m · max step 0.27m · 0.5m 초과 0 · 45° 초과 0
yaw         median_abs 0.0142° · screen_x_sign −1
depth       ±0.6m 내 0.900 (n=14495)
field_conv  gt0 0.924 · moves 1.0 · moves_from 0.938 · live 0.970 · map 1.0 · static 0.957
```

**남은 흠 2 개(게이트는 통과, 최종 데이터에는 영향 없음)**

1. `selfcheck_ready.geo_ok = false` — 증인 렌더가 16 건 중 13 건. `gen_selfcheck` 는
   **모든** 이동에 `witness_file` 을 요구하므로 그대로는 False 다. 수납 이동은
   `hidden_verified` 8/8 이 실질 게이트이므로, `gen_selfcheck` 쪽에서 수납 이동을
   `hidden_verified` 로 판정하도록 하거나(HSSD 공용 스크립트 수정) 증인 탐색을 더
   넓히는 두 선택지가 있다. **결정 필요.**
2. `evidence.closest_m` 중 2 건이 2.84·3.28m — evidence 지점을 못 찾아
   `approach` 후퇴로 갔거나 그마저 실패한 물체다(`notes` 에 기록됨). ② 는 6 건으로
   기준을 넘으므로 표본에는 영향이 없다.

미달 시 `audit.evidence.closest_m` 와 `event_status` 를 먼저 보라 — 어느 물체가
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

### (b) 20채 구성 — 실측 완료

`scripts/og_scene_screen.py` 로 22 씬을 **생성기와 같은 범위 규칙**(라이브 navmesh
연결성 → 실외 방 유형 드롭 → 주방·화장실 필수 → 방 4~8)으로 재었다.
결과는 `og_diag_20260902/scene_screen.jsonl`. **7 씬 통과.**

| 씬 | indoor 방 | 트림 | 비고 |
|---|---|---|---|
| `Rs_int` | 5/5 | — | 생성 검증 완료(seed 0·1) |
| `Pomaria_1_int` | 6/6 | — | 생성 검증 완료(seed 0) |
| `Merom_1_int` | 7/7 | — | |
| `Wainscott_0_int` | 9/9 | ✔ | |
| `Ihlen_1_int` | 10/10 | ✔ | |
| `Wainscott_0_garden` | 11/12 | ✔ | 실외 1방 드롭 |
| `house_single_floor` | 19/20 | ✔ | 실외 1방 드롭 |

탈락 15 씬은 **전부 주방/화장실 미충족**이다 — 방 수가 부족해 떨어진 씬은 없다.
`missing ['kitchen']` 이 8 건으로 가장 많다.

**맵 기반 추정은 믿지 마라.** 이 표를 만들기 전 배포 맵으로 세운 추정과 실측이 크게
달랐다:

- `Beechwood_0_int` — 맵 11방(주방·화장실 포함)인데 라이브 4방, 주방·화장실 둘 다
  미연결 → **탈락**. 부분집합 후보로 적어 뒀던 씬이다
- `Merom_1_int` — 배포 맵 4/12 로 "연결 나쁨" → 라이브 **7/7 통과**
- `Ihlen_1_int` — "2층이라 불가" 로 제외했는데 **10/10 통과**.
  문 일괄 개방(`Open.set_value(True)`)일 때 5/13 이던 것이
  `open_doors_for_clearance`(문마다 통로가 실제로 열리는 관절값)로 10/10 이 됐다
- `Rs_garden`·`house_double_floor_lower` — case4 후보로 언급했으나 화장실 미연결 → 탈락

**구성 방침**: 7 씬 × seed 로 20채를 만든다. 방이 많은 4 씬(`Wainscott_0_int`·
`Ihlen_1_int`·`Wainscott_0_garden`·`house_single_floor`)은 seed 마다
`NavGrid.restrict()` 가 다른 8방 부분집합을 잡으므로 같은 씬에서도 채 구성이 달라진다.
`Pomaria_1_int` 처럼 여유가 얇은 씬을 감안해 `--spare 4~5 --case1 12` 마진을 둘 것.

**스크리너는 씬당 별도 프로세스로 돌려라.** OmniGibson 3.9 는 한 프로세스에서 두
번째 `InteractiveTraversableScene` 로드를 거부한다(`Simulator must be stopped before
loading scene!`) — `env.close()` 로도 안 된다. 셸에서 루프를 돌린다.

### (c) 소품 주입의 위상

BEHAVIOR 씬 내용만으로는 케이스 예산이 **원리적으로 불가능**하다(주방+화장실+4~8방
씬 14개 중 최선이 유니크·이동가능 물체 9개, 필요량 10+). 그래서 소품을 주입한다.
이것을 벤치 설계로 문서에 명시할지, 아니면 "씬 원본만 사용" 을 유지하고 케이스 목표를
낮출지는 과제 정의의 문제다. 현재는 주입을 쓰고 `gt0[].injected` 로 표시해
사후 필터가 가능하게 해 두었다.

---

## 9. 폐기 대상 (조용히 재사용되면 안 되는 것)

- `og20_smoke*` 300프레임 판 전부 — 카메라가 바닥을 본 top-down
- `og20_v2~v5`, `v7`, `v9`~`v12` — 게이트 미통과. `v6`·`v8` 은 통과했지만 방 폴리곤
  버그가 있고, `v10` 은 점프 1 건이 남아 최종 데이터로는 쓰지 말 것.
  **`v13` 이 첫 합격판이다**
- `og51_probe_20260902` 200프레임 입장시험 프레임 — 같은 쿼터니언 버그로 바닥 크롭에
  AUC 를 잰 셈. `og51_rtx7_sweep_20260902.txt` 는 빈 파일 = 판정 미완. **OmniGibson
  입장 시험은 아직 성립하지 않았다**
- `--smoke` 의 direct path fallback — 새 생성기에는 그 경로가 존재하지 않는다
- 하드코딩 `FX = W*17.0/20.995` — 센서에서 읽는다

---

## 10. 다음 단계 (순서대로)

1. **같은 설정으로 seed 를 바꿔 한 번 더** 통과시켜 재현성 확인. ② 가 v6 5 · v8 6 ·
   v9 3 · v10 6 · v13 6 으로 흔들렸으므로 1 회 통과로는 부족하다.
4. `scripts/gen_selfcheck.py` 로 이동 물체 검출 self-test(SPEC 4-2.7).
   수납 이동은 문 열린 상태 증인 크롭을 채점한다.
5. 20채 생성 → 검수 5줄 보고 → `eval_online` 체인.
6. 입장 시험(§9)을 정상 수평 프레임으로 다시 재고 `SIM_SCREENING` 대장을 갱신.

---

## 11. 아이맥 검토 결과 (v2 패치 시점) (2026-09-02 저녁, 패치 적용 후)

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


---

## 12. 아이맥 결정 (2026-09-03, v4 델타 적용 후)

v4 델타는 `patch -p1` 로 붙였다(og_episode 6헝크 중 1 — 예산 자동확장 — 은 v2/v3 차이로
손 병합, `--frames-max-factor` 인자 추가 확인). yaw 변환 블록(§11)은 유지됐다.

세 질문에 대한 답:
1. **20채 생성은 아직 시작하지 말 것.** 검수 영상에서 사용자가 지적한 것이 데이터의 본질
   문제다: "한 사람이 같은 공간을 쉬지 않고 계속 돌고, 너무 빠르다." 10h 생성 뒤 다시 만들면
   낭비이므로 **SPEC 4-4(보행 사실성)** 를 먼저 넣고 짧은 판으로 확인한 뒤 시작.
2. **case4 = 선택지 1**(에피소드 범위 밖 방으로 반출). 평가기는 `tgt ∉ rids` 를 ④ 로 센다.
   그런 방이 없는 씬은 `--case4 0`. 정의 변경 대기 없이 진행.
3. **gen_selfcheck 규칙 확정**: 수납 이동은 `hidden_verified=True` 이면 증인 없이도 기하
   게이트 통과(정의상 2m 시선이 없다). 스크립트에 반영했으니 pull 후 `geo_ok` 가 True 여야 한다.
   `Pomaria_1_int` 처럼 경계에 걸친 씬은 케이스 목표에 마진(+2)을 두고 쓴다.

---

## 13. 아이맥 회신 (2026-09-03 오후, 4채 실패 재분석에 대해)

세 수정 모두 동의: (1) case2-between 을 최근접 방으로, (2) prelude 2회 순회, (3) case4 를 트림된
방으로 실제 구현. 원인 분석(구조적 보행 거리·1회 순회·미구현)이 측정에서 나온 것이라 채택.

**보강 하나 — case4 의 "안 보임" 보장.** 트림된 방은 navmesh 밖이라 궤적이 못 들어가지만, 문
너머로 **시선이 닿으면** `capture()` 의 레이캐스트가 그 물체를 vis 로 잡는다. 그러면 평가기는
`to ∉ rids` 라 ④ 로 세지만 시스템은 폴리곤 없는 곳으로 투영해 옆 방을 답한다(오답이 아니라 정의
불일치). 조치: case4 배치 뒤 **범위 안 보행 셀 전부(또는 방마다 샘플 30개)에서 시선 검사** →
하나라도 닿으면 다른 트림 방/다른 지점으로 재배치, 없으면 `--case4 0`. 증인 렌더는 요구하지
않는 것이 맞고, `supported` 는 유지(떠 있는 물체는 GT 정직성 문제).

패치 `og_spec3_spec44_20260903.patch` 는 드라이브에 아직 없다(E:\ 에만). 올라오면 base d43ce4a
위에 적용한다 — 아이맥 HEAD 의 og_* 는 d6eaff1(v3 본체+v4 델타+yaw 변환) 이후 손대지 않아 충돌 없을 것.


---

## 14. 아이맥 반영 (2026-09-03 저녁) — Wainscott 검증과 20채 재시작에 대해

Wainscott(방 8, `--case4 2`) 단일 검증: ① 15 · ② 7 · ③ 7 · ④ 2/2 통과, `moving_frame_frac` 0.605 만 실패.
§13 의 case4 시선 누출 방지(`raw_room_points` 깊숙한 지점 우선 + `case4_invisible_from_scope` 방마다 20샘플)
구현 확인. 20채는 첫 채 완료 시 멈춰 확인 후 진행 — 맞는 절차.

**결정**: `moving_frame_frac` 게이트를 **≤ 0.6** 으로 완화 (SPEC 4-4.1 수정). 이유: 큰 씬의 보행량은
구조적(prelude 2회 + 증거 방문 3 + 경유방)이라 체류로 못 잡고, 체류 120프레임(2분/정지)을 더 늘리면
사실성이 반대로 깨진다. 보행을 줄이려면 **횟수**를 줄여라: prelude 2회는 재방문이 부족한 정적 물체의
방에만, 경유방은 최근접(이미 반영). `--dwell 120·--filler-dwell-mult 2.0·--frames-max-factor 4.0` 은 유지.
평가 쪽 영향: 체류 프레임이 늘면 앵커 투표·부재 판정에 유리하고 ② 증거 수는 `--evidence` 가 결정하므로 불변.

**요청**: 첫 채 확인 시 `audit.json` 의 5줄 + `dwell_frame_frac`·`mean_dwell_frames`·`room_switches(목적지)`
·`case4_invisible` 결과를 보고. 완료 후 델타 패치(경유방 최근접·prelude 2회·case4 구현·시선 검사)를 드라이브에.
