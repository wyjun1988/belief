# khronos — ADT 위 동적 4D Scene Graph

Meta **ADT(Aria Digital Twin)** 관측 스트림에서 시간에 따라 갱신되는 3D scene graph 를 만든다.
구성요소는 MIT-SPARK 의 **DAAAM**(4D scene graph)·**Khronos**(단기 동역학 + 장기 변화 SLAM)와
**Depth Anything 3**(모노 뎁스). RGB-D 센서가 없는 Aria 스트림에 모노 뎁스를 이식하는 것이
핵심 기술 과제다.

시작 2026-08-14 · 계획서 `~/.claude/plans/melodic-munching-quasar.md`

## 전제 (조사로 확정한 것)

**DAAAM 은 이미 Khronos 를 액티브 윈도우 프론트엔드로 쓴다** — `install/packages.yaml` 이
`nicogorlo/Khronos` 를 받고, standalone 실행이 `--hydra-config coda_dataset_khronos` 다.
따라서 이 프로젝트는 두 시스템을 접붙이는 게 아니라 **CODa(스테레오 뎁스) 자리에 ADT(모노
뎁스)를 물리는 데이터·뎁스 레이어**를 만든다. DAAAM 저장소는 포크하지 않는다 —
`DatasetFactory.register_dataset_type()` 과 `SegmenterInterface` 두 지점만 런타임에 끼운다.

| 항목 | 값 |
|---|---|
| GPU | RTX 3090 24GB (RunPod, Ubuntu 24.04, ROS2 Jazzy). P4 에서 A6000 48GB — 둘 다 sm_86 이라 빌드 재사용 |
| 파일럿 | `party_seq102_M1292`(965프레임, 이동 11건) · `decoration_seq137_M1292`(918프레임, 이동 6건) |
| 프레임 | 10Hz, 704×704 선형(핀홀) 90° HFOV, GT 포즈 |
| 세그멘테이션 | 1차는 **ADT GT**(지각 오차와 기하 오차를 분리하기 위해). SAM 은 P4 |
| 시맨틱 | 1차는 **끔**. DAM 캡셔닝은 P4 (VRAM 20–35GB) |

## 파이프라인

```
ADT VRS ──▶ 언디스토트+정립 ──▶ rgb/ pose/ camera_info.json      (P0, 맥)
                              └▶ gt/depth gt/seg gt/objects.json  (평가 전용)
        ──▶ DA3 포즈조건부 윈도우 ──▶ 전역 앵커 정합 ──▶ depth/    (P1, GPU)
        ──▶ DAAAM(GT마스크) + Khronos + Hydra ──▶ dsg.json        (P2, GPU)
        ──▶ 위치오차·변화감지 P/R·지연·정적오탐 + rerun            (P3, 맥)
```

**뎁스 정합이 이 프로젝트의 핵심 주장이다.** DA3 를 슬라이딩 윈도우로 돌리면 윈도우마다
스케일이 달라 맵이 "숨쉬고", Khronos 는 그 호흡을 물체 이동으로 읽는다. 일반 비디오 SOTA
(DVD·DepthSync·StableDPT)는 인접 윈도우를 이어붙이는 **상대** 정합이라 길어지면 흐른다.
우리는 GT 포즈와 MPS 반-조밀 포인트가 있으므로 **전역 절대 정합**을 한다 — 자세한 근거는
`kx/depth/align.py` 문서화 주석.

## 저장소 지도

```
kx/adt/      calib(어안→선형) export(VRS→시퀀스) gt_timeline(채점 기준)
kx/depth/    anchors(MPS 앵커) align(T2 로버스트 affine + T3 시간 정칙화)
             da3_runner(포즈조건부 윈도우) metrics(정확도·TAE·3D 산포)
kx/bridge/   adt_dataset(DAAAM 로더 등록) gt_segmenter(SegmenterInterface 구현)
kx/eval/     graph_eval  ·  kx/viz/ rerun_viz
scripts/     fetch_adt export_sequence audit_export align_depth eval_depth
             runpod_bootstrap.sh run_daaam.py
data/adt →   ~/work/home-jepa/data/adt   (원본 ADT 단일 출처, 심링크)
data/seq/    내보낸 시퀀스
```

## 재현

```bash
P=~/work/stock-v2/.venv-mps/bin/python

python3 scripts/fetch_adt.py --preset pilot              # 코어 6.2GB
python3 scripts/fetch_adt.py --preset pilot --parts depth # 평가용 GT 뎁스 11.4GB
$P scripts/export_sequence.py --preset pilot
$P scripts/export_sequence.py --preset pilot --depth-only
$P scripts/audit_export.py --seq <name>                  # PASS 필수
```

`data/` 와 결과물은 저장소에 없다. **ADT 는 Meta 연구 라이선스 데이터라 원본도 파생
프레임도 커밋하지 않는다**(`.gitignore` 로 차단).

## 확립된 사실 (검증 로그)

- **MPS world 프레임 = ADT scene 프레임** — `closed_loop_trajectory.csv` 와
  `aria_trajectory.csv` 가 같은 타임스탬프에서 9mm 이내 일치(2026-08-14). 반-조밀 포인트를
  그대로 앵커로 쓸 수 있다는 전제가 여기서 나온다.
- **반-조밀 포인트에 ±43km 발산점이 섞여 있다**(0.1%). `inv_dist_std<0.005 & dist_std<0.02`
  필터로 45%(45만점) 생존 — 필터는 선택이 아니라 필수.
- **VRS 는 GT 보다 먼저 시작한다** — 파티 seq102 에서 365프레임(12.2초), decoration
  seq137 에서 230프레임. 자르지 않으면 provider 가 경계값으로 클램프한 포즈를 조용히
  돌려줘 앞부분이 통째로 오염된다(`export.py:_uniform_indices`).
- **투영 오차 중앙값 3.1–3.3 px** (704px 기준, 감사 C) — 어안→선형·cw90 회전·
  `T_device_camera`·GT 포즈 체인이 실제로 맞물린다는 증거.
- **CLIP 프레임 임베딩은 물체 존재판정에 못 쓴다** (2026-08-19). ADT 918프레임·119어휘
  전수 채점에서 CLIP 최고 F1 0.22, OWLv2 0.39. 재현율 상한이 특히 다르다 — CLIP 은
  어떤 판정 규칙으로도 0.29 를 못 넘고 OWLv2 는 0.89 까지 간다. 두 지각층의 프레임별
  z 상관은 **+0.083** 으로 사실상 합의하지 않으며, GT 와 맞는 쪽은 OWLv2 다.
  세션 간 belief 에서는 부호가 바뀐다(최빈방 대비 −22% → +20%).
  → `docs/RESULTS_OWL_20260819.md`
- **OWLv2 는 배치마다 텍스트를 이미지 수만큼 재인코딩한다** — HF processor 가
  `text=[queries]*B` 를 그대로 넘긴다. 텍스트를 1회 캐시하고 박스 예측을 생략하면
  0.84 → 0.57s/장 (원 경로와 최대오차 0.00e+00). 어휘가 클수록 차이가 커진다.
- **검증 지표를 측정 대상과 섞지 마라** — 프레임 정렬을 두 지각층의 z 상관으로
  검증하려다 +0.083 을 보고 정렬을 의심했다. 실제로는 정렬이 완벽했고(추출 프레임의
  CLIP 임베딩이 `index.npz` 의 같은 행과 최대 코사인, 5세션 전부 6/6) 낮은 상관이
  결과였다. 정렬 검증은 임베딩 자기최고 판정으로 해야 한다(`scripts/owl_align_check.py`).
