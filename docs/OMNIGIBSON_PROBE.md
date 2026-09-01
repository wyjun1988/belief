# OmniGibson 입장 시험 (RTX 노드) — 렌더가 검증기를 살리는가

목적 하나: **SIM_SCREENING 의 입장 시험** — BEHAVIOR-1K 주거 장면에서 근/원거리
물체 크롭 쌍을 만들어 VLM 로짓 AUC 를 잰다.
**통과선: 5m+ 버킷 AUC ≥ 0.70 그리고 전체 ≥ 0.85** (THOR 실측 0.52 / 0.78).
통과 시에만 생성기 포팅에 들어간다. c3c 생성과 GPU 를 나눠 쓰지 말 것 —
c3c 완주 후 시작 권장.

## 1. 설치 (처음 한 번, 디스크 ~30GB)

```bash
# Isaac Sim 은 pip 판이 제일 간단하다 (RTX PRO 6000 = 요구 하드웨어 충족)
python3 -m venv ~/og-venv && source ~/og-venv/bin/activate
pip install --upgrade pip
pip install omnigibson            # isaacsim 의존 포함 자동 설치 (NVIDIA EULA 동의 프롬프트)
python -m omnigibson.install      # BEHAVIOR-1K 자산 다운로드 (라이선스 동의)
# 검증: 헤드리스 렌더 한 장
python -c "import omnigibson; print(omnigibson.__version__)"
```

⚠️ 설치 중 EULA/라이선스 동의·계정 절차가 나오면 **사용자에게 확인** 후 진행
(자동 동의 금지). 실패 지점과 메시지를 그대로 보고.

## 2. 렌더 프로브 — 프레임·GT 덤프

```bash
python scripts/og_probe.py --scenes 5 --frames 200 --out /tmp/og_probe
```

`og_probe.py` 는 뼈대다(환경 확인 후 API 이름은 현지 조정 허용 — 의도 주석 참조):
장면당 무작위 카메라 포즈에서 RGB + 인스턴스 세그 + 물체 거리 GT 를 덤프한다.

## 3. 쌍 생성 → 채점 (THOR §101 과 동일 규약)

```bash
# 근/원거리 크롭 쌍 400개 (양성 = 그 물체 크롭, 음성 = 같은 프레임 다른 물체 크롭에 그 라벨)
python scripts/og_pairs.py --probe /tmp/og_probe --n 400 --out /tmp/og_pairs
# VLM 로짓 채점 (kx-venv 의 기존 스크립트 그대로)
MODEL=Qwen/Qwen3.5-9B PAIRS=/tmp/og_pairs OUT_JSONL=og_scores.jsonl \
  ~/kx-venv/bin/python scripts/exp_vlm_verify3.py
# 판정 (거리 버킷 포함)
~/kx-venv/bin/python scripts/rtx7_sweep.py og_scores.jsonl
```

## 4. 판정 후

- 통과 → 생성기 포팅 설계(배회·이동·case3 — **문·수납 상태 활용**이 이 도메인의
  본전이다: "서랍 속 리모컨" = ③ 표본의 자연 제조)
- 미달 → SIM_SCREENING 대장에 실측 기록 후 SPEAR 로.

병렬: M1 Max 에서 Habitat(+ReplicaCAD→HSSD) 같은 시험 진행 중 — 두 결과를
같은 표에서 비교한다.
