# bench-v2 를 다른 머신(4090 · RTX PRO 6000)에서 돌리기

`git pull` 만으로는 안 된다. **코드는 repo 에 있지만 데이터·환경·채점기가 머신마다 다르다.**
아래 4가지를 갖추면 `scripts/bench_v2_chain.sh` 한 줄로 M1 Max 와 같은 체인이 돈다.

## 1. HSSD 데이터셋 (HF gated, ~10GB)
```bash
huggingface-cli download hssd/hssd-hab --repo-type dataset --local-dir ~/hssd-hab
# scenes-uncluttered · objects(decomposed 포함) · metadata/fpmodels*.csv · semantics/scenes 가 필요
```
- 장면 20채 목록: `docs/bench/hssd20_scenes.txt` (M1 Max 와 동일)
- 시나리오 사전확률: `data/hssd_move.json` (repo 에 있음)

## 2. habitat-sim (bullet 물리 필수 — 없으면 이동이 렌더에 안 나온다)
```bash
conda create -n hab python=3.9 -y && conda activate hab
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y   # Linux: headless(EGL)
```
`HAB=$CONDA_PREFIX/bin/python` 로 넘긴다.

## 3. torch 환경 (OWL·캐시·평가)
기존 `~/kx-venv` 면 된다 (`exp_anchowl` 은 CUDA 자동 선택). `PY=~/kx-venv/bin/python`.

## 4. 채점기 — **여기가 M1 과 다르다**
| | M1 Max | Linux/CUDA |
|---|---|---|
| 스크립트 | `exp_t1_verify_mlx.py` (mlx 4B mxfp4) | `exp_t1_verify_pipeline.py` (HF Qwen3.5-9B) |
| 문턱 | s_ab 2.069 · s_ac 0.887 (mlx4B·HSSD 캘리브) | **그 머신에서 다시 잰다** → `CALIB=1` |

`CALIB=1` 이면 체인이 make_sim_pairs(500쌍) → exp_vlm_verify3 → 기각 0.95/0.90 분위로
VERIFY_TH/TH2 를 자동 산출한다(§89: 문턱은 이식 불가). 채점기가 다르므로 **② 채택률은
M1 수치와 직접 비교하지 않는다** — 같은 데이터·같은 코드에서 "9B 가 4B 보다 얼마나
낫나" 로 읽는다. ①·초기맵·검색·기준선·CI 는 채점기와 무관하니 그대로 비교된다.

## 실행
```bash
git pull
HSSD_DATASET=~/hssd-hab/hssd-hab-uncluttered.scene_dataset_config.json \
HAB=~/miniforge3/envs/hab/bin/python PY=~/kx-venv/bin/python \
SCORER=hf CALIB=1 nohup bash scripts/bench_v2_chain.sh > bench_v2.log 2>&1 &
```
- 생성 20채×1200 (GPU 무관, CPU 렌더 ~1.5h) → **기하 자가검사 게이트**(실패 시 중단:
  `BENCH_V2_FAIL_SELFCHECK` — 이동 물체가 받침 위·시선 안에 있어야 한다, §125)
  → 캐시(BOXES=1) → 초기맵 → 캘리브 → 채점 2회(9B 라 4090 에서 각 1~2h) → 평가 2회
- 완료 표식 `BENCH_V2_DONE`. 보고는 `=== bench-v2 ===` 두 블록(재료 사다리·3경우·기준선·CI)
  과 자가검사 20줄, `캘리브 →` 한 줄.

## 무엇이 같고 무엇이 다른가
- 같음: 장면·시드·프레임·이동 대본·캐시 코드·초기맵·평가군·기준선·CI
- 다름: 렌더 드라이버(EGL vs Metal — 픽셀은 사실상 동일), 채점기(9B vs 4B), 문턱
- 이미 생성한 데이터를 옮겨 쓰려면 `SKIP_GEN=1` + `OUT=` 으로 지정 (rsync 로 20채 복사)
