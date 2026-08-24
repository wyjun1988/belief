# 4090 실행 런북

**목적: 경우 2(부재 → belief) 표본 확보.** 이게 이 프로젝트의 고유 능력인데
현재 데이터로는 통계가 안 나온다.

```
아이맥 16채 × 1시간   이동 45개 → 원래 방 재방문 18개 → 앵커 조건 충족 7개
```

7개로는 어떤 부재 기제를 시험해도 CI 가 무의미하다. **지평을 늘리면 재방문이
늘어난다** — 1시간에 방을 6번 옮기는데, 3시간이면 18번이라 떠난 방을 다시 볼
확률이 크게 오른다. 이동 건수도 8 → 30 으로 올린다.

## 한 줄 실행

```bash
git pull
bash scripts/gpu_preflight.sh                   # ★ 먼저 이것 (1분). 렌더 가능한지 가른다
bash scripts/gpu_4090_run.sh                    # 기본 60채 × 3시간 · 이동 30건
HOUSES=100 HOURS=4 bash scripts/gpu_4090_run.sh # 더 크게
```

**사전점검을 먼저 돌려라.** 실제로 CloudRendering 컨트롤러를 띄워 30프레임을
렌더하고 **fps 와 총 소요 예상**까지 찍는다. 여기서 실패하면 그 장비로는
[1/4] 생성을 못 하므로, 몇 시간을 태우기 전에 알 수 있다.

## H100 / A100 에서도 되나

**추론은 문제없고, 렌더링이 관건이다.**

| 단계 | H100 |
|---|---|
| OWL 캐시 ([2],[3]) | ✅ 4090 보다 빠르다 |
| **생성 ([1])** | ⚠️ Vulkan 이 노출돼야 한다 |

데이터센터 카드도 Vulkan 을 지원하지만, **클라우드 이미지가 그래픽을 안 켜주는
경우가 흔하다** — 컨테이너 기본값이 `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
라 그래픽 라이브러리가 안 붙고, MIG/vGPU 구성이면 아예 막힌다.

성능 자체는 걱정하지 않아도 된다. 384×384 `quality="Low"` 는 래스터 부하가
미미해서 H100 의 약한 그래픽으로도 충분하다. **실패는 이분법적이다** — 되거나
아예 안 되거나. 그래서 사전점검이 있다.

컨테이너면 이렇게 띄운다:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ...
```

**렌더가 안 되면**: 생성만 다른 장비(4090 급 소비자 GPU)에서 하고, 프레임을 옮긴 뒤
[2],[3] 단계만 H100 에서 돌리면 된다. 다만 프레임이 60채×3시간이면 ~30 GB 라
옮기는 비용이 크다 — 한 장비에서 다 하는 편이 낫다.

## 소요 시간 짐작

MBP(M1 Pro) 실측을 기준으로 잡은 것이다:

| 단계 | MBP 실측 | 4090 추정 |
|---|---|---|
| 생성 20채 × 1시간 (72k 프레임) | 53분 (22 fps) | ~20분 |
| 앵커 캐시 16채 stride 8 (7.6k) | 62분 | ~6분 |

기본 설정(60채 × 3시간)은 **648k 프레임**이라 생성이 가장 길다. 사전점검이 찍는
fps 로 실제 예상 시간을 확인하고, 길면 `HOUSES=40` 으로 줄여도 된다
(경우 2 표본이 ~130 개로 충분하다).

끝나면 `harvest_MMDD.tar.gz` 하나만 가져오면 된다.

## 환경 준비 (처음 한 번)

```bash
python -m venv ~/thor-venv
~/thor-venv/bin/pip install ai2thor==5.0.0 prior==1.0.3 numpy==2.0.2 pillow==11.3.0
~/thor-venv/bin/pip install torch torchvision transformers==4.57.6 scipy
```

`ai2thor` 와 torch 를 한 venv 에 둬도 리눅스에서는 대체로 괜찮다(맥에서는 numpy
고정 충돌로 나눴다). 충돌하면 나누고 `PY=` 로 각 단계 인터프리터를 지정하면 된다.

### ⚠️ CloudRendering 이 첫 관문이다

헤드리스 리눅스에는 Unity 창이 없다. `--platform cloud` 를 줘야 뜬다(드라이버가
스크립트에 이미 들어 있다). **Vulkan 이 필요하다:**

```bash
vulkaninfo --summary | head       # deviceName 이 나와야 한다
nvidia-smi                        # 이것만 되는 걸로는 부족하다
```

안 잡히면: `apt install -y vulkan-tools libvulkan1 libnvidia-gl-<버전>`.
컨테이너면 `NVIDIA_DRIVER_CAPABILITIES=all` 로 띄워야 그래픽 라이브러리가 붙는다.

**H100 은 권장하지 않는다** — 데이터센터 카드라 래스터가 빈약하고 Vulkan 이
안 잡히는 인스턴스가 흔하다. 렌더링은 4090 이 맞다.

## 무엇을 가져오나 — 프레임은 두고 온다

실측(1채 = 1시간 3,191프레임):

| | 크기 | |
|---|---|---|
| `live/` JPEG | 54 MB | ❌ 서버에 둔다 (전체의 99.6%) |
| `gt.json` | 3.5 MB | ⚠️ 축약해 가져옴 |
| **`a3_` 앵커 캐시** | **176 KB** | ✅ OWL 추론 결과 |
| **`qc_` 질의 캐시** | **52 KB** | ✅ |

16채 실측: gt 41.6 MB → 6.9 MB, 캐시 2.9 MB, **tar 합계 3.3 MB**.
60채 × 3시간이어도 100 MB 안쪽이다.

**분석은 전부 아이맥에서 돈다**(numpy 뿐). 실험 설계를 바꿔가며 반복해도 GPU 를
다시 안 빌려도 된다 — 이번 세션에서 캐시 하나로 십수 가지 분석을 돌렸다.

### `--stride 4` 인 이유

프레임을 두고 오면 **되돌릴 수 없다.** 검출기를 바꾸거나 stride 를 줄이려면
프레임이 필요하다. 8 대신 4 로 뽑아두면 용량은 2배지만 여전히 작고, 나중에
후회할 일이 없다.

## 단계별로 나눠 돌리기

전체가 오래 걸리므로 중간에 끊겼다면 이어서 할 수 있다. 생성은 주택별 디렉터리,
캐시는 주택별 npz 로 저장되고 **이미 있으면 건너뛴다.**

```bash
PY=~/thor-venv/bin/python
$PY scripts/thor_gen2.py --platform cloud ... --out data/thor4     # [1]
THOR_ROOT=data/thor4 CACHE_PREFIX=/tmp/a3_ $PY scripts/exp_anchowl.py 4   # [2]
THOR_ROOT=data/thor4 QCACHE_PREFIX=/tmp/qc_ STRIDE=4 $PY scripts/exp_imgq.py  # [3]
$PY scripts/harvest.py --root data/thor4 --cache /tmp --out harvest.tar.gz --keep-geom
```

## 되가져온 뒤 (아이맥)

```bash
tar xzf harvest_MMDD.tar.gz -C /tmp/h4
mkdir -p data/thor4 && cp -r /tmp/h4/gt/* data/thor4/
cp /tmp/h4/cache/*.npz /tmp/
THOR_ROOT=data/thor4 python scripts/eval_paths.py
THOR_ROOT=data/thor4 python scripts/eval_update.py 이미지
THOR_ROOT=data/thor4 python scripts/eval_absence_anchor.py 이미지
```

## 알려진 함정

1. **`visibilityDistance` 기본 1.5 m** — 스크립트가 20 으로 넣는다. 직접 부를 때
   빼먹으면 GT 가 오염된다(§58). 물체당 "보인" 프레임이 6장 대 122장으로 갈렸다.
2. **`renderInstanceSegmentation=True`** 없으면 bbox 가 없어 exemplar 를 못 만든다.
3. **사전확률을 넣었으면 생성 데이터에서 되재라.** 유형 확률을 방 인스턴스마다
   그대로 주면 유형 분포가 깨진다(§67). 확인:
   ```bash
   THOR_ROOT=data/thor4 python scripts/eval_movecheck.py
   ```
4. **`text_embeds[0]` 을 쓰지 마라** — 이미 (질의수, dim) 이라 `[0]` 은 첫 물체
   질의를 전체에 복사한다. 실제로 물렸고 글자 질의 수치가 통째로 무효였다.
   `exp_imgq.py` 에 assert 를 넣어뒀다.
5. **exemplar 점수에 시그모이드 금지** — `class_head` 의 shift/scale 은 텍스트
   스케일에 맞춰 학습돼 포화한다(중앙 0.994). 코사인 내적을 그대로 쓴다.
   순위는 살아남아 정밀도·AUC 는 유효하지만 **부재 검출은 절댓값이 필요하다.**

자세한 THOR 셋업은 `docs/THOR_SETUP.md`.
