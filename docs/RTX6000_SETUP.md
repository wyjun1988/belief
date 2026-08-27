# RTX PRO 6000 (Linux) 셋업 & 첫 실행

**이 장비가 왜 중요한가.** ProcTHOR 에서 유일하게 막힌 것이 **크롭 해상도**다 —
384px 렌더의 십수 px 물체는 VLM 이 못 읽는다(시뮬 크롭 AUC 0.621 vs 실사 0.897,
§90). 인식기(OWL)는 시뮬·실사 동등(0.82 vs 0.838)이므로 **해상도만 올리면
검증기 축이 열릴 수 있고, 그러면 최대 병목 T1(27%p)을 시뮬에서 뚫는다.**
게다가 풀 그래픽 GPU 라 H100 의 Vulkan 부재 문제가 없다 — **생성·캐시·검증기를
한 대에서** 돈다.

## 0. 준비 (처음 한 번)

```bash
# 드라이버·Vulkan 확인 — CloudRendering 의 전제
nvidia-smi
vulkaninfo --summary | head        # deviceName 이 나와야 한다
# 없으면: sudo apt install -y vulkan-tools libvulkan1

git clone <repo> khronos && cd khronos
python3 -m venv ~/thor-venv
~/thor-venv/bin/pip install ai2thor==5.0.0 prior==1.0.3 numpy==2.0.2 pillow
python3 -m venv ~/kx-venv
~/kx-venv/bin/pip install torch torchvision transformers pillow numpy scipy

# 사전확률 JSON 2개가 repo 에 포함돼 있다 (data/thor_prior.json, thor_move.json)
bash scripts/gpu_preflight.sh      # 렌더 가능 여부 1분 판정
```

## 1. ★ 첫 실행 — 해상도 실험 (1~2시간)

**목적: 검증기가 몇 px 부터 작동하는가.** 이것이 이 장비의 첫 질문이다.

```bash
PY=~/thor-venv/bin/python
for SZ in 384 768 1024; do
  $PY -u scripts/thor_gen2.py --houses 20 --hours 1 --moves 10 \
    --min-rooms 4 --max-rooms 8 --max-nonbath 6 --min-nonbath 2 \
    --size $SZ --dwell 600 --seed 777 --vis-dist 20 --platform cloud \
    --prior data/thor_prior.json --move data/thor_move.json \
    --out data/res$SZ
done
```

이어서 각 해상도의 **검증기 쌍**을 만들고 로짓 채점:

```bash
KX=~/kx-venv/bin/python
for SZ in 384 768 1024; do
  THOR_ROOT=data/res$SZ CACHE_PREFIX=/tmp/r${SZ}_a_ $KX -u scripts/exp_anchowl.py 8
  THOR_ROOT=data/res$SZ QCACHE_PREFIX=/tmp/r${SZ}_q_ $KX -u scripts/exp_imgq.py
  # 검증기 쌍 200개 (GT vis+ctr 로 양성 / 다른 물체 크롭 음성)
  THOR_ROOT=data/res$SZ A3_PREFIX=/tmp/r${SZ}_a_ QC_PREFIX=/tmp/r${SZ}_q_ \
    OUT=/tmp/pairs$SZ CROP=$((SZ/3)) $KX scripts/make_sim_pairs.py
  MODEL=Qwen/Qwen3.5-9B PAIRS=/tmp/pairs$SZ OUT_JSONL=res${SZ}_scores.jsonl \
    $KX scripts/exp_vlm_verify3.py
done
```

**판정선**: `res*_scores.jsonl` 의 s_ab AUC 가 **0.85+** 이면 그 해상도로 진행.
(384 는 0.62 로 알려져 있다 — 비교 기준)

## 2. 대량 생성 (판정 후, 반나절)

해상도 R 이 통과하면:

```bash
$PY -u scripts/thor_gen2.py --houses 100 --hours 4 --moves 40 \
  --min-rooms 4 --max-rooms 8 --max-nonbath 6 --min-nonbath 2 \
  --size R --dwell 600 --seed 901 --vis-dist 20 --platform cloud --mapwalk \
  --prior data/thor_prior.json --move data/thor_move.json --out data/thor7
# 캐시 4종
THOR_ROOT=data/thor7 CACHE_PREFIX=/tmp/t7_a_ $KX -u scripts/exp_anchowl.py 4
THOR_ROOT=data/thor7 QCACHE_PREFIX=/tmp/t7_q_ STRIDE=4 $KX -u scripts/exp_imgq.py
THOR_ROOT=data/thor7 ACACHE_PREFIX=/tmp/t7_x_ STRIDE=4 $KX -u scripts/exp_anchor_exemplar.py
THOR_ROOT=data/thor7 CLIP_PREFIX=/tmp/t7_c_ STRIDE=4 $KX -u scripts/exp_clip_rooms.py
$KX scripts/harvest.py --root data/thor7 --cache /tmp --prefix t7_ \
  --out harvest_t7.tar.gz --keep-geom
```

가져올 것: `res*_scores.jsonl`(1단계) → 판정 후 `harvest_t7.tar.gz`(수백 MB).
**프레임은 서버에 둔다** — 우리 분석은 전부 캐시 위에서 돈다.

## 함정 (전부 실측으로 물린 것들)

1. `--vis-dist 20` 없으면 GT 오염 — 물체당 보인 프레임 6장 대 122장 (§58)
2. `renderInstanceSegmentation`·`renderDepthImage` 는 스크립트가 켠다. 직접 부를 땐 확인
3. 사전확률을 넣었으면 **생성 데이터에서 분포를 되재라** — `scripts/eval_movecheck.py` (§67)
4. 원격 실행은 **점수만 기록하고 문턱은 로컬 스윕** — 문턱은 도메인·모델마다 밀린다 (§89)
5. 같은 캐시 접두어를 두 프로세스에 주지 말 것 — "있으면 건너뛰기"는 동시성 안전이 아니다
