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

## 0.5 ⚠️ 생성 파라미터는 **임의값이 아니다** — 바꾸기 전에 읽을 것

시나리오는 실사용을 재현하려고 하나씩 근거를 쌓아 정한 것이다. 명령의 플래그를
임의로 조정하면 지금까지의 수치와 비교가 끊긴다.

| 플래그 | 값 | 근거 |
|---|---|---|
| `--prior data/thor_prior.json` | Qwen 3.5 9B 생성 | ProcTHOR 기본 배치는 유형 규칙("머그=부엌")이라 사전확률이 실제보다 강해진다 → belief 단독이 시스템을 이기는 왜곡. LLM 이 만든 **현실적으로 퍼진** 분포로 t=0 배치 (§thor_prior_llm) |
| `--move data/thor_move.json` | Qwen 생성 3종 | 배회·이동이 **전부 균등 난수**였던 것을 고침(§67). ① 방 체류(거실 0.40·화장실 0.15) ② 물체별 이동성향(CellPhone 0.95·Pillow 0.30) ③ **이동 목적지**(머그컵은 부엌에 놓이지만 거실에서 발견 — 배치와 다르다). 목적지가 균등이면 belief 가 원리적으로 이동 물체를 못 맞혀 그 몫(38%)이 왜곡된다 |
| `--min-rooms 4 --max-rooms 8`<br>`--max-nonbath 6 --min-nonbath 2` | 한국 주거 | ProcTHOR 에 방 10개+ 집이 섞여 있고 그런 집이 들어가면 국소화 천장이 무너진다(0.897→0.621). "침실 4개 중 어느 침실"은 우리 시나리오가 아니다. 한국 아파트 = 방 3 + 거실 + 주방 ≈ 5 (§20주택) |
| `--dwell 600` | 방당 평균 10분 | 사람이 한 방에 머무는 시간. 짧으면 배회가 순간이동처럼 되어 방 인지·재방문 판정이 비현실적으로 쉬워진다 |
| `--vis-dist 20` | 가시성 20 m | **기본 1.5 m 는 "보인다"를 거리로 잘라버린다** — 물체당 보인 프레임이 6장 대 122장으로 갈렸다. 이 값이 빠지면 GT 가 오염되고 모든 결론이 무효 (§58) |
| `--mapwalk` | 초기 맵 촬영 | 순간이동 정지샷이 아니라 **사람이 집을 둘러보듯 걷는 연속 궤적**(0.25 m 걸음 + 방마다 360° 스캔). mono-depth·SfM 사슬이 걷는 영상의 시차를 전제하므로 정지샷으로는 초기맵을 못 만든다 |
| `--hours 4 --moves 40` | 지평·이동 밀도 | 1시간·8건에서는 "떠난 방을 다시 본다"가 40%뿐이라 경우2(부재→belief) 표본이 16건에 그쳤다. 3시간에서 재방문 0.69 로 개선 — **4시간·40건은 그 추세의 연장**이며 이 장비에서 처음 가능한 규모 |
| `data/thor_queries.json` | Qwen 확장질의 | 물체별 표현 4종("alarm clock" / "digital alarm clock" / "round white clock" / "clock on nightstand"). 글자 질의를 한 표현으로만 하면 검출이 표현 운에 좌우된다. **정적 물체(앵커)에 특히 유효** — 잘 안 움직이는 것들은 확장질의가 도움이 되고 잘 움직이는 것은 오히려 방해(§확장질의 실험). `thor2_owl.py --queries` 로 사용 |
| (자동) 타입 단일 타겟 | 조건① | 오검출의 74%가 유사종 혼동이고 **같은 타입 다른 인스턴스는 0%**. "똑같이 생긴 물건 여럿"은 우리 시나리오가 아니라 평가에서 제외한다 |

**이동 물체는 집에 타입이 하나뿐인 것으로** 뽑히도록 해야 before/부재 문항이 생긴다
(ep1 에서 books×2 라 before 문항이 0개였다 — `docs/NEWSIM_EPISODES_20260826.md`).

### Qwen 3.5 9B 산출물 3종 — repo 에 포함, 재생성 불필요

| 파일 | 생성 스크립트 | 내용 |
|---|---|---|
| `data/thor_prior.json` | `thor_prior_llm.py` | P(방 유형 \| 물체 유형) — t=0 배치 |
| `data/thor_move.json` | `thor_move_llm.py` | 방 체류 · 이동성향 · **이동 목적지**(배치와 다름) |
| `data/thor_queries.json` | `thor_queryexp.py` | 물체별 확장질의 4종 |

⚠️ 재생성하면 **이전 판과 비교가 끊긴다.** 새로 만들 일이 있으면(어휘 확장 등)
그 사실을 결과에 명시하고, 가능하면 기존 판과 나란히 재라. 재생성 시 llama-server
필요: `~/khronos-llm/run_server.sh 9b 8080` (모델 gguf ~5.7GB, HF 재다운 가능).
⚠️ Qwen 3.5 는 thinking 모델이라 `enable_thinking=False` 필수 — 안 그러면 추론에
토큰을 다 쓰고 답이 잘린다.

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

**판정 완료 (2026-08-27)**: 384 → 0.621 · **768 → 0.944** · 1024 → 0.946.
**768 로 확정**(1024 는 포화, 렌더비만 증가). 아래 §2 의 `--size` 를 768 로.

## 2. 대량 생성 (판정 후, 반나절)

해상도 R 이 통과하면:

```bash
$PY -u scripts/thor_gen2.py --houses 100 --hours 4 --moves 40 \
  --min-rooms 4 --max-rooms 8 --max-nonbath 6 --min-nonbath 2 \
  --size 768 --dwell 600 --seed 901 --vis-dist 20 --platform cloud --mapwalk \
  --prior data/thor_prior.json --move data/thor_move.json --out data/thor7
# ↓ 캐시·harvest 는 손으로 하지 말고 **자동 체인을 걸어두라** (생성 도중에 걸어도 된다):
#   nohup bash scripts/rtx7_finish.sh > finish.log 2>&1 &
#   → gt.json 100개(완주)를 감지하면 캐시 4종 → harvest_t7.tar.gz 까지 알아서 돈다.
#   아래 수동 명령은 참고용 (자동 체인과 같은 내용).
THOR_ROOT=data/thor7 CACHE_PREFIX=/tmp/t7_a_ $KX -u scripts/exp_anchowl.py 4
THOR_ROOT=data/thor7 QCACHE_PREFIX=/tmp/t7_q_ STRIDE=4 $KX -u scripts/exp_imgq.py
THOR_ROOT=data/thor7 ACACHE_PREFIX=/tmp/t7_x_ STRIDE=4 $KX -u scripts/exp_anchor_exemplar.py
THOR_ROOT=data/thor7 CLIP_PREFIX=/tmp/t7_c_ STRIDE=4 $KX -u scripts/exp_clip_rooms.py
$KX scripts/harvest.py --root data/thor7 --cache /tmp --prefix t7_ \
  --out harvest_t7.tar.gz --keep-geom
# 검증기 채점 — 자동 체인의 5단계. 프레임이 서버에만 있어 **여기서만 가능**.
# 후보 실크롭 2AFC 로짓 전부 기록(문턱은 로컬 스윕, §89). 크롭 상자 W/3
# = 해상도 실험과 동일 기하라 768 운용점(0.944)이 그대로 이식된다.
THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ QC_PREFIX=/tmp/t7_q_ \
  OUT_JSONL=t1_scores_t7.jsonl $KX -u scripts/exp_t1_verify_pipeline.py
```

**역할 분담** — 이 장비: 생성 + 캐시 + 검증기 채점(프레임·GPU 의존 전부).
로컬 기준 머신(M1 Max): 문턱 스윕 + 최종 채점(eval_answerable 등 — numpy 판
편차 4%p 때문에 최종 수치는 기준 머신에서만 낸다).

가져올 것 3개: `harvest_t7.tar.gz` · `t1_scores_t7.jsonl` ·
`res768_scores.jsonl`(1단계 산출물 — 문턱 스윕의 캘리브레이션 기준).

가져올 것: `res*_scores.jsonl`(1단계) → 판정 후 `harvest_t7.tar.gz`(수백 MB).
**프레임은 서버에 둔다** — 우리 분석은 전부 캐시 위에서 돈다.

### 2.5 중간 점검 (생성 완주 전, ~65채 시점부터 가능)

완주를 기다리지 않고 지금 완료된 채만으로 조기 판독판을 뽑을 수 있다:

```bash
HOUSES=60 PREFIX=t7p_ nohup bash scripts/rtx7_finish.sh > finish_p.log 2>&1 &
```

같은 스크립트다 — `HOUSES` 가 현재 완료 수 이하면 대기 없이 즉시 발화하고,
완료된 채(gt.json 있는 것)만 심볼릭링크 뷰로 모아 캐시→`harvest_t7p.tar.gz` 를
만든다. **접두어 `t7p_` 를 본판 `t7_` 와 절대 섞지 말 것** — exp_anchowl 의
어휘(vocab)가 채 집합에서 유도되므로 부분판·완판 캐시는 호환되지 않는다.
부분판은 조기 판독용이고, 완주 후 본판 체인(t7_)이 따로 돈다. 둘을 동시에
걸어도 접두어가 다르니 안전하다(함정 #5 는 **같은** 접두어 얘기다).
GPU 는 생성과 공유 — 렌더가 약간 느려질 수 있으나 이 장비에선 허용 범위.

## 함정 (전부 실측으로 물린 것들)

1. `--vis-dist 20` 없으면 GT 오염 — 물체당 보인 프레임 6장 대 122장 (§58)
2. `renderInstanceSegmentation`·`renderDepthImage` 는 스크립트가 켠다. 직접 부를 땐 확인
3. 사전확률을 넣었으면 **생성 데이터에서 분포를 되재라** — `scripts/eval_movecheck.py` (§67)
4. 원격 실행은 **점수만 기록하고 문턱은 로컬 스윕** — 문턱은 도메인·모델마다 밀린다 (§89)
5. 같은 캐시 접두어를 두 프로세스에 주지 말 것 — "있으면 건너뛰기"는 동시성 안전이 아니다
