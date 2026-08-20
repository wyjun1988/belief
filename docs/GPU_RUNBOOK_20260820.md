# 4090 · H100 실행 런북

**최종 갱신 2026-08-20** — SceneDiff 355쌍 전수 결과 반영판.

⚠️ **2절의 355곳 표는 잘못된 조건이다**(집을 안 걸렀다). 반드시 `--group home` 으로
다시 잴 것 — 자세한 것은 2절의 경고를 볼 것.

---

## 0. 지금 상황 한 장

우리 시스템은 층별로 이렇게 서 있다.

| 층 | 상태 | 수치 |
|---|---|---|
| 사건 검색 | **작동** | hit@5 **0.75** (SuperMemory) |
| 부재 판정 — **장소를 알 때** | **작동** | AUC **0.655** (p=0.010) · 조건 조이면 0.764 |
| **장소 찾기** | **막힘** ← 전 층의 병목 | 1등비율 **42~51%** (355곳 중) |
| 저장 예산 | 작동 | ≈50 MB / 12h |
| 갱신 예산 | 조건부 | OWLv2 전수 불가(12h 기록 → 24.5h) |

**장소를 못 찾는 것 하나가 전부를 막고 있다.** 부재 판정은 장소를 손으로 짚어주면
되고(0.655), 못 짚어주면 우연으로 떨어진다. 그래서 이 런북의 90%는 장소 찾기다.

---

## 1. 환경 (한 번만)

```bash
git clone https://github.com/wyjun1988/belief.git khronos && cd khronos
python3 -m venv .venv && . .venv/bin/activate
pip install -U torch torchvision transformers pillow numpy scipy scikit-learn \
                pycocotools opencv-python-headless
```

**torch 2.6 이상**이면 맥에서 겪은 우회가 전부 불필요하다. 맥의 벽 3가지(참고):

| 벽 | 맥 증상 | GPU |
|---|---|---|
| MPS **Conv3D 미지원** | V-JEPA2 가 CPU 로만 돌아 4프레임밖에 못 넣었다 | 해결 |
| MPS `upsample_bicubic2d` 미지원 | DINOv2 가 죽는다 | 해결 |
| OWLv2 속도 | **2,046 ms/프레임** → 12h 기록에 24.5h | 10~20배 기대 |

데이터(11.9 GB · MIT · 게이트 없음):

```bash
mkdir -p data/scenediff && cd data/scenediff
wget https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip
unzip -q scenediff_benchmark.zip && cd ../..
```

⚠️ 데이터셋 README 의 URL 은 **파일명 오타로 404** 다
(`scenediff_bechmark.zip` → `scenediff_benchmark.zip`). 위가 맞다.

---

## 2. 맥에서 나온 기준선 — **반드시 전수로 비교할 것**

`place_repr_bench.py` · SceneDiff **355쌍 전수** · MBP(torch 2.8 · MPS).

측정하는 세 거리: **①** 같은 클립 안(같은 장소·시점만 다름) · **②** 같은 장소 v1↔v2 ·
**③** 남의 장소 최고. **여백 ②−③** 이 양수여야 장소를 찾고, **시점여백 ①−③** 이
양수여야 각도에 견딘다.

| 표현 | ①시점 | ②같은장소 | ③남의장소 | 여백②−③ | 시점①−③ | 1등비율 |
|---|---|---|---|---|---|---|
| **dino_vlad** (AnyLoc) | 0.415 | 0.485 | 0.467 | **+0.023** | −0.091 | **51%** |
| clip_cls | 0.910 | 0.931 | 0.923 | +0.000 | −0.025 | 49% |
| clip_g2 | 0.926 | 0.946 | 0.947 | −0.004 | −0.026 | 46% |
| vjepa2 ⚠️ | (nan) | 0.952 | 0.964 | −0.014 | (nan) | 30% |
| dino_cls | 0.645 | 0.706 | 0.724 | −0.016 | −0.102 | 45% |
| dino_g2 | 0.739 | 0.784 | 0.804 | −0.018 | −0.084 | 44% |
| ijepa | 0.803 | 0.847 | 0.864 | −0.020 | −0.071 | 42% |

### ⚠️ 위 표는 **잘못된 조건**에서 잰 것이다 — 반드시 `--group home` 을 쓸 것

위 355곳 표는 **서로 다른 집·상점·사무실을 한 풀에** 넣고 "전 세계 355곳 중에서
이 자리를 찾아라" 를 시킨 것이다. **실사용은 그렇지 않다.** 우리 설계 2-1 절에
**공간 1차 필터링**이 있다 — "지금 어느 씬그래프(집/사무실) 안인가부터 거른다".
Nymeria 에서도 `graph_uid` 하나로 11세션이 **정합 절차 없이** 한 지도가 된다.

집을 먼저 거르면 후보는 **그 집의 자리 몇십 개**지 355곳이 아니다. 노트북을 찾는데
남의 집 거실까지 후보에 넣고 "안 된다" 고 한 셈이다.

SD-K 는 **참가자가 곧 한 부엌**이라 그대로 묶을 수 있다:

| 참가자 | 쌍 수 (= 한 부엌 안의 장면) |
|---|---|
| P09 | 55 |
| P01 | 40 |
| P02 | 30 |
| P04 | 30 |

```bash
python scripts/place_repr_bench.py --root data/scenediff/scenediff_benchmark/data \
  --device cuda --group home --reps clip_cls,dino_cls,dino_vlad,owlvec
```

`--group home` 은 ③(남의 장소)을 **같은 집 안으로만** 제한한다.

### 규모 효과는 그래도 실재한다

| 표현 | 44곳 | 355곳 |
|---|---|---|
| dino_cls | +0.043 | **−0.016** |
| clip_cls | +0.017 | **+0.000** |

후보가 늘면 ③이 커져 여백이 무너지는 것 자체는 사실이다(검색 풀이 3,106 → 10,891
프레임일 때 hit@5 가 0.75 → 0.30 으로 무너진 것과 같은 현상). **다만 "그러니 355곳이
현실" 은 틀렸다** — 현실 규모는 **집 하나 안의 자리 수**이고, 그것을 정하는 것이
공간 1차 필터링이다. 축소판이 낙관적인 것과, 잘못된 규모로 재는 것은 다른 문제다.

### 읽어야 할 두 줄

1. **시점여백이 7종 전부 음수다.** 같은 장소를 각도만 바꿔 보면 **남의 장소보다도
   멀어진다.** 표현을 바꿔서 풀리는 문제가 아니라는 강한 신호다.
2. **1등비율 42~51%** — 355곳 중 하나를 고르는데 절반은 틀린다.

---

## J1. 장소 표현 — **최우선** (4090 · 2~4시간)

### J1-a AnyLoc 계열 스윕 ← **1순위**

전수에서 **유일하게 양수**(+0.023)인 유일한 표현이다. 백본과 VLAD 어휘 수를 키운다.

```bash
python scripts/place_repr_bench.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --nframes 8 \
  --reps dino_vlad,dino_cls --dino facebook/dinov2-large --vlad-k 64 \
  | tee out_j1a_k64.log

python scripts/place_repr_bench.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --nframes 8 \
  --reps dino_vlad --dino facebook/dinov2-large --vlad-k 128 \
  | tee out_j1a_k128.log
```

**판정**: 여백이 +0.023 에서 뚜렷이 오르는가, **그리고 시점여백이 양수로 뒤집히는가.**
후자가 핵심이다 — 여백만 오르고 시점여백이 음수면 각도 바뀔 때 계속 틀린다.

### J1-b V-JEPA2 **공정 재측정** — 이전 결과는 무효

맥에서는 MPS 가 Conv3D 를 지원하지 않아 CPU 로 내렸고, 그 제약 때문에 **절반당
4프레임**만 넣었다(64프레임 학습 모델). 시점여백이 nan 으로 나온 것도 그 탓이다.

```bash
python scripts/place_repr_bench.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --reps vjepa2 --nframes 64 --vj-frames 32 \
  --vjepa facebook/vjepa2-vitl-fpc64-256 | tee out_j1b_vjepa32.log
```

H100 이면 더 큰 것도:

```bash
python scripts/place_repr_bench.py --root data/scenediff/scenediff_benchmark/data \
  --device cuda --reps vjepa2 --nframes 64 --vj-frames 64 \
  --vjepa facebook/vjepa2-vitg-fpc64-384 | tee out_j1b_vjepa_g.log
```

### J1-c 큰 백본 + 결합

```bash
python scripts/place_repr_bench.py \
  --root data/scenediff/scenediff_benchmark/data --device cuda --nframes 8 \
  --clip openai/clip-vit-large-patch14 --dino facebook/dinov2-large \
  --reps clip_cls,clip_g2,dino_cls,dino_g2,dino_vlad,ijepa | tee out_j1c_large.log
```

CLIP 은 1등비율 49% 인데 여백이 0, DINO 는 여백은 있는데 1등비율 45% — **상보적**이라
결합이 유효할 수 있다(현재 스크립트에 결합 항목은 없다. 두 로그의 표를 보고 판단).

---

## J2. 물체 조합 벡터 — 비용 0 경로 (4090 · 1~2시간)

**왜**: 시각 임베딩 7종이 전부 실패했는데, **우리가 장소 식별에 성공한 적이 있는
유일한 방식**이 물체 조합이다 — ADT 방 서명(TF-IDF 물체 조합 투표) **0.971** 로
GT 사전(0.966)과 동률이었다.

두 가지 이점:
- **추가 비용 0** — OWL 검출 메타는 우리가 이미 저장하는 것이다
- **시점에 원리적으로 둔감** — 부엌은 어느 각도에서 봐도 같은 물체 집합이다.
  지금 전 계열을 무너뜨린 것이 정확히 시점여백이므로 여기가 급소다

```bash
python scripts/place_repr_bench.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --nframes 6 --reps owlvec,owlvec_idf,clip_cls | tee out_j2_owlvec.log
```

⚠️ 어휘 1,037단어를 조각내 `owlnet(**inp)` 를 반복하면 **이미지 인코더가 조각 수만큼**
다시 돈다. 현재 스크립트는 **텍스트 임베딩을 1회만** 만들고 프레임당 이미지 인코더를
**1회**만 돌린다(MBP `owl_fast.py` 에서 원 경로 대비 최대오차 **0.00e+00** 로 검증).
이 경로를 건드리지 말 것.

**판정**: 여백과 시점여백이 시각 임베딩(+0.023 / −0.091)을 넘는가.
`owlvec_idf`(어디에나 뜨는 wall·floor 비중을 낮춘 판)와도 비교한다.

---

## J3. 부재 지표 전수 + 비용 실측 (4090 · 2~3시간)

**왜**: 맥에서는 다운로드 중 앞부분 **33쌍**으로만 쟀다. 그 결과가
**AUC 0.655 (p=0.010)**, 조건②를 조이면 **0.764 (p=0.007)** 였다. 전수로 확인한다.

```bash
python scripts/scenediff_absence.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --nframes 10 --out rows_all.json | tee out_j3_absence.log
```

이 지표의 조건 두 가지(㉓ 단계 진단에서 나온 것):
- **조건①** 같은 이름의 다른 개체가 없어야 한다(인스턴스 유일성)
- **조건②** 있을 때 실제로 검출됐어야 한다 — 조일수록 AUC 가 오른다

같은 자리에서 **비용도 재둔다**(예산표의 GPU 열):

```bash
python - <<'PY' | tee out_j3_bench.log
import time, torch
from PIL import Image
from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                          CLIPImageProcessor, CLIPVisionModelWithProjection)
dev="cuda"; ims=[Image.new("RGB",(1024,1024),(120,130,140))]*8
cp=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
cn=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
op=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
on=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
def bench(n,f,rep=10):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(rep): f()
    torch.cuda.synchronize(); d=(time.time()-t)/rep
    print("%-22s %8.1f ms → 12시간(43,200프레임) %5.1f시간"%(n,d*1000,d*43200/3600))
with torch.no_grad():
    bench("CLIP 배치8", lambda: cn(**cp(images=ims,return_tensors="pt").to(dev)))
    w=["object %d"%i for i in range(64)]
    bench("OWLv2 어휘64", lambda: on(**op(text=[w],images=ims[0],return_tensors="pt").to(dev)))
PY
```

맥 실측 대조군: CLIP 배치8 **19.1 ms/프레임**(12h → 1.8h) · OWLv2 **2,046 ms/프레임**
(12h → 24.5h). 어휘 16개나 64개나 차이 없다 — 비용이 **이미지 인코더**에 있다.

**판정**: OWLv2 가 12시간 기록을 **하룻밤(8시간) 안에** 처리하는가. 못 하면
키프레임 선별이 설계상 필수다.

---

## J4. 타임라인 부재 판정 — J1·J2 승자가 나온 뒤에만

**왜**: 부재를 "두 영상 비교" 가 아니라 **하나의 타임라인에서** 찾게 한 조건.

    ① 있음 → 다른방 → **없음** → 다른방      정답 = **없음**
    ② 있음 → 다른방 → 다른방                 정답 = **있음**

②는 재방문이 없으니 **마지막으로 본 상태**로 답한다(기권이 아니다).
다른 방은 함정이 아니라 과제의 일부다 — 아무 프레임이나 집어오지 않는지 보는 장치다.

⚠️ **현재 구현에 알려진 결함이 있다.** 장소 게이트를 퍼센타일(상위 20%)로 걸어서
항상 타임라인 **맨 끝**(방해 구간)을 고른다 — 장소 적중률이 pmi·latent 모두 정확히
**0.00** 으로 나온 원인이다. **절대 문턱으로 바꿔야** 의미 있는 측정이 된다
(앵커 자기 유사도 분포에서 문턱을 잡으면 GT 없이 교정 가능).

→ **J1·J2 에서 여백·시점여백이 뚜렷한 표현이 나온 뒤에 이 수정을 하고 돌릴 것.**
지금 돌리면 표현이 아니라 게이트 결함을 재게 된다.

---

## 하지 않기로 한 것 (근거 있는 철회)

| 항목 | 철회 근거 |
|---|---|
| **JEPA 파인튜닝** | **설계상**: JEPA 는 가린 영역의 표현을 문맥에서 **채워 넣도록** 학습한다 → 문맥으로 유추 안 되는 개별 디테일을 버린다. 부재 판정은 정확히 그 버려지는 쪽이라, 물건이 없어진 장면을 **메워서** 있던 것처럼 표현할 유인이 있다. **실측**: I-JEPA 시점여백 **−0.056**(44곳) / **−0.071**(355곳) 로 ViT 계열 최하위 — 공정한 조건에서 나온 값이다 |
| **물체 마스킹 앵커** | 마스크가 가린 패치 비율이 **중앙 0.03**(196패치 중 6개). 마스크판과 패치평균이 **소수점까지 동일**(여백 0.025 · 1등 64%). 물체가 프레임 임베딩을 거의 안 흔들어 **고칠 편향 자체가 없다** |
| 부재 문맥 정규화 | 인스턴스·이동구간별 채점에서 원 drop 0.492 vs 정규화 0.449 |
| 검색층 OWLv2 재순위 | 4B·9B 모두 악화(9B 재검 0.47→0.40) |

**대체 방향**: 장소 찾기는 **판별 학습**이 맞는 계열이다 — 같은 장소 다른 시점을 양성,
다른 장소를 음성으로 놓고 그 거리를 직접 줄인다. 우리가 못 넘는 지표(시점여백)를
그대로 목적함수로 삼는 것이다. J1-a(무학습 AnyLoc)가 실패하면 그때 학습으로 간다.
학습 쌍은 **ADT·Nymeria 가 포즈를 갖고 있으므로 자동 생성 가능**하다(위치가 가까운
두 구간 = 같은 장소 다른 시점).

---

## 역할 분리 — 한 표현에 둘 다 요구하지 않는다

| 역할 | 필요한 성질 | 담당 |
|---|---|---|
| 장소 찾기 | 시점 불변 · **개별 물체 무시** | 장소 인식(판별) · 물체 조합 |
| 부재 판정 | **개별 물체 민감** | OWLv2 검출 |

두 성질이 정반대다. JEPA 가 어긋난 것도, 마스킹이 헛돈 것도 이 지점을 안 가른 탓이다.

---

## 결과 보내는 법

`out_*.log` 를 그대로 주면 된다. 각 스크립트가 판정 기준까지 찍는다.

**가장 중요한 한 줄**: J1·J2 로그 맨 아래 **"여백 순위"** 와, 표의 **시점여백 열**.
시점여백이 양수로 뒤집힌 표현이 하나라도 나오면 전체가 연결된다 — 그게 없으면
판별 학습으로 넘어가야 한다.
