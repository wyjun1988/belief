# 4090 · H100 실행 런북 — 2026-08-20

맥에서 막혔거나 너무 느린 것만 모았다. **위에서부터 순서대로** 돌리면 되고,
각 작업에 **판정 기준**을 달았다 — 기준을 넘으면 다음으로, 못 넘으면 거기서 멈춘다.

## 왜 GPU 로 넘기나 — 맥의 벽 3가지 (실측)

| 벽 | 증상 | GPU 에서 |
|---|---|---|
| **Conv3D 미지원(MPS)** | V-JEPA2 가 아예 안 돌아 CPU 로 내려야 했다 | 해결 |
| `upsample_bicubic2d` 미지원(MPS) | DINOv2 가 죽는다(폴백으로 우회 중) | 해결 |
| torchvision 없음 | `AutoVideoProcessor` 불가 → 전처리 수동 | 해결 |
| **OWLv2 속도** | **2,046 ms/프레임** → 12시간 기록 처리에 **24.5시간** | 10~20배 기대 |

참고 — 맥 실측(iMac · MPS): CLIP 배치8 **19.1 ms/프레임**(12h 기록 → 1.8h),
OWLv2 **2,046 ms/프레임**(12h 기록 → 24.5h). 어휘 16개나 64개나 차이 없다
(비용이 텍스트가 아니라 **이미지 인코더**에 있다).

## 0. 환경 (한 번만)

```bash
git clone git@github.com:<this-repo> khronos && cd khronos
python3 -m venv .venv && . .venv/bin/activate
pip install -U torch torchvision transformers pillow numpy scipy pycocotools ffmpeg-python
```

**torch 2.6 이상**이면 위 우회 3개가 전부 불필요하다. 맥의 `.venv-mps` 는
stock-v2 와 공유라 torch 2.2.2 로 묶여 있었을 뿐이다.

데이터(11.9 GB · MIT · 게이트 없음):

```bash
mkdir -p data/scenediff && cd data/scenediff
wget https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip
unzip -q scenediff_benchmark.zip && cd ../..
```

⚠️ 데이터셋 README 의 다운로드 URL 은 **파일명 오타로 404** 다
(`scenediff_bechmark.zip` → `scenediff_benchmark.zip`). 위 주소가 맞는 것이다.

---

## J1. 장소 표현 벤치 — **최우선** (4090 · 약 1~2시간)

**왜**: 지금 전 층을 막고 있는 것이 **장소 식별**이다. 맥 실측에서 CLIP 은
장소를 못 가른다 — 같은 장소 0.933인데 **남의 장소 최고가 0.903**(여백 **+0.017**),
게다가 같은 장소를 각도만 바꿔 본 것(0.909)이 **남의 장소보다 덜 닮는다**
(시점여백 **−0.004**). 장소를 못 찾으니 부재 판정이 우연으로 떨어진다.

```bash
python scripts/place_repr_bench.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --nframes 8 --vj-frames 8 \
  --clip openai/clip-vit-large-patch14 \
  --dino facebook/dinov2-large \
  --vjepa facebook/vjepa2-vitl-fpc64-256 \
  | tee out_repr_L.log
```

큰 백본도(H100 이면 특히) 같이:

```bash
python scripts/place_repr_bench.py --root data/scenediff/scenediff_benchmark/data \
  --device cuda --vjepa facebook/vjepa2-vitg-fpc64-384 --vj-frames 16 \
  --reps vjepa2,dino_cls,dino_g2 | tee out_repr_G.log
```

**보는 것**: 표현별 `여백 ②−③` 와 `시점여백 ①−③`, `1등비율`.
**판정**: 여백이 CLIP-B(+0.017)보다 뚜렷이 크고 **시점여백이 양수**인 표현이 있으면
→ J2 로. 전부 0 근처면 → J4(학습)로.

---

## J2. 타임라인 부재 판정 — J1 승자로 (4090 · 30분)

**왜**: 부재를 "두 영상 비교" 가 아니라 **하나의 타임라인에서** 찾게 한 조건.
① 있음→다른방→**없음**→다른방 (정답 없음) / ② 있음→다른방→다른방 (정답 있음).
②는 재방문이 없으니 **마지막으로 본 상태**로 답한다.

```bash
python scripts/scenediff_timeline.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --place latent --cache cache.npz | tee out_timeline.log
```

⚠️ **현재 구현에 알려진 결함이 있다.** 장소 게이트를 퍼센타일(상위 20%)로 걸어서
항상 타임라인 **맨 끝**(방해 구간)을 고른다 — 장소 적중률이 pmi·latent 모두
정확히 0.00 으로 나온 원인이다. J1 에서 여백이 큰 표현이 나오면 **절대 문턱**으로
바꿔야 한다(앵커 자기 유사도 분포에서 문턱을 잡으면 GT 없이 교정 가능).
**J1 결과를 받은 뒤 이 수정을 먼저 반영할 것.**

**판정**: 균형정확도가 0.50 을 유의하게 넘는가.

---

## J3. 부재 지표 전수 + 비용 실측 (4090 · 2~3시간)

**왜**: 맥에서는 다운로드 중 앞부분 **33쌍**으로만 쟀다. 전체 350쌍으로 재확인한다.
그 33쌍 결과는 **AUC 0.655 (p=0.010)**, 조건②를 조이면 **0.764 (p=0.007)** 였다.

```bash
python scripts/scenediff_absence.py \
  --root data/scenediff/scenediff_benchmark/data \
  --device cuda --nframes 10 --out rows_all.json | tee out_absence.log
```

같은 자리에서 **비용도 재둔다**(맥 표에 GPU 열을 채우기 위해):

```bash
python - <<'PY' | tee out_bench.log
import time, torch, numpy as np
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
bench("CLIP 배치8", lambda: cn(**cp(images=ims,return_tensors="pt").to(dev)))
w=["object %d"%i for i in range(64)]
bench("OWLv2 어휘64", lambda: on(**op(text=[w],images=ims[0],return_tensors="pt").to(dev)))
PY
```

**판정**: 전수 AUC 가 0.65 근처를 유지하는가. OWLv2 가 12시간 기록을
**하룻밤(8시간) 안에** 처리하는가 — 못 하면 키프레임 선별이 필수다.

---

## J4. (조건부) V-JEPA2 파인튜닝 — J1 이 실패했을 때만 (H100)

**목표 함수**: **시점이 바뀌어도, 물건 하나가 없어져도 같은 장소로 붙는 임베딩.**
JEPA 계열이 이것에 맞는 이유 — 가린 영역의 **표현**을 문맥에서 예측하므로
픽셀 복원과 달리 지엽적 물체 세부를 자연히 버린다.

학습 쌍 만드는 법(라벨 불필요):
- **양성**: 같은 장소 · 다른 시각/각도 — 한 세션 안에서 위치가 가까운 두 구간
  (ADT·Nymeria 는 포즈가 있으므로 바로 만들 수 있다. SceneDiff 는 v1↔v2 쌍)
- **음성**: 다른 장소
- 붕괴 방지는 우리 경험을 그대로: **VICReg(분산+공분산) + imputation 앵커**
  (home-jepa 에서 latent 붕괴를 이걸로 잡았다). 2-스테이지(사전학습 → 인코더 동결
  → 헤드)가 joint 대비 IC +31% 였던 것도 같이 쓴다.

⚠️ **SceneDiff 는 학습에 너무 작다**(350쌍 · 영상 3~19초 · 중앙 7.2초).
학습으로 가면 **HD-EPIC**(Aria 41시간 · 4.4M 프레임 · CC BY-NC)이나 우리가 이미 가진
Nymeria·ADT 로 쌍을 만들어야 한다.

---

## 결과 보내주는 법

`out_*.log` 파일만 그대로 주면 된다. 각 스크립트가 판정 기준까지 찍는다.
J1 의 **여백 순위 한 줄**이 가장 중요하다 — 그 한 줄로 J2·J4 중 어디로 갈지 갈린다.

---

# 갱신 (2026-08-20 · 맥 축소판 결과 반영)

## 맥에서 나온 표현 벤치 (SceneDiff 44곳 · 참고 기준선)

| 표현 | 여백②−③ | 시점①−③ | 1등비율 |
|---|---|---|---|
| **dino_cls** | **+0.043** | −0.011 | 59% |
| dino_g2 | +0.039 | −0.008 | 55% |
| dino_patch | +0.018 | −0.001 | 57% |
| clip_cls | +0.017 | −0.004 | **70%** |
| clip_g2 | +0.012 | −0.007 | 64% |
| **ijepa** | +0.009 | **−0.056** | 57% |
| clip_patch | +0.008 | −0.002 | 57% |
| ~~vjepa2~~ | ~~−0.002~~ | ~~−0.002~~ | ~~41%~~ |
| colorlay | −0.006 | −0.064 | 43% |
| tiny | −0.147 | −0.291 | 20% |

**세 줄 요약**
1. DINOv2 가 CLIP 의 2.5배지만 1등비율 59% — 41%는 남의 장소가 더 닮았다.
2. **10종 전부 시점여백 음수.** 각도만 바뀌면 남의 장소보다 멀어진다. 표현을
   바꿔서 풀리는 문제가 아니라는 강한 신호다.
3. ~~vjepa2~~ 는 **무효** — 64프레임 학습 모델에 CPU 제약으로 4프레임만 넣었다.

## J1 수정 — 우선순위를 바꾼다

### J1-a **AnyLoc 계열(DINOv2 + VLAD)** ← 새 1순위, 학습 불필요

**왜**: 우리 벤치 1위가 DINOv2 이고, AnyLoc 은 그 특징에 VLAD 집계를 얹어
**시점 강건 장소 인식**을 무학습으로 얻는 계열이다. 우리가 못 넘고 있는 지표
(시점여백)를 정면으로 겨냥한다.

집계만 바꾸면 되므로 `place_repr_bench.py` 에 `dino_vlad` 를 추가해 같은 표에서 비교한다
(DINOv2 패치 특징 → k-means 어휘 32~64 → VLAD → 정규화).

### J1-b V-JEPA2 **재측정** (이번 결과는 무효)

```bash
python scripts/place_repr_bench.py --root data/scenediff/scenediff_benchmark/data \
  --device cuda --reps vjepa2 --vj-frames 32 \
  --vjepa facebook/vjepa2-vitl-fpc64-256 | tee out_vjepa_fair.log
```

### J1-c DINOv2-large · CLIP-L, 그리고 **clip_cls + dino_cls 결합**

CLIP 은 1등비율 70% 인데 여백이 작고, DINO 는 반대다 — 상보적이라 결합이 유효할 수 있다.

## ⚠️ J4(JEPA 파인튜닝)는 **철회한다**

**이유(설계상)**: JEPA 는 가린 영역의 표현을 **문맥에서 채워 넣도록** 학습한다.
그러면 인코더는 문맥으로 유추 가능한 구조를 담고 **유추 안 되는 개별 디테일을 버린다.**
부재 판정은 정확히 그 버려지는 쪽이다 — 물건이 없어진 장면을 **메워서** 있던 것처럼
표현할 유인이 있다. 같은 이유로 "물체를 가린 앵커" 도 JEPA 류에서는 위험하다.

**이유(실측)**: I-JEPA 의 시점여백이 **−0.056** 으로 ViT 계열 중 최악이다.
이것은 V-JEPA2 와 달리 **공정한 조건**(정상 해상도·정상 입력)에서 나온 값이다.

**대체**: 장소 찾기는 **판별 학습**이 맞는 계열이다 — 같은 장소 다른 시점을 양성,
다른 장소를 음성으로 놓고 그 거리를 직접 줄인다. 우리가 못 넘는 지표를 그대로
목적함수로 삼는 것이다. 먼저 J1-a(무학습)로 확인하고, 부족하면 그때 학습한다.
학습 쌍은 **ADT·Nymeria 가 포즈를 갖고 있으므로 자동 생성 가능**하다
(위치가 가까운 두 구간 = 같은 장소 다른 시점).

## 역할 분리 — 하나의 표현으로 둘 다 하려 하지 않는다

| 역할 | 필요한 성질 | 담당 |
|---|---|---|
| 장소 찾기 | 시점 불변 · **개별 물체 무시** | 장소 인식(판별) |
| 부재 판정 | **개별 물체 민감** | OWLv2 검출 |

두 성질이 정반대라 한 인코더에 둘 다 요구하면 안 된다. JEPA 가 어긋난 것도
이 지점이다.
