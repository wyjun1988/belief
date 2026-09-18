# H100 에 필요한 데이터 (2026-09-18, 9-50 배치)

Google Drive `DriveSyncFiles/` 에서 **아래 8개를 `~/khcache/` 에 풀면** 됩니다. 이미 있는 건 건너뛰세요 — 배치가 알아서 확인합니다.

```bash
KH=~/khcache; mkdir -p $KH
for Z in lora_adopt_v2 lora_adopt_c2_0917 lora_adopt_hn lora_adopt_real lora_presence_v2 val_fixed adopt_infer_v2; do
  [ -f ~/$Z.zip ] && unzip -q -n ~/$Z.zip -d $KH/
done
mv $KH/lora_adopt_c2_0917 $KH/lora_adopt_c2 2>/dev/null   # 이름만 맞춰줍니다
ls -d $KH/lora_* $KH/val_fixed $KH/adopt_infer_v2
```

| 파일 | 크기 | 무엇 | 왜 필요 |
|---|---|---|---|
| `lora_adopt_v2.zip` | 155 MB | 채택 시뮬 HSSD 133채 | 모든 채택 판의 바탕 |
| `lora_adopt_c2_0917.zip` | 95 MB | 채택 시뮬 ②재촬영 69채 | 이동 후 표본 보강 |
| `lora_adopt_hn.zip` | 159 MB | 교차집 어려운음성 | `A2_hn` 이 재는 변수 |
| **`lora_adopt_real.zip`** | **72 MB** | **실사 IT3DEgo 48영상** | **`A3_real` 이 재는 변수** |
| `lora_presence_v2.zip` | 233 MB | 부재 시뮬 | `P0_presence`·합본 |
| **`val_fixed.zip`** | **68 MB** | **고정 검증셋 1,594행** | **★ 모든 판을 같은 잣대로 채점** |
| `adopt_infer_v2.zip` | 216 MB | 133채 실측 6,996건 | 3단계 추론(최종 판정 재료) |

## OmniGibson 학습셋 2개는 그 기계에서 만든 것입니다
`lora_adopt_og` · `lora_presence_og` 는 9-39/9-40 때 **H100/RTX 에서 직접 생성**해서 M2 에도 Drive 에도 없습니다.

- 그 기계에 `data/hssd_og` 가 남아 있으면 **배치가 알아서 다시 만듭니다**(0.2 단계).
- 없으면 **그냥 진행하세요.** 모든 판에서 똑같이 빠지므로 **판 사이 비교는 그대로 유효**합니다. 기준선 절대값만 조금 내려갑니다. 요약 파일에 그렇게 찍힙니다.
- `data/hssd_og` 자체가 필요하면 Drive 의 `omnigibson.zip`(73 MB)이 원본입니다.

## 모델 가중치
`hf_models_rtx.tar.gz`(2.9 GB)에 Qwen3.5-4B 가 들어 있습니다. 그 기계에 이미 있으면 불필요합니다.

## 확인 한 줄
```bash
bash scripts/h100_batch_20260918.sh 2>&1 | head -30
```
0단계에서 "풀림/⚠ 없음" 이 찍히고, `--val-data` 가 없다고 나오면 `git pull` 이 안 된 것입니다.
