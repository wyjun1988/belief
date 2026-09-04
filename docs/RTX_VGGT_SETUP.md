# VGGT 재국소화 — 설치·실행·문제해결 (RTX PRO 6000)

우리 구조: **지도 1회 + live 프레임별 등록**. 1fps live 는 프레임 간 겹침이 없어 추적이 불가능하므로,
매핑워크를 한 번에 재구성하고 live 는 [지도 앵커 + 묶음]으로 독립 등록한다. COLMAP 이 HSSD 20채 중 14채에서
무너진 원인(지도 접힘)을 피하려는 것이 목적이다.

## 1. 설치 (한 번만)

```bash
# 우리 venv 안에서. torch 는 CUDA 판이 이미 있어야 한다 (python -c "import torch; print(torch.cuda.is_available())" → True)
pip install "git+https://github.com/facebookresearch/vggt"
pip install "huggingface_hub>=0.24" transformers safetensors    # 이미 있으면 생략
```
가중치는 첫 실행에서 자동으로 받는다(`facebook/VGGT-1B`, 약 5GB → `~/.cache/huggingface`).
사내망에서 막히면 미리:
```bash
huggingface-cli download facebook/VGGT-1B
```
확인:
```bash
python - <<'PY'
import torch; from vggt.models.vggt import VGGT
m = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
print("OK", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
PY
```

## 2. 실행 (채마다 2단계)

**2-1. VGGT 로 원시 포즈 뽑기** (GPU. 지도 1회 + live 묶음들)
```bash
python scripts/vggt_reloc.py <house_dir> \
  --out /mnt/ssd2/wooyeol/work/khcache/vggt/raw_<house>.jsonl \
  --live-step 4 --chunk 8 --anchors 8 --map-max 96
```
- `--live-step 4` : live 4장마다 1장 (5~7.6천 프레임 에피소드용). 1,200~2,000 프레임이면 1 로.
- `--chunk 8 --anchors 8` : 한 번에 16장을 푼다. 96GB 면 `--chunk 16 --anchors 12` 까지 올려도 된다(빠름).
- `--map-max 96` : 지도 프레임 상한. 매핑워크가 128~184장이면 균등 표본한다. 메모리가 남으면 144 로.
- 출력은 `{"name","c","f"}` jsonl (미터). 척도는 **DA-V2 metric 깊이 / VGGT 깊이 비율 중앙값**으로 자동(§146 원리, GT 불필요).

**2-2. 정렬·평가는 COLMAP 판과 같은 코드로** (CPU, 수십 초)
```bash
python scripts/sfm_reloc.py <house_dir> --from-poses /mnt/.../raw_<house>.jsonl \
  --scale da --align sites \
  --out /mnt/ssd2/wooyeol/work/khcache/vggt/pose_<house>.jsonl
```
`--from-poses` 는 COLMAP 을 통째로 건너뛰고 정렬(방 라벨)·평가·요약만 한다. 화면에 찍히는 줄이 곧 보고 내용이다:
```
[   Ns] 라벨 정렬: 지점 A/B 이 제 방 폴리곤 안 · live 카메라 폴리곤 안 C/D · yaw ...
[   Ns] live 커버리지 0.xx · ATE 중앙 x.xxm 평균 ... · yaw 중앙 x.x° · 카메라방 적중 0.xx
```
요약 json 은 `~/khcache/vggt/<house>/summary_<house>.json`.

**2-3. 4채 묶어서**
```bash
for h in <og4>/house_000{0,1,2,3}; do hn=$(basename $h)
  python scripts/vggt_reloc.py $h --out ~/khcache/vggt/raw_$hn.jsonl --live-step 4
  python scripts/sfm_reloc.py $h --from-poses ~/khcache/vggt/raw_$hn.jsonl --scale da --align sites \
    --work ~/khcache/vggt/$hn --out ~/khcache/vggt/pose_$hn.jsonl
done
cat ~/khcache/vggt/pose_house_*.jsonl > ~/khcache/vggt/pose_og4_vggt.jsonl
```

## 3. 평가 체인에 넣기
```bash
SKIP_GEN=1 SCORER=hf CALIB=1 POSE_JSONL=~/khcache/vggt/pose_og4_vggt.jsonl \
  OUT=<og4> BENCH_DIR=~/khcache/bench-og4-vggt bash scripts/bench_v2_chain.sh
```
사다리 첫 줄에 `위치:SfM · 포즈:SfM` 이 찍히면 정상(외부 포즈도 같은 칸으로 표기된다).

## 4. 보고 형식 (COLMAP 기준선과 나란히)
| 채 | 방법 | 지도프레임 | live 커버리지 | ATE 중앙 | <0.5m | yaw 중앙 | 카메라방 | 채당 시간 |
그리고 **모듈별 성능표 + 3+1 전체표** 두 가지(docs/EVAL_PROTOCOL_V2.md 형식).

## 5. 판정
실패 예상 지점은 **질감 많은 집의 지도 접힘**이다. HSSD 20채에서 COLMAP 은 6채만 정렬됐다.
VGGT 가 같은 4채에서 *지점 정렬(라벨 정렬 줄의 A/B)* 을 0.5 이상으로 올리면 채택하고 20채로 간다.

## 6. 문제해결
- **CUDA OOM**: `--chunk 4 --anchors 6`, 그래도면 `--map-max 64`, `--res 392`.
- **`pose_encoding_to_extri_intri` ImportError**: VGGT 버전 차이. `from vggt.utils.pose_enc import pose_encoding_to_extri_intri`
  경로가 바뀌었으면 리포지토리의 `demo_*.py` 에서 쓰는 이름으로 바꿔 달라(그 한 줄만 고치면 된다).
- **깊이 키가 없다(`pred["depth"]` KeyError)**: 그 빌드가 깊이를 안 내면 척도가 1.0 으로 남는다 →
  `--from-poses` 단계에서 척도가 틀어지므로, 알려 달라(대안: 앵커 정렬을 sim3 로 바꿔 척도까지 풀 수 있다).
- **묶음 채택률이 낮다**(마지막 줄 `묶음 채택 n/N`): 앵커 정렬 rms 가 커서 기권한 것. `--anchors 12 --rms-max 0.5` 로 완화.
- **`--from-poses` 에서 `라벨 정렬` 지점 수가 0**: gt.json 에 `scene_meta.polys`(방 폴리곤)와 `map[i].room`(지점 라벨)이
  있어야 한다. OG 생성기 출력에 이 둘이 들어가는지 먼저 확인할 것.
- **속도**: 96GB 기준 지도 96장 1회 + live 묶음 (1,900/8=238회) ≈ 10~20분/채. 이보다 훨씬 느리면 `--res 392`.

## 7. 우선순위 (변경 없음)
VGGT > CPU COLMAP 기준선 완주 > MASt3R-SLAM > CUT3R. GPU0 는 계속 미접촉.

## 8. 158/158 기권 사례 (2026-09-04) — 원인과 조치
증상: `묶음 채택 0/158`. **문턱이 엄격해서가 아니다.** 두 가지를 고쳤다(푸시됨).
1. **입력 순서**: `vggt.utils.load_fn.load_and_preprocess_images` 는 버전에 따라 경로를 **내부 정렬**한다. 그러면
   `[앵커 8장, live 8장]` 으로 넣은 순서가 깨져 앵커 행이 엉뚱한 프레임이 되고 정렬이 100% 실패한다.
   → 한 장씩 전처리해 쌓도록 바꿨다(`VGGT_ORDER_SAFE=0` 으로 옛 동작 복원 가능).
2. **앵커 검색**: 32×32 회색조 기술자만으로는 렌더에서 엉뚱한 방을 고를 수 있다. 이제 **직전 묶음이 성공한 위치
   근처의 지도 프레임**을 절반 섞는다(1fps 라도 사람은 순간이동하지 않는다). 첫 묶음만 기술자 단독.
3. 진단이 추가됐다: 첫 묶음의 앵커 잔차(m)·척도·정규화 rms 와, 끝에 rms 10/50/90 분위. 다시 전부 기권하면
   **그 세 줄을 그대로 보내 달라** — rms 중앙값이 1 이상이면 검색 불량, 0.4~0.9 면 `--rms-max` 조정으로 해결된다.

재실행(가중치는 이미 받았으니 그대로):
```bash
git pull
python scripts/vggt_reloc.py <house_0000> --out ~/khcache/vggt/raw_house_0000.jsonl \
  --live-step 4 --anchors 12 --rms-max 0.6
```

## 9. rms 1.0 초과 (2026-09-04 2차) — 앵커가 서로 안 겹친다
분위수 1.069 / 1.338 / 1.548 은 "문턱이 빡빡하다"가 아니라 **앵커 8장이 서로 정합되지 않았다**는 뜻이다.
원인: 기술자 상위 8장은 집 안 여기저기에 **흩어져 있어 서로 시야가 겹치지 않는다**. VGGT 는 한 묶음을 **함께** 푸는
모델이라, 서로 안 겹치는 이미지들은 상대 위치를 정할 근거가 없다. 지도 통과(60장 연속 경로)에서는 프레임이 사슬로
이어져 있어 잘 풀렸던 것이다.

**조치(푸시됨): `--anchor-mode window`(기본)** — 매핑워크는 순서가 곧 경로이므로 **연속 구간 8장**을 앵커로 쓴다.
구간 중심은 첫 묶음에선 기술자 최상위, 이후엔 직전 묶음 위치의 최근접 지도 프레임.

**먼저 자가검사 한 번**(1분, 원인 확정):
```bash
git pull
python scripts/vggt_reloc.py <house_0000> --selfcheck --live-step 999 --out /tmp/sc.jsonl
```
세 줄이 찍힌다:
```
자가검사 연속 8장:     잔차 중앙 …m · 정규화 rms …
자가검사 연속 8장(중간): …
자가검사 흩어진 8장:   …
```
- **연속이 작고(≲0.2) 흩어진 것이 크면** → 원인 확정, 아래 본 실행으로 진행.
- **셋 다 크면** → VGGT 좌표 규약이나 재현성 문제이니 그 세 줄을 보내 달라(내가 extrinsic 해석을 고친다).

본 실행:
```bash
python scripts/vggt_reloc.py <house_0000> --out ~/khcache/vggt/raw_house_0000.jsonl \
  --live-step 4 --anchor-mode window --anchors 8 --rms-max 0.5
```
그래도 채택이 0 이면 `--anchors 12 --chunk 4` 로 (앵커 구간을 넓히고 묶음을 줄여 앵커 비중을 키운다).

## 10. 연속 앵커도 rms>1 (2026-09-04 3차) — 전처리가 원인이었다
자가검사에서 **연속 앵커도** rms>1 → 앵커 선택 문제가 아니라는 진단이 맞다. 원인은 내 코드의 순서 보장 방식이었다.

`load_and_preprocess_images` 는 **묶음 전체를 공통 크기로 리사이즈**한다. 9절에서 순서를 지키려고 한 장씩 불렀더니
프레임마다 크기·주점이 달라져 기하가 통째로 깨졌다(지도 통과도 함께 오염 → 두 통과가 서로 안 맞음).
→ **임시 디렉터리에 `%04d_` 접두사 심볼릭 링크**를 만들어, 로더가 정렬해도 내 순서와 같게 하고 **전처리는 묶음으로** 되돌렸다.

재현성 자가검사도 추가했다(동일 입력 재실행):
```bash
git pull
python scripts/vggt_reloc.py <house_0000> --selfcheck --live-step 999 --out /tmp/sc.jsonl
```
네 줄이 찍힌다. **첫 줄 `자가검사 동일 N장 재실행` 의 정규화 rms 가 0 에 가까워야 정상**이다.
- 첫 줄이 0 에 가깝고 "연속 8장"도 작으면 → 본 실행으로.
- **첫 줄이 크면** 모델이 같은 입력에 다른 결과를 낸다는 뜻이니 그 줄을 보내 달라(그때는 extrinsic 해석을 뜯는다).

본 실행은 9절과 동일:
```bash
python scripts/vggt_reloc.py <house> --out ~/khcache/vggt/raw_<h>.jsonl --live-step 4 --anchor-mode window --anchors 8 --rms-max 0.5
```

## 11. r4 회신 (2026-09-04) — **VGGT 는 잠시 보류, COLMAP 기준선 먼저**
r4 결과(재현성 0.0001 정상 · live 채택 0 · rms 중앙 1.444)는 우리 쪽 M2 재현과 **정확히 일치**한다. 같은 코드로 또 돌리면
같은 값이 나올 뿐이니 **VGGT 실행은 잠시 멈춰 달라.** 원인 가설(아래)을 우리가 로컬에서 검증 중이고, 확정되면 설정을
확정해 한 번에 보낸다.

가설: 우리가 지도 프레임을 `--map-max` 로 **균등 표본**해 왔다. 128장을 20장으로 줄이면 "연속 앵커 8장"이 원본 기준으로는
7장씩 건너뛴 것이라 서로 겹치지 않는다. VGGT 는 묶음을 함께 푸는 모델이라 안 겹치는 이미지들의 상대 위치를 정할 수 없다.
→ **지도를 표본 없이 전부** 넣으면(96GB 면 128~184장 가능) 앵커가 원본 기준 진짜 인접 프레임이 되어 해결될 것으로 본다.
검증 중(map-max 48 vs 96 의 자가검사 rms 비교). 결과가 나오면 12절에 확정 설정을 적는다.

### 지금 할 것: COLMAP 기준선 표 채우기 (GPU 불필요, 채당 1분)
완료된 house_0000·0001·0003 에 대해 **2-2 단계**를 돌려 주면 VGGT 없이도 비교표 첫 행들이 채워진다.
```bash
for h in <og4>/house_0000 <og4>/house_0001 <og4>/house_0003; do hn=$(basename $h)
  python scripts/sfm_reloc.py $h --from-poses <그 채의 COLMAP 원시 포즈 jsonl> \
    --scale da --align sites --work ~/khcache/colmap/$hn --out ~/khcache/colmap/pose_$hn.jsonl
done
```
※ CPU COLMAP 을 우리 `sfm_reloc.py` 로 돌렸다면 `--from-poses` 없이 그대로 결과가 나와 있을 것이다. 그 경우
`~/khcache/<...>/summary_<house>.json` 세 개를 보내 주면 된다. 필요한 값: `cov · ate_med · ate_lt05 · yaw_med · room_hit · sim3_inl · sec`.
보고는 §2 표 형식(채별 한 줄) 그대로.

## 12. COLMAP 기준선 3채 해석 (2026-09-04) — 숫자 읽는 법과 결론

받은 표를 그대로 옮기면:
| 채 | coverage | ATE 중앙 | <0.5m | yaw 중앙 | 카메라방 | sim3 inlier | 시간 |
|---|---|---|---|---|---|---|---|
| 0000 | 0.19 | 2.49m | 0.05 | 82.1° | 0.44 | 0.22 | 5,657s |
| 0001 | 0.18 | 4.14m | 0.00 | 101.4° | 0.33 | 0.14 | 12,960s |
| 0003 | 0.02 | 0.11m | 0.98 | 70.3° | 0.56 | 1.00 | 1,208s |

**(1) coverage 는 원본 프레임 수 기준이다.** `--live-step 4` 면 상한이 0.25 다. 즉 0.19 는 "표본의 76% 등록"이고
0.18 은 72%, 0.02 는 **8%**(house_0003 만 등록이 거의 안 됨). 혼동을 없애려고 `live 표본 등록 N/M(비율)` 줄을 따로
찍도록 고쳤다(푸시됨) — 다음 실행부터 두 줄이 나온다.

**(2) 진짜 문제는 yaw 70~101°.** 이건 HSSD 20채에서 14채가 무너진 것과 **같은 증상**(지도 접힘 → 어떤 강체 변환으로도
지점을 제 방에 못 넣음)이다. house_0003 만 sim3 inlier 1.00 인데, 등록된 프레임이 8% 뿐이라 "적게 등록했지만 그건 정확"한
경우다. → **OG 에서도 CPU COLMAP 은 기준선 이상의 의미가 없다**는 것이 3채로 확인됐다. §2-d(피드포워드) 판단이 옳았다.

**(3) 시간.** 채당 1,208~12,960초. house_0002 가 아직 도는 것도 이 때문이다. 끝나면 그 한 줄만 추가해 주고,
**COLMAP 은 여기서 종료**해도 된다(20채 확대 불필요).

**결론: 다음 라운드는 VGGT 하나로 간다.** 우리 로컬(M2)에서 원인 검증이 끝나는 대로 확정 설정을 13절에 적는다.
그때까지 GPU1 은 유휴로 두거나, 원하면 §0-b 의 20채 생성(1,200~2,000 프레임)을 미리 시작해도 좋다 — 어차피 필요하다.

## 13. 확정 설정 (2026-09-04) — 단일 전역 통과
로컬(M2) 검증 결론 둘: (a) 동일 입력 재현성은 완벽(rms 0.0000)이고 (b) **작은 묶음을 따로 풀어 이어 붙이는 방식 자체가
안 된다** — 8장짜리 부분집합은 전체 재구성과 다른 기하를 낸다(연속 앵커도 rms 0.38~1.09). 그리고 조밀 매핑워크(0-e)는
live 등록엔 필요하지만 접힘은 못 고친다(§156 2차). 따라서 **VGGT 는 배치 재구성기답게 한 번에 푼다**:

```bash
git pull
python scripts/vggt_reloc.py <house> --out ~/khcache/vggt/raw_<h>.jsonl \
  --map-max 400 --global-live-step 16 --chunk 8 --anchors 8 --rms-max 0.5
```
- `--map-max 400`: 지도 프레임을 **표본 없이 전부**(0-e 재생성 후 250~500장).
- `--global-live-step 16`: live 16장마다 1장(5,000장 에피소드면 ~310장)을 **지도와 같은 통과**에 넣는다 → 그 프레임들의
  포즈는 지도 좌표계에서 바로 확정된다(이어 붙이기 없음). 96GB 면 지도 400 + live 310 ≈ 700장은 `--res 392` 로 시도,
  OOM 이면 `--global-live-step 32` 또는 `--map-max 250`.
- 나머지 live 는 묶음마다 **시간상 이웃인 전역-통과 live 프레임**을 앵커로 쓴다(지점 회전 앵커와 달리 서로 겹친다).
보고: 끝줄 `지도 N · live M/전체 (묶음 채택 k/K)` 와 `앵커 정렬 rms 분위`, 그리고 2-2 정렬·평가 결과 줄. **채택이 0 이면 전역
통과의 live 표본만이라도 `--from-poses` 로 평가**하면 된다(표본 포즈만으로도 커버리지 1/16 의 정직한 표가 나온다).
