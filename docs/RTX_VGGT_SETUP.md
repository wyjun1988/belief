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
