# CUT3R 재국소화 — 설치·실행·문제해결 (H100 또는 RTX PRO 6000 GPU1)

왜 CUT3R 인가: 영속 상태에 프레임을 하나씩 더해 **모든 프레임을 같은 월드 좌표계**로 낸다. 우리 구조(지도 1회 + live 프레임별
등록)와 원래 맞는 모델이라 VGGT 처럼 부분 재구성을 이어 붙일 필요가 없다. 1fps 무겹침 live 도 "상태에 추가"로 처리된다.

## 1. 설치
```bash
git clone https://github.com/CUT3R/CUT3R.git && cd CUT3R
conda create -y -n cut3r python=3.11 && conda activate cut3r
pip install torch torchvision            # CUDA 판
pip install -r requirements.txt
# 체크포인트 (리포지토리 README 의 다운로드 스크립트 또는 HF). 512 dpt 판을 쓴다:
bash scripts/download_model.sh 2>/dev/null || echo "README 의 링크로 cut3r_512_dpt_4_64.pth 를 받아 src/ 에 두라"
pip install transformers safetensors    # DA 척도 보정용
```
확인:
```bash
python - <<'PY'
import sys; sys.path.insert(0,"."); sys.path.insert(0,"src")
from dust3r.model import ARCroco3DStereo; from dust3r.inference import inference; from dust3r.utils.image import load_images
print("import OK")
PY
```
import 가 실패하면 `demo.py` 상단의 import 세 줄을 보고 `scripts/cut3r_reloc.py` 의 `_load_cut3r()` 만 맞추면 된다.

## 2. 실행 (채당 2단계, 우리 리포지토리 루트에서)
```bash
# 2-1 자가검사(1분): live 를 더 넣어도 지도 포즈가 안 흔들리는지
python scripts/cut3r_reloc.py <house> --cut3r-root /path/CUT3R --model-path /path/CUT3R/src/cut3r_512_dpt_4_64.pth --selfcheck
# 2-2 본 실행: 지도 전부 → live(4장마다 1장) 순서로 한 세션
python scripts/cut3r_reloc.py <house> --cut3r-root /path/CUT3R --model-path /path/CUT3R/src/cut3r_512_dpt_4_64.pth \
  --live-step 4 --out ~/khcache/cut3r/raw_<h>.jsonl
# 2-3 정렬·평가 (COLMAP·VGGT 와 같은 코드 → 같은 표)
python scripts/sfm_reloc.py <house> --from-poses ~/khcache/cut3r/raw_<h>.jsonl --scale da --align sites \
  --work ~/khcache/cut3r/<h> --out ~/khcache/cut3r/pose_<h>.jsonl
```
매핑워크는 **0-e 의 연속 보행판**(이동 프레임 포함)을 쓴다. 회전만 있는 지도는 어떤 모델도 못 세운다.

## 3. 보고
§2 표 형식(채별: 커버리지 · ATE 중앙 · <0.5m · yaw · 카메라방 · 지점정렬 · 채당 시간)에 **CUT3R 행**을 추가. 자가검사의
"이동 중앙 … m" 한 줄도 함께.

## 4. 문제해결
- **OOM**: `--size 224` 또는 `--live-step 8`, `--max-frames 1200`. 세션 길이에 메모리가 비례한다(80/96GB 면 512 로 1,500장 안팎).
- **결과 키 KeyError(`camera_pose`)**: 그 버전의 demo.py 가 쓰는 키 이름을 `_run_stream()` 의 두 줄에 넣는다.
- **자가검사 이동이 크다(>0.2 m)**: live 를 넣을 때 지도가 다시 쓰이는 것 — 리포지토리의 상태 고정 옵션(freeze / no-update)을 찾아
  지도 이후 프레임에 적용. 없으면 그 사실을 보고(그때는 "지도만 CUT3R, live 는 VGGT 전역 통과"로 조합한다).
- **DA 척도 비율이 1 에서 많이 벗어남(0.3 또는 3 이상)**: 정상 — 렌더 도메인. 자동으로 곱해지므로 조치 불필요.
- **속도**: 512 로 1,500장 세션은 GPU 에서 수 분. 훨씬 느리면 `--size 224`.

## 5. 어디서 돌리나
H100 이 비어 있으면 CUT3R 은 H100, VGGT 는 RTX GPU1 — **같은 4채, 같은 표**로 나란히. 둘 다 GPU0(BEHAVIOR)는 미접촉.
