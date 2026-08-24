#!/bin/bash
# **빌린 GPU 에서 렌더링이 되는지 1분 안에 확인한다.**
#
# H100·A100 같은 데이터센터 카드는 Vulkan 을 지원하긴 하나, 클라우드 이미지가
# 그래픽을 안 켜주는 경우가 흔하다(컨테이너 기본 NVIDIA_DRIVER_CAPABILITIES 가
# compute,utility 라 그래픽 라이브러리가 안 붙는다. MIG/vGPU 면 아예 막힌다).
# 실패는 이분법적이다 — 되거나 아예 안 되거나. 그러니 본 실행 전에 여기서 가른다.
#
# 사용:  bash scripts/gpu_preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
ok=0; bad=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✅ $1"; ok=$((ok+1)); else echo "  ❌ $1"; bad=$((bad+1)); fi; }

echo "=== ① 컴퓨트 ==="
$PY -c "import torch;print('  torch',torch.__version__,'· cuda',torch.cuda.is_available(),
  '·',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 없음')" 2>&1 | tail -2
chk "torch.cuda 사용 가능" "$PY -c 'import torch;assert torch.cuda.is_available()'"

echo "=== ② 그래픽 (렌더링의 전제) ==="
echo "  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<미설정>}"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | grep -m2 -E "deviceName|driverName" | sed 's/^/  /' || true
  chk "vulkaninfo 가 장치를 보고함" "vulkaninfo --summary 2>/dev/null | grep -q deviceName"
else
  echo "  ❌ vulkaninfo 없음 → apt install -y vulkan-tools libvulkan1"
  bad=$((bad+1))
fi

echo "=== ③ 실제 렌더 (이게 진짜 시험이다) ==="
$PY - <<'PYEOF' 2>&1 | tail -8
import time, sys
try:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    import prior
except Exception as e:
    print("  ❌ import 실패:", e); sys.exit(1)
try:
    ds = prior.load_dataset("procthor-10k")["train"]
    t = time.time()
    c = Controller(scene=ds[0], width=384, height=384, quality="Low",
                   platform=CloudRendering, visibilityDistance=20.0,
                   renderInstanceSegmentation=True)
    boot = time.time() - t
    e = c.step("Pass")
    assert e.frame is not None and e.frame.shape[0] == 384, "프레임이 안 나온다"
    assert e.instance_detections2D is not None, "instance_detections2D 없음"
    n = 30; t = time.time()
    for _ in range(n):
        c.step("RotateRight")
    fps = n / (time.time() - t)
    print("  ✅ CloudRendering 동작 · 기동 %.0fs · **%.1f fps**" % (boot, fps))
    print("     60채×3시간(648k 프레임) 예상 %.1f시간" % (648000 / fps / 3600))
    c.stop()
except Exception as e:
    print("  ❌ 렌더 실패:", type(e).__name__, str(e)[:200])
    print("     → 컨테이너면 NVIDIA_DRIVER_CAPABILITIES=all 로 다시 띄울 것")
    print("     → 그래도 안 되면 이 장비로는 생성 불가. 추론(캐시)만 여기서 하고")
    print("        생성은 4090 급 소비자 GPU 에서 한다.")
    sys.exit(1)
PYEOF
rc=$?

echo "=== 판정 ==="
if [ $rc -eq 0 ] && [ $bad -eq 0 ]; then
  echo "  전부 통과 — bash scripts/gpu_4090_run.sh 실행 가능"
elif [ $rc -eq 0 ]; then
  echo "  렌더는 되나 점검 $bad 건 실패 — 진행해도 되지만 로그를 지켜볼 것"
else
  echo "  ❌ 렌더 불가. 이 장비로는 [1/4] 생성을 못 한다."
  echo "     추론만 쓰려면: 생성은 다른 장비에서, 프레임을 여기로 옮긴 뒤"
  echo "     THOR_ROOT=... scripts/exp_anchowl.py / exp_imgq.py 만 실행."
fi
exit $rc
