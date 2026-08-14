#!/usr/bin/env bash
# RunPod(3090/A6000, Ubuntu 24.04) 위에 DAAAM + Khronos + Hydra 스택을 세운다.
#
#   scp scripts/runpod_bootstrap.sh pod:/workspace/ && ssh pod 'bash /workspace/runpod_bootstrap.sh'
#
# 설계 원칙 두 가지:
#   1. **모든 산출물은 /workspace 아래.** 컨테이너 루트(/)는 57GB 뿐이고 팟을 반납하면 사라진다.
#      네트워크 볼륨에 두면 3090 → A6000 으로 갈아탈 때 빌드를 재사용할 수 있다(둘 다 sm_86).
#   2. **SSH 키를 팟에 올리지 않는다.** packages.yaml 이 git@github.com: 를 쓰지만 17개 저장소가
#      전부 공개이므로 git 의 insteadOf 로 https 로 갈아끼운다. 자격증명은 로컬에 남는다.
#
# 단계별로 마커를 남기므로 끊겨도 다시 실행하면 이어서 간다.
# -u 는 쓰지 않는다: ROS2 의 setup.bash 가 AMENT_TRACE_SETUP_FILES 를 미정의 상태로
# 참조해서 source 하는 순간 죽는다.
set -eo pipefail

WS=/workspace/ros2_ws
MARK=/workspace/.kx_stage
export DEBIAN_FRONTEND=noninteractive
export PIP_BREAK_SYSTEM_PACKAGES=1          # Ubuntu 24.04 PEP668. ROS2 는 시스템 python3.12 를 쓴다
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/hf
export TORCH_HOME=/workspace/.cache/torch
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" /workspace
touch "$MARK"

done_stage() { grep -qx "$1" "$MARK"; }
mark()       { echo "$1" >> "$MARK"; echo "=== stage done: $1 ==="; }

# --- 1. ROS 2 Jazzy -----------------------------------------------------------
if ! done_stage ros2; then
  echo "=== [1/5] ROS 2 Jazzy ==="
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    curl gnupg lsb-release locales software-properties-common ca-certificates
  locale-gen en_US.UTF-8 || true
  add-apt-repository -y universe

  # 키 로테이션에 안 물리도록 공식 apt-source 패키지를 먼저 시도한다.
  CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
  if ! curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/latest/download/ros2-apt-source_${CODENAME}_all.deb" \
      || ! dpkg -i /tmp/ros2-apt-source.deb; then
    echo "  (apt-source deb 실패 — 원시 키로 폴백)"
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${CODENAME} main" > /etc/apt/sources.list.d/ros2.list
  fi

  apt-get update -qq
  apt-get install -y ros-jazzy-ros-base ros-dev-tools python3-rosdep \
                     python3-colcon-common-extensions python3-vcstool
  mark ros2
fi
source /opt/ros/jazzy/setup.bash

# --- 2. rosdep ----------------------------------------------------------------
if ! done_stage rosdep; then
  echo "=== [2/5] rosdep ==="
  rosdep init 2>/dev/null || true
  rosdep update --rosdistro=jazzy
  mark rosdep
fi

# --- 3. 소스 받기 (https 강제) -------------------------------------------------
if ! done_stage src; then
  echo "=== [3/5] DAAAM 소스 ==="
  git config --global url."https://github.com/".insteadOf "git@github.com:"
  mkdir -p "$WS/src"
  [ -d "$WS/src/daaam" ] || git clone https://github.com/MIT-SPARK/DAAAM.git "$WS/src/daaam"
  mark src
fi

# --- 4. 빌드 (여기가 2~4시간) ---------------------------------------------------
if ! done_stage build; then
  echo "=== [4/5] colcon build (GTSAM 포함, 오래 걸린다) ==="
  cd "$WS/src"
  bash daaam/install/install.sh
  mark build
fi

# --- 4.5 파이썬 부가 의존 ------------------------------------------------------
# ⚠️ pip 는 /usr/local/lib/.../dist-packages 에 깔린다 — **네트워크 볼륨 밖**이라
# 팟을 내리면 사라진다. 볼륨에 남는 ros2_ws 와 달리 매번 다시 깔아야 한다.
# open_clip 은 세그 트랙의 외형 임베딩(ReID·제로샷 카테고리)에 쓴다.
if ! done_stage pydeps; then
    pip install -q --break-system-packages open_clip_torch || \
        pip install -q open_clip_torch
    python3 -c 'import open_clip; print("open_clip", open_clip.__version__)'
    mark pydeps
fi

# --- 5. 확인 -------------------------------------------------------------------
source "$WS/install/setup.bash"
echo "=== [5/5] 확인 ==="
python3 - <<'PY'
import importlib, sys
for m in ["torch", "hydra_python", "spark_dsg", "daaam", "rclpy"]:
    try:
        mod = importlib.import_module(m)
        print("  OK  %-14s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        print("  --  %-14s %s: %s" % (m, type(e).__name__, e))
import torch
print("  cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "KX_BOOTSTRAP_DONE"
