#!/bin/bash
# Qwen3.5-4B 를 5500 XT GPU 로 — Vulkan(MoltenVK) 경유.
#
# ⚠️ llama.cpp 의 **Metal 백엔드는 Apple Silicon 전용**이다. Intel 맥 + 외장 AMD 는
# Vulkan 백엔드 + MoltenVK 가 유일한 GPU 경로다(2026-08-17 실측).
# ⚠️ DYLD_LIBRARY_PATH 에 우리 빌드를 **맨 앞**에 둬야 한다. brew 의 ggml 패키지가
# 먼저 잡히면 조용히 CPU(BLAS) 로 떨어진다 — 실측: 그때 생성 7.4 t/s, GPU 34.3 t/s.
# 실측(Q4_K_M, 4.21B): pp64 32.7→39.0 t/s · tg16 **7.4→34.3 t/s (4.6×)**
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/usr/local/bin:$PATH
export DYLD_LIBRARY_PATH="$ROOT/vendor/llama.cpp/build-vk/bin:/usr/local/lib"
export VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json
exec "$ROOT/vendor/llama.cpp/build-vk/bin/$1" "${@:2}"
