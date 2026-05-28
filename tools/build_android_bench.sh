#!/usr/bin/env bash
# Builds llama-bench for Android ARM64 without SVE (NEON only).
#
# Why no SVE: Android manufacturers ship kernels without CONFIG_ARM64_SVE=y.
# This binary matches what FUTO Keyboard runs — enabling SVE would require
# a custom kernel build (e.g. LineageOS with CONFIG_ARM64_SVE=y), which is
# device-specific and non-trivial. A SVE-enabled build would be ~20-30% faster.
#
# Requirements: curl, unzip, cmake, ninja
# Output: tools/android/llama-bench-arm64-v8a-neon  (also cached in .cache/)
#
# Usage:
#   bash tools/build_android_bench.sh

set -euo pipefail

NDK_VER="r27c"
NDK_DIR="$HOME/.android-ndk-${NDK_VER}"
LLAMA_DIR="/tmp/llama_bench_build"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$OUT_DIR/android/llama-bench-arm64-v8a-neon"

# ── 1. Android NDK ────────────────────────────────────────────────────────────
if [ ! -d "$NDK_DIR" ]; then
  echo "=== Downloading Android NDK ${NDK_VER} (~300 MB) ==="
  TMP_ZIP=$(mktemp --suffix=.zip)
  curl -L "https://dl.google.com/android/repository/android-ndk-${NDK_VER}-linux.zip" \
       -o "$TMP_ZIP"
  unzip -q "$TMP_ZIP" -d "$HOME"
  mv "$HOME/android-ndk-${NDK_VER}" "$NDK_DIR"
  rm "$TMP_ZIP"
  echo "NDK installed: $NDK_DIR"
fi

TOOLCHAIN="$NDK_DIR/build/cmake/android.toolchain.cmake"
STRIP="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"

# ── 2. llama.cpp source ───────────────────────────────────────────────────────
if [ ! -d "$LLAMA_DIR" ]; then
  echo "=== Cloning llama.cpp ==="
  git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

# ── 3. Build: ARM64, NEON only, no SVE, no OpenMP ────────────────────────────
BUILD_DIR="$LLAMA_DIR/build-android-neon"
echo "=== Configuring (NEON only, SVE=OFF, OpenMP=OFF) ==="
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SVE=OFF \
  -DGGML_NEON=ON \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -GNinja

echo "=== Building ==="
ninja -C "$BUILD_DIR" llama-bench

echo "=== Stripping ==="
"$STRIP" "$BUILD_DIR/bin/llama-bench"

mkdir -p "$(dirname "$OUT")"
cp "$BUILD_DIR/bin/llama-bench" "$OUT"

# Also update .cache for 17_benchmark_device.py
mkdir -p "$(cd "$(dirname "$0")/.." && pwd)/.cache/llama_bench"
cp "$OUT" "$(cd "$(dirname "$0")/.." && pwd)/.cache/llama_bench/llama-bench-arm64-v8a-neon"

echo ""
echo "=== Done: $OUT ($(du -sh "$OUT" | cut -f1)) ==="
echo ""
echo "Note: This binary uses NEON only (no SVE)."
echo "A kernel with CONFIG_ARM64_SVE=y would give ~20-30% more performance,"
echo "but Android manufacturers disable SVE by default."
