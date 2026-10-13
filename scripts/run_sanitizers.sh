#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-cmake-build-sanitize}"

cmake -S . -B "$BUILD_DIR" \
  -DCHESS_BUILD_GUI=OFF \
  -DBUILD_TESTING=ON \
  -DCHESS_ENABLE_WARNINGS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build "$BUILD_DIR" --parallel

CHESS_TEST_QUICK=1 ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "Address/undefined sanitizer gate: PASS"
