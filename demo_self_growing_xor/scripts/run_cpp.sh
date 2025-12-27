#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/out"
CPP_SRC="${ROOT_DIR}/cpp/baseline_xor.cpp"
CPP_BIN="${ROOT_DIR}/cpp/baseline_xor"
CSV_PATH="${OUT_DIR}/cpp_loss.csv"

mkdir -p "${OUT_DIR}"

echo "[run_cpp] compiling ${CPP_SRC}..."
g++ -O3 -std=c++17 -o "${CPP_BIN}" "${CPP_SRC}"

echo "[run_cpp] running ${CPP_BIN}..."
"${CPP_BIN}" "${CSV_PATH}"
