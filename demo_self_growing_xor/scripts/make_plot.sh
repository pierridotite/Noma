#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_PATH="${ROOT_DIR}/out/noma_loss.csv"
PNG_PATH="${ROOT_DIR}/out/loss.png"
PLOTTER="${ROOT_DIR}/python/plot_loss.py"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[make_plot] missing ${CSV_PATH}. Run scripts/run_noma.sh first." >&2
  exit 1
fi

python3 "${PLOTTER}" "${CSV_PATH}" "${PNG_PATH}"
echo "[make_plot] wrote ${PNG_PATH}"
