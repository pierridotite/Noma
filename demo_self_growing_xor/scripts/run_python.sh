#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="${ROOT_DIR}/python"
OUT_DIR="${ROOT_DIR}/out"
BASELINE="${PYTHON_DIR}/baseline_xor.py"
REQS="${PYTHON_DIR}/requirements.txt"

mkdir -p "${OUT_DIR}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

if [[ -f "${REQS}" ]]; then
  ${PYTHON} -m pip install -r "${REQS}"
fi

${PYTHON} "${BASELINE}" "${OUT_DIR}/python_loss.csv"
