#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/out"
PROGRAM="${ROOT_DIR}/noma/self_growing_xor.noma"
CSV_PATH="${OUT_DIR}/noma_loss.csv"
RAW_LOG="${OUT_DIR}/noma_raw.log"

mkdir -p "${OUT_DIR}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "[run_noma] cargo not found. Install Rust (rustup) first." >&2
    exit 1
fi

python3 - "${PROGRAM}" "${CSV_PATH}" "${RAW_LOG}" <<'PY'
import csv
import re
import subprocess
import sys
from pathlib import Path
program_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
raw_log_path = Path(sys.argv[3])

cmd = ["cargo", "run", "--release", "--", "run", str(program_path)]
print(f"[run_noma] running: {' '.join(cmd)}")

with raw_log_path.open("w", encoding="utf-8") as raw_f, csv_path.open("w", newline="", encoding="utf-8") as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow(["step", "loss", "hidden"])

    # Stream subprocess output line by line
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    step = 0
    hidden = 2
    growth_done = False
    # Match integers or floats, optionally with scientific notation
    loss_pattern = re.compile(r"(?<!\d)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

    for raw_line in proc.stdout:  # type: ignore
        raw_line = raw_line.rstrip("\n")
        raw_f.write(raw_line + "\n")

        if not raw_line:
            continue

        # Explicitly detect the growth: after phase1 max iterations warning, hidden becomes 16
        sentinel = raw_line.strip()
        if "Optimize loop reached max iterations" in sentinel:
            if not growth_done:
                print("GROWTH TRIGGERED: hidden 2 -> 16")
                hidden = 16
                growth_done = True
            continue

        # Parse standard NOMA print lines like "[print] <number>"
        if raw_line.startswith("[print] "):
            num_str = raw_line[len("[print] "):].strip()
            m = loss_pattern.fullmatch(num_str)
            if m:
                try:
                    loss_val = float(m.group(1))
                except ValueError:
                    print(raw_line)
                    continue
                print(f"step={step} loss={loss_val:.4f} hidden={hidden}")
                writer.writerow([step, f"{loss_val:.6f}", hidden])
                step += 1
            else:
                print(raw_line)
            continue

        # Treat as loss only if the entire line is a numeric literal
        m = loss_pattern.fullmatch(raw_line.strip())
        if m:
            try:
                loss_val = float(m.group(1))
            except ValueError:
                print(raw_line)
                continue
            print(f"step={step} loss={loss_val:.4f} hidden={hidden}")
            writer.writerow([step, f"{loss_val:.6f}", hidden])
            step += 1
        else:
            print(raw_line)

    return_code = proc.wait()

if return_code != 0:
    sys.exit(return_code)
PY
