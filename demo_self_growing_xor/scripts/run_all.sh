#!/bin/bash
#
# Master script: run complete benchmark suite and generate all outputs.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  XOR Benchmark Suite - Complete Run"
echo "========================================"
echo ""

# Clean previous outputs
"$SCRIPT_DIR/clean_out.sh"

# Run benchmark
cd "$ROOT_DIR"
python3 python/benchmark.py "$@"

# Generate plots
python3 python/plot_loss.py
python3 python/summarize.py

# Count lines of code
"$SCRIPT_DIR/count_loc.sh"

echo ""
echo "========================================"
echo "  Complete! Outputs in out/"
echo "========================================"
echo ""
ls -la "$ROOT_DIR/out/"
