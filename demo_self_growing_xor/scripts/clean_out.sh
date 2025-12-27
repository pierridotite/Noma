#!/bin/bash
#
# Clean all generated outputs.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$ROOT_DIR/out"

echo "Cleaning output directory..."

# Keep init/ but clean everything else
if [ -d "$OUT_DIR" ]; then
    find "$OUT_DIR" -maxdepth 1 -type f -delete
    for dir in "$OUT_DIR"/run_*/; do
        [ -d "$dir" ] && rm -rf "$dir"
    done
    echo "Cleaned: $OUT_DIR"
else
    echo "Nothing to clean."
fi
