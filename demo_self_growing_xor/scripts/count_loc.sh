#!/bin/bash
#
# Count lines of code for each implementation.
# Rules:
#   - No blank lines
#   - No comment-only lines
#   - No closing braces alone
#   - No imports/includes
#   - No type declarations without logic
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$ROOT_DIR/out"

mkdir -p "$OUT_DIR"

count_python() {
    local file=$1
    grep -v '^\s*$' "$file" | \
    grep -v '^\s*#' | \
    grep -v '^\s*import ' | \
    grep -v '^\s*from .* import' | \
    grep -v '^\s*"""' | \
    grep -v "^\s*'''" | \
    grep -v '^\s*pass\s*$' | \
    wc -l
}

count_cpp() {
    local file=$1
    grep -v '^\s*$' "$file" | \
    grep -v '^\s*//' | \
    grep -v '^\s*#include' | \
    grep -v '^\s*#pragma' | \
    grep -v '^\s*}\s*$' | \
    grep -v '^\s*{\s*$' | \
    grep -v '^\s*};' | \
    grep -v '^\s*else\s*{' | \
    wc -l
}

count_noma() {
    local file=$1
    grep -v '^\s*$' "$file" | \
    grep -v '^\s*//' | \
    grep -v '^\s*#' | \
    wc -l
}

echo "Counting lines of code..."
echo ""

# NOMA
noma_file="$ROOT_DIR/noma/xor.noma"
if [ -f "$noma_file" ]; then
    noma_loc=$(count_noma "$noma_file")
    echo "NOMA:         $noma_loc lines"
else
    noma_loc=0
    echo "NOMA:         (not found)"
fi

# NumPy manual
numpy_file="$ROOT_DIR/baselines/python_numpy_manual/xor.py"
if [ -f "$numpy_file" ]; then
    numpy_loc=$(count_python "$numpy_file")
    echo "NumPy:        $numpy_loc lines"
else
    numpy_loc=0
    echo "NumPy:        (not found)"
fi

# PyTorch eager
torch_eager_file="$ROOT_DIR/baselines/python_torch/xor_eager.py"
if [ -f "$torch_eager_file" ]; then
    torch_eager_loc=$(count_python "$torch_eager_file")
    echo "Torch Eager:  $torch_eager_loc lines"
else
    torch_eager_loc=0
    echo "Torch Eager:  (not found)"
fi

# PyTorch compile
torch_compile_file="$ROOT_DIR/baselines/python_torch/xor_compile.py"
if [ -f "$torch_compile_file" ]; then
    torch_compile_loc=$(count_python "$torch_compile_file")
    echo "Torch Compile: $torch_compile_loc lines"
else
    torch_compile_loc=0
    echo "Torch Compile: (not found)"
fi

# C++ manual
cpp_file="$ROOT_DIR/baselines/cpp_manual/xor.cpp"
if [ -f "$cpp_file" ]; then
    cpp_loc=$(count_cpp "$cpp_file")
    echo "C++:          $cpp_loc lines"
else
    cpp_loc=0
    echo "C++:          (not found)"
fi

# Write JSON output
cat > "$OUT_DIR/loc.json" << EOF
{
  "noma": {
    "file": "noma/xor.noma",
    "loc_total": $noma_loc
  },
  "numpy_manual": {
    "file": "baselines/python_numpy_manual/xor.py",
    "loc_total": $numpy_loc
  },
  "torch_eager": {
    "file": "baselines/python_torch/xor_eager.py",
    "loc_total": $torch_eager_loc
  },
  "torch_compile": {
    "file": "baselines/python_torch/xor_compile.py",
    "loc_total": $torch_compile_loc
  },
  "cpp_manual": {
    "file": "baselines/cpp_manual/xor.cpp",
    "loc_total": $cpp_loc
  }
}
EOF

echo ""
echo "Saved: $OUT_DIR/loc.json"
