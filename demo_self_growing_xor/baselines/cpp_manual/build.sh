#!/bin/bash
set -e
cd "$(dirname "$0")/.."
g++ -O3 -std=c++17 -o xor xor.cpp
echo "Built: xor"
