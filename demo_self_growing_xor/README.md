# Self-Growing XOR (NOMA-first)

Self-growing XOR: model starts tiny, hits a plateau, grows the hidden layer via `realloc`, and keeps training without restart.

## Run (one-liner)

```bash
bash demo_self_growing_xor/scripts/run_noma.sh
```

Prereq: Rust toolchain available in `PATH` (cargo).

## What you should see
- Console lines like `step=120 loss=0.6931 hidden=2` followed by `GROWTH TRIGGERED: hidden 2 -> 16`
- A fresh [demo_self_growing_xor/out/noma_loss.csv](demo_self_growing_xor/out/noma_loss.csv)
- Optionally, a plot at [demo_self_growing_xor/out/loss.png](demo_self_growing_xor/out/loss.png)

Sample terminal excerpt:
```
[run_noma] running: cargo run --release -- run /.../self_growing_xor.noma
step=0 loss=0.2602 hidden=2
step=100 loss=0.1876 hidden=2
step=200 loss=0.1817 hidden=2
GROWTH TRIGGERED: hidden 2 -> 16
step=201 loss=0.1817 hidden=16
step=300 loss=0.0423 hidden=16
step=489 loss=0.0020 hidden=16
Result: 0.00198...
```

## Plot (optional but slick)

```bash
bash demo_self_growing_xor/scripts/make_plot.sh
```

## NOMA vs Python comparison

```bash
python3 demo_self_growing_xor/python/plot_comparison.py
```

Generates [demo_self_growing_xor/out/comparison.png](demo_self_growing_xor/out/comparison.png) showing:
- **NOMA**: `realloc` + continue training seamlessly
- **Python**: manual array resizing + Adam optimizer state **reset**

Both converge, but Python requires ~80 lines of plumbing code while NOMA does it in 2 lines.

## Performance benchmark

```bash
python3 demo_self_growing_xor/python/benchmark.py
```

Runs all 3 implementations 5 times and generates [demo_self_growing_xor/out/benchmark.png](demo_self_growing_xor/out/benchmark.png):

| Implementation | Time (542 steps) | Lines of code | Optimizer state on growth |
|----------------|------------------|---------------|---------------------------|
| **C++** | ~3ms | ~220 | ❌ Reset |
| **NOMA** | ~110ms | ~60 | ✅ Preserved |
| **Python** | ~185ms | ~100 | ❌ Reset |

**Key insight**: NOMA is ~1.7× faster than Python while being simpler. C++ is fastest but requires 4× more code and manual optimizer reset.

## C++ baseline

```bash
bash demo_self_growing_xor/scripts/run_cpp.sh
```

Writes [demo_self_growing_xor/out/cpp_loss.csv](demo_self_growing_xor/out/cpp_loss.csv). Shows the verbosity of implementing self-growth in C++ (~220 lines).

## Python baseline (pedagogical)

```bash
bash demo_self_growing_xor/scripts/run_python.sh
```
- Writes [demo_self_growing_xor/out/python_loss.csv](demo_self_growing_xor/out/python_loss.csv) using a hand-rolled NumPy MLP
- Growth resets optimizer state on purpose to highlight the plumbing cost versus NOMA

## File map
- [demo_self_growing_xor/noma/self_growing_xor.noma](demo_self_growing_xor/noma/self_growing_xor.noma): main demo (dynamic growth with optimizer state kept intact)
- [demo_self_growing_xor/scripts/run_noma.sh](demo_self_growing_xor/scripts/run_noma.sh): one-command runner + CSV logger
- [demo_self_growing_xor/scripts/make_plot.sh](demo_self_growing_xor/scripts/make_plot.sh): plot `out/loss.png`
- [demo_self_growing_xor/scripts/run_python.sh](demo_self_growing_xor/scripts/run_python.sh): optional NumPy baseline
- [demo_self_growing_xor/python/plot_loss.py](demo_self_growing_xor/python/plot_loss.py): draws loss curve + growth marker
- [demo_self_growing_xor/assets/expected_terminal_output.txt](demo_self_growing_xor/assets/expected_terminal_output.txt): reference log slice for filming

## Tips for filming
- Keep the terminal tall enough to show the plateau then the growth jump
- Run `bash demo_self_growing_xor/scripts/make_plot.sh` right after for a clean `loss.png`
- Delete `demo_self_growing_xor/out/*` between takes to show fresh outputs
