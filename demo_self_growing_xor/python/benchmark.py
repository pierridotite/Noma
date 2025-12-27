#!/usr/bin/env python3
"""
Benchmark NOMA vs Python vs C++ execution time for the self-growing XOR demo.
Runs each multiple times and plots the comparison.
"""
import csv
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "out"
NOMA_PROGRAM = ROOT / "noma" / "self_growing_xor.noma"
PYTHON_BASELINE = ROOT / "python" / "baseline_xor.py"
CPP_BINARY = ROOT / "cpp" / "baseline_xor"
CPP_SOURCE = ROOT / "cpp" / "baseline_xor.cpp"

N_RUNS = 5  # Number of runs for averaging


def compile_cpp() -> bool:
    """Compile C++ baseline if needed."""
    if not CPP_SOURCE.exists():
        print("[benchmark] C++ source not found, skipping")
        return False
    result = subprocess.run(
        ["g++", "-O3", "-std=c++17", "-o", str(CPP_BINARY), str(CPP_SOURCE)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[benchmark] C++ compile error: {result.stderr}")
        return False
    return True


def time_noma() -> float:
    """Run NOMA program and return execution time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--", "run", str(NOMA_PROGRAM)],
        capture_output=True,
        text=True,
    )
    end = time.perf_counter()
    if result.returncode != 0:
        print(f"NOMA error: {result.stderr}")
    return end - start


def time_python() -> float:
    """Run Python baseline and return execution time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(
        ["python3", str(PYTHON_BASELINE), str(OUT_DIR / "python_loss_bench.csv")],
        capture_output=True,
        text=True,
    )
    end = time.perf_counter()
    if result.returncode != 0:
        print(f"Python error: {result.stderr}")
    return end - start


def time_cpp() -> float:
    """Run C++ baseline and return execution time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(
        [str(CPP_BINARY), str(OUT_DIR / "cpp_loss_bench.csv")],
        capture_output=True,
        text=True,
    )
    end = time.perf_counter()
    if result.returncode != 0:
        print(f"C++ error: {result.stderr}")
    return end - start


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Compile C++
    has_cpp = compile_cpp()
    
    print(f"[benchmark] Running {N_RUNS} iterations each...")
    
    # Warm-up runs
    print("[benchmark] Warm-up NOMA...")
    time_noma()
    print("[benchmark] Warm-up Python...")
    time_python()
    if has_cpp:
        print("[benchmark] Warm-up C++...")
        time_cpp()
    
    # Timed runs
    noma_times = []
    python_times = []
    cpp_times = []
    
    for i in range(N_RUNS):
        print(f"[benchmark] Run {i+1}/{N_RUNS}...")
        noma_times.append(time_noma())
        python_times.append(time_python())
        if has_cpp:
            cpp_times.append(time_cpp())
    
    noma_times = np.array(noma_times)
    python_times = np.array(python_times)
    cpp_times = np.array(cpp_times) if has_cpp else np.array([])
    
    # Stats
    noma_mean, noma_std = noma_times.mean(), noma_times.std()
    python_mean, python_std = python_times.mean(), python_times.std()
    cpp_mean = cpp_times.mean() if has_cpp else 0
    cpp_std = cpp_times.std() if has_cpp else 0
    
    print(f"\n[benchmark] Results ({N_RUNS} runs, 542 steps each):")
    print(f"  NOMA:   {noma_mean*1000:.1f}ms ± {noma_std*1000:.1f}ms")
    print(f"  Python: {python_mean*1000:.1f}ms ± {python_std*1000:.1f}ms")
    if has_cpp:
        print(f"  C++:    {cpp_mean*1000:.1f}ms ± {cpp_std*1000:.1f}ms")
    print(f"\n  NOMA vs Python: {python_mean/noma_mean:.2f}x faster")
    if has_cpp:
        print(f"  NOMA vs C++:    {cpp_mean/noma_mean:.2f}x {'faster' if cpp_mean > noma_mean else 'slower'}")
    
    # Save to CSV
    csv_path = OUT_DIR / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["run", "noma_time", "python_time"]
        if has_cpp:
            header.append("cpp_time")
        writer.writerow(header)
        for i in range(N_RUNS):
            row = [i, f"{noma_times[i]:.4f}", f"{python_times[i]:.4f}"]
            if has_cpp:
                row.append(f"{cpp_times[i]:.4f}")
            writer.writerow(row)
    print(f"[benchmark] Saved {csv_path}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart with error bars
    ax = axes[0]
    if has_cpp:
        x = ["NOMA", "C++", "Python"]
        means = [noma_mean * 1000, cpp_mean * 1000, python_mean * 1000]
        stds = [noma_std * 1000, cpp_std * 1000, python_std * 1000]
        colors = ["#3498db", "#9b59b6", "#e74c3c"]
    else:
        x = ["NOMA", "Python"]
        means = [noma_mean * 1000, python_mean * 1000]
        stds = [noma_std * 1000, python_std * 1000]
        colors = ["#3498db", "#e74c3c"]
    
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Time (milliseconds)")
    ax.set_title(f"Execution Time Comparison\n({N_RUNS} runs, 542 steps each)")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f"{mean:.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    # Individual runs scatter
    ax = axes[1]
    runs = np.arange(1, N_RUNS + 1)
    ax.plot(runs, noma_times * 1000, "o-", color="#3498db", label="NOMA", markersize=10, linewidth=2)
    ax.plot(runs, python_times * 1000, "s-", color="#e74c3c", label="Python", markersize=10, linewidth=2)
    if has_cpp:
        ax.plot(runs, cpp_times * 1000, "^-", color="#9b59b6", label="C++", markersize=10, linewidth=2)
    ax.set_xlabel("Run #")
    ax.set_ylabel("Time (milliseconds)")
    ax.set_title("Per-Run Execution Times")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(runs)
    
    fig.suptitle("Self-Growing XOR: NOMA vs Python vs C++ Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    png_path = OUT_DIR / "benchmark.png"
    plt.savefig(png_path, dpi=150)
    print(f"[benchmark] Saved {png_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
