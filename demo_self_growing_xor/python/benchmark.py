#!/usr/bin/env python3
"""
Benchmark orchestrator: runs all implementations N times, collects metrics,
produces summary.json and aggregated statistics.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median, quantiles
from typing import Any

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.json"
OUT_DIR = ROOT / "out"


def run_command(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def compile_noma(noma_file: Path, output: Path, cargo_dir: Path) -> float:
    """Compile NOMA to binary, return compile time in ms."""
    start = time.perf_counter()
    run_command(
        ["cargo", "run", "--release", "--quiet", "--", "build-exe", str(noma_file), "-o", str(output)],
        cwd=cargo_dir
    )
    return (time.perf_counter() - start) * 1000


def compile_cpp(source: Path, output: Path) -> float:
    """Compile C++ to binary, return compile time in ms."""
    start = time.perf_counter()
    run_command(["g++", "-O3", "-std=c++17", "-o", str(output), str(source)])
    return (time.perf_counter() - start) * 1000


def run_noma_compiled(binary: Path, run_dir: Path) -> dict[str, Any]:
    """Run NOMA compiled binary and parse output."""
    run_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.perf_counter()
    rc, stdout, stderr = run_command([str(binary)])
    total_ms = (time.perf_counter() - start) * 1000
    
    # Parse NOMA output (just prints loss values)
    losses = []
    for line in stdout.strip().split('\n'):
        try:
            losses.append(float(line))
        except ValueError:
            pass
    
    # Write stdout.txt
    with open(run_dir / "stdout.txt", "w") as f:
        f.write(stdout)
    
    # Generate loss.csv from output
    with open(run_dir / "loss.csv", "w") as f:
        f.write("step,loss,accuracy,hidden,phase\n")
        for i, loss in enumerate(losses):
            hidden = 2 if i < 201 else 16
            phase = 1 if i < 201 else 2
            acc = 1.0 if loss < 0.01 else (0.75 if loss < 0.1 else 0.5)
            f.write(f"{i},{loss:.6f},{acc:.2f},{hidden},{phase}\n")
    
    timings = {
        "total_ms": total_ms,
        "final_loss": losses[-1] if losses else 1.0,
        "iters_total": len(losses),
        "impl": "noma_compiled"
    }
    
    with open(run_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2)
    
    return timings


def run_noma_interpreted(noma_file: Path, run_dir: Path, cargo_dir: Path) -> dict[str, Any]:
    """Run NOMA in interpreter mode."""
    run_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.perf_counter()
    rc, stdout, stderr = run_command(
        ["cargo", "run", "--release", "--quiet", "--", "run", str(noma_file)],
        cwd=cargo_dir
    )
    total_ms = (time.perf_counter() - start) * 1000
    
    # Parse [print] lines
    losses = []
    for line in stdout.strip().split('\n'):
        if line.startswith("[print]"):
            try:
                val = float(line.replace("[print]", "").strip())
                if not losses or val != losses[-1]:
                    losses.append(val)
            except ValueError:
                pass
    
    with open(run_dir / "stdout.txt", "w") as f:
        f.write(stdout)
    
    with open(run_dir / "loss.csv", "w") as f:
        f.write("step,loss,accuracy,hidden,phase\n")
        for i, loss in enumerate(losses):
            hidden = 2 if i < 201 else 16
            phase = 1 if i < 201 else 2
            acc = 1.0 if loss < 0.01 else (0.75 if loss < 0.1 else 0.5)
            f.write(f"{i},{loss:.6f},{acc:.2f},{hidden},{phase}\n")
    
    timings = {
        "total_ms": total_ms,
        "final_loss": losses[-1] if losses else 1.0,
        "iters_total": len(losses),
        "impl": "noma_interpreted"
    }
    
    with open(run_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2)
    
    return timings


def run_python_impl(script: Path, run_dir: Path) -> dict[str, Any]:
    """Run a Python implementation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.perf_counter()
    rc, stdout, stderr = run_command(["python3", str(script), str(run_dir)])
    total_ms = (time.perf_counter() - start) * 1000
    
    # Load timings from output
    timings_file = run_dir / "timings.json"
    if timings_file.exists():
        with open(timings_file) as f:
            timings = json.load(f)
    else:
        timings = {"total_ms": total_ms, "impl": script.stem}
    
    return timings


def run_cpp_impl(binary: Path, run_dir: Path) -> dict[str, Any]:
    """Run C++ implementation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.perf_counter()
    rc, stdout, stderr = run_command([str(binary), str(run_dir)])
    total_ms = (time.perf_counter() - start) * 1000
    
    timings_file = run_dir / "timings.json"
    if timings_file.exists():
        with open(timings_file) as f:
            timings = json.load(f)
    else:
        timings = {"total_ms": total_ms, "impl": "cpp_manual"}
    
    return timings


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple runs into summary statistics."""
    if not runs:
        return {}
    
    total_times = [r.get("total_ms", 0) for r in runs]
    final_losses = [r.get("final_loss", 1.0) for r in runs]
    
    result = {
        "runs": len(runs),
        "total_ms_median": median(total_times),
        "total_ms_min": min(total_times),
        "total_ms_max": max(total_times),
        "final_loss_median": median(final_losses),
    }
    
    if len(total_times) >= 4:
        q = quantiles(total_times, n=4)
        result["total_ms_p25"] = q[0]
        result["total_ms_p75"] = q[2]
    
    # Copy other fields from first run
    for key in runs[0]:
        if key not in result and key not in ["total_ms", "final_loss"]:
            result[key] = runs[0][key]
    
    return result


def main() -> int:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    num_runs = config["benchmark"]["num_runs"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = OUT_DIR / "runs" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"BENCHMARK: {num_runs} runs per implementation")
    print(f"Output: {runs_dir}")
    print("=" * 70)
    
    # Generate init weights
    print("\n[1] Generating initial weights...")
    subprocess.run(["python3", str(ROOT / "python" / "init_gen.py")], check=True)
    
    # Compile binaries
    print("\n[2] Compiling binaries...")
    # Use xor_bench.noma with fixed iterations for fair comparison
    noma_file = ROOT / "noma" / "xor_bench.noma"
    noma_binary = runs_dir / "noma_compiled_bin"
    cpp_source = ROOT / "baselines" / "cpp_manual" / "xor.cpp"
    cpp_binary = runs_dir / "cpp_manual_bin"
    
    noma_compile_time = compile_noma(noma_file, noma_binary, ROOT.parent)
    cpp_compile_time = compile_cpp(cpp_source, cpp_binary)
    
    print(f"  NOMA compile: {noma_compile_time:.0f}ms")
    print(f"  C++ compile:  {cpp_compile_time:.0f}ms")
    
    # Run implementations
    implementations = {
        "noma_compiled": [],
        "noma_interpreted": [],
        "noma_reset": [],  # NOMA with optimizer state reset after growth (control)
        "numpy_manual": [],
        "cpp_manual": [],
    }
    
    # Check for PyTorch
    try:
        import torch
        has_torch = True
        implementations["torch_eager"] = []
        implementations["torch_compile"] = []
    except ImportError:
        has_torch = False
        print("\n  [Note] PyTorch not available, skipping torch baselines")
    
    print(f"\n[3] Running benchmarks ({num_runs} runs each)...")
    
    for run_idx in range(num_runs):
        print(f"\n  Run {run_idx + 1}/{num_runs}")
        
        # NOMA compiled
        run_dir = runs_dir / f"run_{run_idx}" / "noma_compiled"
        timings = run_noma_compiled(noma_binary, run_dir)
        implementations["noma_compiled"].append(timings)
        print(f"    noma_compiled: {timings['total_ms']:.2f}ms")
        
        # NOMA interpreted
        run_dir = runs_dir / f"run_{run_idx}" / "noma_interpreted"
        timings = run_noma_interpreted(noma_file, run_dir, ROOT.parent)
        implementations["noma_interpreted"].append(timings)
        print(f"    noma_interpreted: {timings['total_ms']:.2f}ms")
        
        # NOMA reset (control - resets optimizer state after growth)
        noma_reset_file = ROOT / "noma" / "xor_reset.noma"
        run_dir = runs_dir / f"run_{run_idx}" / "noma_reset"
        timings = run_noma_interpreted(noma_reset_file, run_dir, ROOT.parent)
        implementations["noma_reset"].append(timings)
        print(f"    noma_reset: {timings['total_ms']:.2f}ms (control)")
        
        # NumPy manual
        run_dir = runs_dir / f"run_{run_idx}" / "numpy_manual"
        timings = run_python_impl(ROOT / "baselines" / "python_numpy_manual" / "xor.py", run_dir)
        implementations["numpy_manual"].append(timings)
        print(f"    numpy_manual: {timings['total_ms']:.2f}ms")
        
        # C++ manual
        run_dir = runs_dir / f"run_{run_idx}" / "cpp_manual"
        timings = run_cpp_impl(cpp_binary, run_dir)
        implementations["cpp_manual"].append(timings)
        print(f"    cpp_manual: {timings['total_ms']:.2f}ms")
        
        # PyTorch eager
        if has_torch:
            run_dir = runs_dir / f"run_{run_idx}" / "torch_eager"
            timings = run_python_impl(ROOT / "baselines" / "python_torch" / "xor_eager.py", run_dir)
            implementations["torch_eager"].append(timings)
            print(f"    torch_eager: {timings['total_ms']:.2f}ms")
            
            run_dir = runs_dir / f"run_{run_idx}" / "torch_compile"
            timings = run_python_impl(ROOT / "baselines" / "python_torch" / "xor_compile.py", run_dir)
            implementations["torch_compile"].append(timings)
            print(f"    torch_compile: {timings['total_ms']:.2f}ms")
    
    # Aggregate results
    print("\n[4] Aggregating results...")
    summary = {
        "timestamp": timestamp,
        "config": config,
        "compile_times_ms": {
            "noma": noma_compile_time,
            "cpp": cpp_compile_time
        },
        "results": {}
    }
    
    for impl, runs in implementations.items():
        if runs:
            summary["results"][impl] = aggregate_runs(runs)
    
    # Save summary
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS (median of {} runs)".format(num_runs))
    print("=" * 70)
    print(f"{'Implementation':<25} {'Time (ms)':<15} {'Final Loss':<15}")
    print("-" * 55)
    
    for impl, stats in summary["results"].items():
        time_str = f"{stats['total_ms_median']:.2f}"
        loss_str = f"{stats['final_loss_median']:.6f}"
        print(f"{impl:<25} {time_str:<15} {loss_str:<15}")
    
    print("\nCompile times:")
    print(f"  NOMA: {noma_compile_time:.0f}ms")
    print(f"  C++:  {cpp_compile_time:.0f}ms")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
