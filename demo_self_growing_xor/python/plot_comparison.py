#!/usr/bin/env python3
"""
Plot NOMA vs Python vs C++ loss curves side by side.
Shows that all three converge, but highlights:
1. Python/C++ require manual growth plumbing (optimizer reset)
2. NOMA does it with just `realloc`
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV and return steps, losses, hidden arrays."""
    steps, losses, hiddens = [], [], []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            hiddens.append(int(row["hidden"]))
    return np.array(steps), np.array(losses), np.array(hiddens)


def find_growth_step(steps: np.ndarray, hiddens: np.ndarray) -> int | None:
    """Find step where hidden changes."""
    for i in range(1, len(hiddens)):
        if hiddens[i] != hiddens[i - 1]:
            return int(steps[i])
    return None


def main() -> int:
    out_dir = Path(__file__).parent.parent / "out"
    noma_csv = out_dir / "noma_loss.csv"
    python_csv = out_dir / "python_loss.csv"
    cpp_csv = out_dir / "cpp_loss.csv"
    out_png = out_dir / "comparison.png"

    if not noma_csv.exists():
        print(f"[plot] missing {noma_csv}")
        return 1
    if not python_csv.exists():
        print(f"[plot] missing {python_csv}")
        return 1

    noma_steps, noma_loss, noma_hidden = load_csv(noma_csv)
    py_steps, py_loss, py_hidden = load_csv(python_csv)
    
    has_cpp = cpp_csv.exists()
    if has_cpp:
        cpp_steps, cpp_loss, cpp_hidden = load_csv(cpp_csv)

    noma_growth = find_growth_step(noma_steps, noma_hidden)
    py_growth = find_growth_step(py_steps, py_hidden)
    cpp_growth = find_growth_step(cpp_steps, cpp_hidden) if has_cpp else None

    n_plots = 3 if has_cpp else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)

    # NOMA plot
    ax = axes[0]
    ax.plot(noma_steps, noma_loss, "-", color="#3498db", linewidth=1.5, label="NOMA loss")
    if noma_growth is not None:
        ax.axvline(noma_growth, color="red", linestyle="--", linewidth=1.5, label=f"growth @ step {noma_growth}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("NOMA\n2 lines: realloc W1, W2")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Python plot
    ax = axes[1]
    ax.plot(py_steps, py_loss, "-", color="#e74c3c", linewidth=1.5, label="Python loss")
    if py_growth is not None:
        ax.axvline(py_growth, color="red", linestyle="--", linewidth=1.5, label=f"growth @ step {py_growth}")
    ax.set_xlabel("Step")
    ax.set_title("Python\n~100 lines + Adam reset")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # C++ plot
    if has_cpp:
        ax = axes[2]
        ax.plot(cpp_steps, cpp_loss, "-", color="#9b59b6", linewidth=1.5, label="C++ loss")
        if cpp_growth is not None:
            ax.axvline(cpp_growth, color="red", linestyle="--", linewidth=1.5, label=f"growth @ step {cpp_growth}")
        ax.set_xlabel("Step")
        ax.set_title("C++\n~220 lines + Adam reset")
        ax.legend(loc="upper right")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Self-Growing XOR: Loss Comparison (all converge, NOMA = simplest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[plot] saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
