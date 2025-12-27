#!/usr/bin/env python3
"""Plot loss curve with growth marker.
Usage: python plot_loss.py out/noma_loss.csv out/loss.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_csv(path: Path) -> Tuple[List[int], List[float], List[int]]:
    steps: List[int] = []
    losses: List[float] = []
    hidden: List[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            hidden.append(int(float(row["hidden"])))
    return steps, losses, hidden


def find_growth_step(hidden: List[int]) -> int | None:
    if not hidden:
        return None
    start = hidden[0]
    for idx, h in enumerate(hidden):
        if h != start:
            return idx
    return None


def plot_loss(csv_path: Path, png_path: Path) -> None:
    steps, losses, hidden = load_csv(csv_path)
    growth_idx = find_growth_step(hidden)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(steps, losses, label="NOMA loss", color="#1f77b4")

    if growth_idx is not None:
        growth_step = steps[growth_idx]
        plt.axvline(growth_step, color="#d62728", linestyle="--", linewidth=1.2, label=f"growth @ step {growth_step}")
        plt.scatter([growth_step], [losses[growth_idx]], color="#d62728", zorder=5)

    plt.xlabel("step")
    plt.ylabel("loss (MSE)")
    plt.title("Self-growing XOR in NOMA")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=160)
    print(f"[plot] saved {png_path}")


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python plot_loss.py <input_csv> <output_png>")
        return 1
    csv_path = Path(sys.argv[1])
    png_path = Path(sys.argv[2])
    plot_loss(csv_path, png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
