#!/usr/bin/env python3
"""
Generate loss.png with growth marker from latest benchmark run.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "out"


def load_loss_csv(path: Path) -> tuple[list[int], list[float], list[int]]:
    """Load loss.csv and return steps, losses, hidden sizes."""
    steps, losses, hiddens = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            hiddens.append(int(row["hidden"]))
    return steps, losses, hiddens


def find_latest_run() -> Path | None:
    """Find the most recent run directory."""
    runs_dir = OUT_DIR / "runs"
    if not runs_dir.exists():
        return None
    
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name != "latest"])
    return run_dirs[-1] if run_dirs else None


def main() -> int:
    latest = find_latest_run()
    if not latest:
        print("No benchmark runs found. Run benchmark.py first.")
        return 1
    
    print(f"Loading data from: {latest}")
    
    # Find run_0 for each implementation
    run0 = latest / "run_0"
    if not run0.exists():
        print("No run_0 directory found.")
        return 1
    
    implementations = {}
    colors = {
        "noma": "#2ecc71",
        "noma_interpreted": "#27ae60",
        "noma_reset": "#e67e22",  # Orange for reset (control)
        "numpy_manual": "#e74c3c",
        "cpp_manual": "#9b59b6",
        "torch_eager": "#3498db",
        "torch_compile": "#1abc9c",
    }
    
    # Labels for display (cleaner names)
    display_names = {
        "noma": "NOMA (preserve)",
        "noma_interpreted": "NOMA (interpreted)",
        "noma_reset": "NOMA (reset)",
        "numpy_manual": "NumPy",
        "cpp_manual": "C++",
        "torch_eager": "PyTorch Eager",
        "torch_compile": "PyTorch Compile",
    }
    
    for impl_dir in run0.iterdir():
        if impl_dir.is_dir():
            loss_file = impl_dir / "loss.csv"
            if loss_file.exists():
                steps, losses, hiddens = load_loss_csv(loss_file)
                # Skip implementations with too few data points (compiled binaries)
                if len(steps) < 10:
                    print(f"  Skipping {impl_dir.name}: only {len(steps)} points")
                    continue
                # Rename noma_interpreted to just "noma" for display
                name = "noma" if impl_dir.name == "noma_interpreted" else impl_dir.name
                # Skip noma_compiled since we use noma_interpreted for curves
                if impl_dir.name == "noma_compiled":
                    continue
                implementations[name] = (steps, losses, hiddens)
    
    if not implementations:
        print("No loss.csv files found.")
        return 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Line styles to differentiate overlapping curves
    line_styles = {
        "noma": "-",
        "noma_reset": "-.",  # Dash-dot for reset control
        "numpy_manual": "-",
        "cpp_manual": "--",  # Dashed to show when overlapping with NumPy
        "torch_eager": "-",
        "torch_compile": ":",
    }
    line_widths = {
        "noma": 2.0,
        "noma_reset": 2.0,
        "numpy_manual": 1.5,
        "cpp_manual": 2.5,  # Thicker dashed line
        "torch_eager": 1.5,
        "torch_compile": 2.0,
    }
    
    # Plot 1: Full loss curves
    # Sort to ensure consistent plotting order (NOMA first, then others)
    plot_order = ["noma", "noma_reset", "cpp_manual", "numpy_manual", "torch_eager", "torch_compile"]
    sorted_impls = sorted(implementations.items(), key=lambda x: plot_order.index(x[0]) if x[0] in plot_order else 99)
    
    for impl, (steps, losses, hiddens) in sorted_impls:
        color = colors.get(impl, "#95a5a6")
        label = display_names.get(impl, impl)
        ls = line_styles.get(impl, "-")
        lw = line_widths.get(impl, 1.5)
        ax1.plot(steps, losses, ls, color=color, linewidth=lw, label=label, alpha=0.9)
        
        # Mark growth point
        growth_idx = next((i for i, h in enumerate(hiddens) if h > hiddens[0]), None)
        if growth_idx:
            ax1.axvline(steps[growth_idx], color=color, linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Convergence')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add growth annotation
    ax1.axvline(200, color='gray', linestyle=':', alpha=0.5)
    ax1.annotate('Growth\n(2->16)', xy=(200, 0.1), xytext=(220, 0.2),
                 fontsize=9, ha='left', va='bottom',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # Plot 2: Post-growth zoom (first 50 steps after growth)
    for impl, (steps, losses, hiddens) in sorted_impls:
        growth_idx = next((i for i, h in enumerate(hiddens) if h > hiddens[0]), None)
        if growth_idx and growth_idx + 50 < len(losses):
            color = colors.get(impl, "#95a5a6")
            label = display_names.get(impl, impl)
            ls = line_styles.get(impl, "-")
            lw = line_widths.get(impl, 1.5)
            x = range(50)
            y = losses[growth_idx:growth_idx + 50]
            ax2.plot(x, y, ls, color=color, linewidth=lw, label=label, alpha=0.9)
    
    ax2.set_xlabel('Steps after growth')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Post-Growth Convergence (Effect of Optimizer State)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('XOR Training: Dynamic Architecture Growth', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output = OUT_DIR / "loss.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
