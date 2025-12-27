#!/usr/bin/env python3
"""
Generate summary tables and comparison.png from summary.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "out"


def main() -> int:
    summary_path = OUT_DIR / "summary.json"
    if not summary_path.exists():
        print("No summary.json found. Run benchmark.py first.")
        return 1
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    results = summary["results"]
    compile_times = summary.get("compile_times_ms", {})
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 10))
    
    # Prepare data
    impls = list(results.keys())
    times = [results[impl]["total_ms_median"] for impl in impls]
    
    colors = {
        "noma_compiled": "#2ecc71",
        "noma_interpreted": "#27ae60",
        "numpy_manual": "#e74c3c",
        "cpp_manual": "#9b59b6",
        "torch_eager": "#3498db",
        "torch_compile": "#1abc9c",
    }
    bar_colors = [colors.get(impl, "#95a5a6") for impl in impls]
    
    # Plot 1: Execution time (log scale)
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(impls, times, color=bar_colors, edgecolor='black')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time (median)')
    ax1.set_yscale('log')
    ax1.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 f'{t:.1f}', ha='center', va='bottom', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Execution time (linear, without slowest)
    ax2 = fig.add_subplot(2, 2, 2)
    # Filter out interpreted versions for clearer comparison
    fast_impls = [impl for impl in impls if "interpreted" not in impl]
    fast_times = [results[impl]["total_ms_median"] for impl in fast_impls]
    fast_colors = [colors.get(impl, "#95a5a6") for impl in fast_impls]
    
    bars = ax2.bar(fast_impls, fast_times, color=fast_colors, edgecolor='black')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('Execution Time (compiled only)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, fast_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{t:.1f}', ha='center', va='bottom', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Compile time
    ax3 = fig.add_subplot(2, 2, 3)
    compile_impls = list(compile_times.keys())
    compile_vals = list(compile_times.values())
    compile_colors = [colors.get(f"{impl}_compiled", "#95a5a6") for impl in compile_impls]
    
    if compile_vals:
        bars = ax3.bar(compile_impls, compile_vals, color=['#2ecc71', '#9b59b6'][:len(compile_impls)], edgecolor='black')
        ax3.set_ylabel('Compile Time (ms)')
        ax3.set_title('Compile Time')
        for bar, t in zip(bars, compile_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f'{t:.0f}', ha='center', va='bottom', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Build table data
    table_data = [['Implementation', 'Time (ms)', 'Speedup vs NumPy', 'Final Loss']]
    
    numpy_time = results.get("numpy_manual", {}).get("total_ms_median", 1)
    for impl in impls:
        time_ms = results[impl]["total_ms_median"]
        speedup = numpy_time / time_ms if time_ms > 0 else 0
        loss = results[impl]["final_loss_median"]
        table_data.append([impl, f"{time_ms:.2f}", f"{speedup:.1f}x", f"{loss:.6f}"])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('XOR Benchmark: Implementation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output = OUT_DIR / "comparison.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")
    
    # Print markdown table
    print("\n## Results (Markdown)\n")
    print("| Implementation | Time (ms) | Speedup | Final Loss |")
    print("|----------------|-----------|---------|------------|")
    for row in table_data[1:]:
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
