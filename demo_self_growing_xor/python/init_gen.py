#!/usr/bin/env python3
"""
Generate deterministic initial weights for all implementations.
Ensures identical starting conditions across NOMA, Python, C++, and PyTorch.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.json"
OUTPUT_PATH = ROOT / "out" / "init" / "init_weights.json"


def main() -> int:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    hp = config["hyperparams"]
    hidden_initial = hp["hidden_initial"]
    hidden_final = hp["hidden_final"]
    
    # Fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Phase 1 weights (small network)
    W1 = rng.normal(0.0, 0.5, size=(2, hidden_initial)).tolist()
    b1 = np.zeros(hidden_initial).tolist()
    W2 = rng.normal(0.0, 0.5, size=(hidden_initial, 1)).tolist()
    b2 = np.zeros(1).tolist()
    
    # Extra weights for growth (hidden_final - hidden_initial new neurons)
    extra_neurons = hidden_final - hidden_initial
    W1_extra = rng.normal(0.0, 0.3, size=(2, extra_neurons)).tolist()
    b1_extra = np.zeros(extra_neurons).tolist()
    W2_extra = rng.normal(0.0, 0.3, size=(extra_neurons, 1)).tolist()
    
    init_data = {
        "seed": 42,
        "hidden_initial": hidden_initial,
        "hidden_final": hidden_final,
        "phase1": {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        },
        "growth_extension": {
            "W1_extra": W1_extra,
            "b1_extra": b1_extra,
            "W2_extra": W2_extra
        }
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(init_data, f, indent=2)
    
    print(f"Generated: {OUTPUT_PATH}")
    print(f"  Phase 1: W1={len(W1)}x{len(W1[0])}, W2={len(W2)}x{len(W2[0])}")
    print(f"  Growth:  W1_extra={len(W1_extra)}x{len(W1_extra[0])}, W2_extra={len(W2_extra)}x{len(W2_extra[0])}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
