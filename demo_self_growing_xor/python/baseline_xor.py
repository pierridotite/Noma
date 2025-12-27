#!/usr/bin/env python3
"""
Minimal Python baseline for XOR.
- Phase 1: tiny hidden layer (2 units) with vanilla SGD
- Phase 2: manual growth to 16 units that *resets* optimizer state to show the plumbing cost
Outputs: csv at path provided as first arg (default: out/python_loss.csv)
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

RNG = np.random.default_rng(42)  # seed for reproducibility


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = sigmoid(x @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    return h, out


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred - target
    return float(np.mean(diff * diff))


@dataclass
class AdamState:
    """Adam optimizer state for a single parameter."""
    m: np.ndarray  # first moment
    v: np.ndarray  # second moment
    t: int = 0     # timestep

    @classmethod
    def zeros_like(cls, arr: np.ndarray) -> "AdamState":
        return cls(m=np.zeros_like(arr), v=np.zeros_like(arr), t=0)


def adam_update(param: np.ndarray, grad: np.ndarray, state: AdamState, 
                lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[np.ndarray, AdamState]:
    state.t += 1
    state.m = beta1 * state.m + (1 - beta1) * grad
    state.v = beta2 * state.v + (1 - beta2) * (grad ** 2)
    m_hat = state.m / (1 - beta1 ** state.t)
    v_hat = state.v / (1 - beta2 ** state.t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, state


def train_phase_adam(x: np.ndarray, y: np.ndarray, steps: int, lr: float, hidden: int,
                     W1: np.ndarray, W2: np.ndarray,
                     adam_states: dict | None = None) -> Tuple[Tuple[np.ndarray, np.ndarray], list[float], dict]:
    """Train with Adam optimizer (no biases, matching NOMA). Returns updated params, losses, and adam states."""
    if adam_states is None:
        adam_states = {
            "W1": AdamState.zeros_like(W1),
            "W2": AdamState.zeros_like(W2),
        }
    
    losses: list[float] = []
    for _ in range(steps):
        # Forward (no biases, like NOMA)
        h = sigmoid(x @ W1)
        out = sigmoid(h @ W2)
        
        # Compute loss
        diff = out - y
        loss = float(np.mean(diff * diff))
        losses.append(loss)
        
        # Backward
        dloss = 2.0 * diff / x.shape[0]
        dout = dloss * out * (1.0 - out)
        dW2 = h.T @ dout
        
        dh = dout @ W2.T
        dh_act = dh * h * (1.0 - h)
        dW1 = x.T @ dh_act
        
        # Adam updates
        W1, adam_states["W1"] = adam_update(W1, dW1, adam_states["W1"], lr)
        W2, adam_states["W2"] = adam_update(W2, dW2, adam_states["W2"], lr)
    
    return (W1, W2), losses, adam_states


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/python_loss.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [1.0], [1.0], [0.0]])

    # Phase 1: tiny network (hidden=2), expect plateau — SAME AS NOMA
    hidden_small = 2
    W1 = RNG.normal(scale=0.5, size=(2, hidden_small))
    W2 = RNG.normal(scale=0.5, size=(hidden_small, 1))

    params, loss1, adam_states = train_phase_adam(X, Y, steps=200, lr=0.05, hidden=hidden_small,
                                                   W1=W1, W2=W2, adam_states=None)

    # Phase 2: grow to 16 hidden units — MUST RESET ADAM STATE (this is the pain point!)
    hidden_big = 16
    W1_old, W2_old = params
    
    # Create new larger arrays and copy old weights
    W1_new = RNG.normal(scale=0.3, size=(2, hidden_big))
    W1_new[:, :hidden_small] = W1_old

    W2_new = RNG.normal(scale=0.3, size=(hidden_big, 1))
    W2_new[:hidden_small, :] = W2_old

    # RESET Adam state — this is what NOMA avoids!
    params2, loss2, _ = train_phase_adam(X, Y, steps=342, lr=0.12, hidden=hidden_big,
                                          W1=W1_new, W2=W2_new,
                                          adam_states=None)  # <-- RESET!

    # Write CSV with hidden column
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "hidden"])
        step = 0
        for loss in loss1:
            writer.writerow([step, f"{loss:.6f}", hidden_small])
            step += 1
        for loss in loss2:
            writer.writerow([step, f"{loss:.6f}", hidden_big])
            step += 1

    print(f"[baseline] wrote {out_path} (total steps={step})")
    print("[baseline] note: growth RESETS optimizer state — unlike NOMA demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
