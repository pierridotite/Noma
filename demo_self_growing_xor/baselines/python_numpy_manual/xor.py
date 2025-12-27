#!/usr/bin/env python3
"""
XOR baseline with manual gradients using NumPy.
Produces normalized output: loss.csv, timings.json, stdout.txt
"""
import json
import sys
import time
import resource
from pathlib import Path
from dataclasses import dataclass
import numpy as np

ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT / "config.json"
INIT_PATH = ROOT / "out" / "init" / "init_weights.json"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred > 0.5).astype(float) == target))


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def zeros_like(cls, arr: np.ndarray) -> "AdamState":
        return cls(m=np.zeros_like(arr), v=np.zeros_like(arr), t=0)

    def update(self, param: np.ndarray, grad: np.ndarray, lr: float,
               beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> np.ndarray:
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "runs" / "latest" / "numpy_manual"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    hp = config["hyperparams"]
    bench = config["benchmark"]
    ds = config["dataset"]

    X = np.array(ds["X"], dtype=np.float64)
    Y = np.array(ds["Y"], dtype=np.float64)

    # Load initial weights
    with open(INIT_PATH) as f:
        init = json.load(f)

    W1 = np.array(init["phase1"]["W1"], dtype=np.float64)
    W2 = np.array(init["phase1"]["W2"], dtype=np.float64)

    growth_step = hp["growth_step"]
    hidden_initial = hp["hidden_initial"]
    hidden_final = hp["hidden_final"]

    # Adam states
    adam_W1 = AdamState.zeros_like(W1)
    adam_W2 = AdamState.zeros_like(W2)

    # Timing
    start_total = time.perf_counter()
    cold_start_end = time.perf_counter()
    cold_start_ms = (cold_start_end - start_total) * 1000

    loss_records = []
    stdout_lines = []
    step = 0
    phase = 1
    hidden = hidden_initial
    lr = hp["lr_phase1"]

    warmup_iters = bench["warmup_iters"]
    warmup_start = None
    warmup_end = None
    steady_start = None
    growth_start = None
    growth_end = None
    steady_times = []

    max_iter = hp["max_iter_phase1"] + hp["max_iter_phase2"]
    threshold_final = hp["threshold_final"]
    time_to_threshold = None

    while step < max_iter:
        iter_start = time.perf_counter()

        # Forward
        h = sigmoid(X @ W1)
        pred = sigmoid(h @ W2)
        err = pred - Y
        loss = float(np.mean(err * err))
        acc = compute_accuracy(pred, Y)

        # Record
        loss_records.append((step, loss, acc, hidden, phase))
        line = f"step={step} loss={loss:.6f} acc={acc:.2f} hidden={hidden}"
        stdout_lines.append(line)

        if loss < threshold_final and time_to_threshold is None:
            time_to_threshold = (time.perf_counter() - start_total) * 1000

        # Check termination
        if phase == 1 and (loss < hp["threshold_phase1"] or step >= hp["max_iter_phase1"] - 1):
            if step >= growth_step - 1:
                # Growth event
                growth_start = time.perf_counter()
                stdout_lines.append(f"GROWTH TRIGGERED: hidden {hidden_initial} -> {hidden_final}")

                # Extend weights
                W1_extra = np.array(init["growth_extension"]["W1_extra"], dtype=np.float64)
                W2_extra = np.array(init["growth_extension"]["W2_extra"], dtype=np.float64)

                W1_new = np.hstack([W1, W1_extra])
                W2_new = np.vstack([W2, W2_extra])

                # Reset Adam state (this is the cost we measure)
                adam_W1 = AdamState.zeros_like(W1_new)
                adam_W2 = AdamState.zeros_like(W2_new)

                W1, W2 = W1_new, W2_new
                hidden = hidden_final
                phase = 2
                lr = hp["lr_phase2"]
                growth_end = time.perf_counter()

                step += 1
                continue

        if phase == 2 and loss < threshold_final:
            break

        # Backward (manual gradients)
        dloss = 2.0 * err / X.shape[0]
        dout = dloss * pred * (1.0 - pred)
        dW2 = h.T @ dout
        dh = dout @ W2.T
        dh_act = dh * h * (1.0 - h)
        dW1 = X.T @ dh_act

        # Adam update
        W1 = adam_W1.update(W1, dW1, lr, hp["beta1"], hp["beta2"], hp["epsilon"])
        W2 = adam_W2.update(W2, dW2, lr, hp["beta1"], hp["beta2"], hp["epsilon"])

        iter_end = time.perf_counter()

        # Timing bookkeeping
        if step == 0:
            warmup_start = iter_start
        if step == warmup_iters - 1:
            warmup_end = iter_end
            steady_start = iter_end
        if step >= warmup_iters:
            steady_times.append((iter_end - iter_start) * 1e6)  # microseconds

        step += 1

    end_total = time.perf_counter()

    # Compute timing metrics
    total_ms = (end_total - start_total) * 1000
    warmup_ms = ((warmup_end - warmup_start) * 1000) if warmup_end else 0
    steady_ms = ((end_total - steady_start) * 1000) if steady_start else 0
    growth_ms = ((growth_end - growth_start) * 1000) if growth_start and growth_end else 0

    steady_times_arr = np.array(steady_times) if steady_times else np.array([0])
    steady_median = float(np.median(steady_times_arr))
    steady_p95 = float(np.percentile(steady_times_arr, 95))

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    timings = {
        "cold_start_ms": cold_start_ms,
        "compile_overhead_ms": 0,
        "train_warmup_ms": warmup_ms,
        "train_steady_ms": steady_ms,
        "steady_step_us_median": steady_median,
        "steady_step_us_p95": steady_p95,
        "growth_event_ms": growth_ms,
        "total_ms": total_ms,
        "iters_total": step,
        "time_to_threshold_ms": time_to_threshold,
        "final_loss": loss_records[-1][1],
        "final_accuracy": loss_records[-1][2],
        "peak_rss_kb": peak_rss,
        "precision": "f64",
        "device": "cpu",
        "impl": "numpy_manual"
    }

    # Write outputs
    with open(out_dir / "loss.csv", "w") as f:
        f.write("step,loss,accuracy,hidden,phase\n")
        for step, loss, acc, hid, ph in loss_records:
            f.write(f"{step},{loss:.6f},{acc:.2f},{hid},{ph}\n")

    with open(out_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2)

    with open(out_dir / "stdout.txt", "w") as f:
        f.write("\n".join(stdout_lines))

    print(f"[numpy_manual] {step} iters, final_loss={loss_records[-1][1]:.6f}, total={total_ms:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
