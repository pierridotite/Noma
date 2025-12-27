#!/usr/bin/env python3
"""
XOR with PyTorch eager mode (runtime autodiff).
Produces normalized output: loss.csv, timings.json, stdout.txt
"""
import json
import sys
import time
import resource
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT / "config.json"
INIT_PATH = ROOT / "out" / "init" / "init_weights.json"


class XORNet(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(h))


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred > 0.5).float() == target).float().mean())


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "runs" / "latest" / "torch_eager"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float64)

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    hp = config["hyperparams"]
    bench = config["benchmark"]
    ds = config["dataset"]

    X = torch.tensor(ds["X"], dtype=torch.float64)
    Y = torch.tensor(ds["Y"], dtype=torch.float64)

    # Load initial weights
    with open(INIT_PATH) as f:
        init = json.load(f)

    hidden_initial = hp["hidden_initial"]
    hidden_final = hp["hidden_final"]
    growth_step = hp["growth_step"]

    # Create model with initial weights
    model = XORNet(hidden_initial)
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(init["phase1"]["W1"]).T)
        model.fc2.weight.copy_(torch.tensor(init["phase1"]["W2"]).T)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr_phase1"],
                                  betas=(hp["beta1"], hp["beta2"]), eps=hp["epsilon"])

    # Timing
    start_total = time.perf_counter()
    cold_start_ms = 0  # Already loaded

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
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss_val = loss.item()
        acc = compute_accuracy(pred, Y)

        # Record
        loss_records.append((step, loss_val, acc, hidden, phase))
        line = f"step={step} loss={loss_val:.6f} acc={acc:.2f} hidden={hidden}"
        stdout_lines.append(line)

        if loss_val < threshold_final and time_to_threshold is None:
            time_to_threshold = (time.perf_counter() - start_total) * 1000

        # Check for growth
        if phase == 1 and step >= growth_step - 1:
            growth_start = time.perf_counter()
            stdout_lines.append(f"GROWTH TRIGGERED: hidden {hidden_initial} -> {hidden_final}")

            # Create new larger model
            old_W1 = model.fc1.weight.data.clone()
            old_W2 = model.fc2.weight.data.clone()

            model = XORNet(hidden_final)
            with torch.no_grad():
                # Copy old weights
                model.fc1.weight[:hidden_initial, :] = old_W1
                model.fc2.weight[:, :hidden_initial] = old_W2
                # Initialize new weights
                W1_extra = torch.tensor(init["growth_extension"]["W1_extra"]).T
                W2_extra = torch.tensor(init["growth_extension"]["W2_extra"]).T
                model.fc1.weight[hidden_initial:, :] = W1_extra
                model.fc2.weight[:, hidden_initial:] = W2_extra

            # New optimizer (state reset - this is the cost)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr_phase2"],
                                          betas=(hp["beta1"], hp["beta2"]), eps=hp["epsilon"])

            hidden = hidden_final
            phase = 2
            growth_end = time.perf_counter()
            step += 1
            continue

        if phase == 2 and loss_val < threshold_final:
            break

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_end = time.perf_counter()

        # Timing bookkeeping
        if step == 0:
            warmup_start = iter_start
        if step == warmup_iters - 1:
            warmup_end = iter_end
            steady_start = iter_end
        if step >= warmup_iters:
            steady_times.append((iter_end - iter_start) * 1e6)

        step += 1

    end_total = time.perf_counter()

    # Compute timing metrics
    total_ms = (end_total - start_total) * 1000
    warmup_ms = ((warmup_end - warmup_start) * 1000) if warmup_end else 0
    steady_ms = ((end_total - steady_start) * 1000) if steady_start else 0
    growth_ms = ((growth_end - growth_start) * 1000) if growth_start and growth_end else 0

    import numpy as np
    steady_times_arr = np.array(steady_times) if steady_times else np.array([0])
    steady_median = float(np.median(steady_times_arr))
    steady_p95 = float(np.percentile(steady_times_arr, 95))

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

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
        "impl": "torch_eager"
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

    print(f"[torch_eager] {step} iters, final_loss={loss_records[-1][1]:.6f}, total={total_ms:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
