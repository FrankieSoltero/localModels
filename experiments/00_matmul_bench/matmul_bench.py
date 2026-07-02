#!/usr/bin/env python3
"""
Matmul cost baseline for this machine.

Measures GEMM throughput (GFLOP/s) across matrix sizes and dtypes, on every
available torch device. This table is the exchange rate between "theoretical
FLOPs saved" and "wall-clock time saved" for all later experiments.

Run: python experiments/00_matmul_bench/matmul_bench.py [--sizes 512 1024 2048] [--csv out.csv]
"""

import argparse
import csv
import sys
import time

import torch


def bench_matmul(n: int, dtype: torch.dtype, device: str, min_seconds: float = 0.5) -> float:
    """Return sustained GFLOP/s for an n x n @ n x n matmul."""
    try:
        a = torch.randn(n, n, dtype=dtype, device=device)
        b = torch.randn(n, n, dtype=dtype, device=device)
    except (RuntimeError, TypeError):
        return float("nan")  # dtype unsupported on this device

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        a @ b
    sync()

    # Timed: run enough iterations to fill min_seconds
    iters = 1
    while True:
        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        sync()
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            break
        iters = max(iters * 2, int(iters * min_seconds / max(elapsed, 1e-9)) + 1)

    flops = 2 * n**3 * iters  # n^3 multiply-adds
    return flops / elapsed / 1e9


def transformer_context(gflops_table: dict) -> None:
    """Translate measured throughput into training-time estimates."""
    print("\n--- What this means for training ---")
    # Training cost ~ 6 * params * tokens FLOPs
    scenarios = [
        ("10M-param tiny GPT, 100M tokens", 10e6, 100e6),
        ("124M-param GPT-2 small, 1B tokens", 124e6, 1e9),
        ("3B-param model, 1B tokens (fine-tune scale)", 3e9, 1e9),
    ]
    best = max((v for v in gflops_table.values() if v == v), default=None)
    if not best:
        return
    print(f"Best sustained throughput measured: {best:.1f} GFLOP/s")
    for name, params, tokens in scenarios:
        total_flops = 6 * params * tokens
        hours = total_flops / (best * 1e9) / 3600
        print(f"  {name}: ~{total_flops:.2e} FLOPs -> ~{hours:,.1f} h at that rate")
    print("(Upper bound: real training hits memory/attention overheads, so expect worse.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--csv", type=str, default=None, help="optional path to write results CSV")
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    dtypes = [torch.float32, torch.bfloat16, torch.float16]

    print(f"torch {torch.__version__} | devices: {devices} | threads: {torch.get_num_threads()}")
    header = f"{'device':>6} {'dtype':>10} " + " ".join(f"{n:>10}" for n in args.sizes)
    print(header)
    print("-" * len(header))

    rows = []
    summary = {}
    for device in devices:
        for dtype in dtypes:
            results = []
            for n in args.sizes:
                gf = bench_matmul(n, dtype, device)
                results.append(gf)
                rows.append({"device": device, "dtype": str(dtype), "n": n, "gflops": gf})
                summary[(device, str(dtype), n)] = gf
            cells = " ".join(f"{r:>10.1f}" if r == r else f"{'n/a':>10}" for r in results)
            print(f"{device:>6} {str(dtype).replace('torch.', ''):>10} {cells}")

    transformer_context(summary)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["device", "dtype", "n", "gflops"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
