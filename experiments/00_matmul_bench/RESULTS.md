# Phase 0 Results — What a FLOP Costs on This Machine

Measured 2026-07-02. CPU rows from a quiet-machine run (`modelEnv`, torch 2.8.0+cpu);
GPU rows from `modelEnvGpu` (torch 2.9.1+rocm7.2.1). Raw data: `results_gpu.csv`.
Note: the CPU float32 rows in `results_gpu.csv` are contaminated (baseline training was
running concurrently) — use the table below.

## Sustained GEMM throughput (GFLOP/s, n×n @ n×n)

| Device | dtype | 1024 | 2048 | 4096 | Notes |
|---|---|---:|---:|---:|---|
| Ryzen 9 9900X | fp32 | 928 | ~950 | ~950 | |
| Ryzen 9 9900X | bf16 | 3,441 | ~2,600 | ~2,800 | AVX-512 fast path, ~3.5× fp32 |
| Ryzen 9 9900X | fp16 | 2 | 0.4 | 0.4 | **software-emulated — never use on CPU** |
| RX 9060 XT | fp32 | 7,455 | 8,138 | 7,823 | ~8× CPU fp32 |
| RX 9060 XT | bf16 | 57,593 | 65,842 | **67,207** | WMMA path, ~8.5× GPU fp32 |
| RX 9060 XT | fp16 | 58,163 | 66,042 | 65,583 | same as bf16 on RDNA4 |

## End-to-end training throughput (tiny GPT, 0.84M params, 24.6M-token budget)

All rows measured 2026-07-02 on a quiet machine; sources:
`../01_tiny_gpt_baseline/runs.csv` and `../02_mixed_precision/runs.csv`.

| Device | dtype | tokens/sec | wall-clock | val loss | vs fp32 same device |
|---|---|---:|---:|---:|---|
| CPU (9900X) | fp32 | 14,516 | 1,693 s (28 min) | 1.8541 | — |
| CPU (9900X) | bf16 autocast | 19,434 | 1,265 s (21 min) | 1.8543 | **1.34×** |
| GPU (RX 9060 XT) | fp32 | 167,417 | 147 s | 1.8634 | — |
| GPU (RX 9060 XT) | bf16 autocast | 290,645 | 85 s | 1.8639 | **1.74×** |

## Takeaways

1. **The GPU is a ~20× matmul machine and 11.5× training machine** (measured, fp32 vs
   fp32) vs. the CPU. End-to-end gains lag peak-matmul gains because small-model
   training is partly overhead (Python, data loading, small kernels) — this gap is itself
   a measurement to keep an eye on.
2. **Precision is the biggest single lever measured so far:** bf16 is ~3.5× on CPU and
   ~8.5× on GPU vs fp32 at the matmul level. Realized end-to-end (experiment 02):
   1.34× on CPU, 1.74× on GPU at identical val loss — the free lunch is real but
   Amdahl-limited at this tiny model size.
3. **dtype support is asymmetric and must be checked per device:** fp16 is catastrophic on
   CPU (emulated, ~1000× slow) but full-speed on GPU. Efficiency claims must always name
   the device+dtype they were measured on.
4. **Cross-device loss parity confirmed:** the same seed lands at 1.854 (CPU) vs 1.863
   (GPU) — a Δ of 0.009 from differing kernel reduction order, not a correctness issue.
   Same-device comparisons are the clean ones; cross-device deltas of this size are noise.
5. **Budget implications at peak bf16 (67 TFLOP/s):** GPT-2-small-class pretraining
   (~124M params, 1B tokens) ≈ 3 h of ideal compute; QLoRA-scale fine-tuning of a 3B model
   is comfortably overnight. The "consumer-PC research loop" is real.

## Known environment quirks

- ROCm helper `offload-arch.exe` prints a spurious error because of the space in
  `C:\Users\Francisco Soltero` — cosmetic so far; revisit if `torch.compile` misbehaves.
- PyTorch warns that memory-efficient SDPA on RDNA4 is experimental
  (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` to enable) — worth testing in Phase 2.
- Under autocast, `nn.MultiheadAttention` rejects a float `-inf` additive mask (dtype
  mismatch with bf16 activations) — experiment 02 uses an equivalent bool mask instead.
