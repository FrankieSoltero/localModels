# Experiment 02 — Mixed Precision (autocast)

**Hypothesis:** wrapping the training forward pass in `torch.autocast` (bf16)
cuts wall-clock time with no loss of quality, because Phase 0 measured bf16
matmul at ~8.5× fp32 on the GPU and ~3.5× on the CPU
(`../00_matmul_bench/RESULTS.md`).

**One variable changed** vs `01_tiny_gpt_baseline`: the autocast wrapper.
Same model, data, seed, and 24.6M-token budget. Val loss is computed in fp32
either way, so quality is directly comparable.

**Success criterion:** same-or-better val loss than the fp32 baseline *on the
same device*, in less wall-clock time.

## Runs

```powershell
# GPU (auto-tags bf16-gpu; compare to baseline-gpu in ../01_tiny_gpt_baseline/runs.csv)
& "modelEnvGpu\Scripts\python.exe" experiments\02_mixed_precision\train.py

# CPU (auto-tags bf16-cpu; compare to baseline-cpu) — run on a quiet machine
& "modelEnv\Scripts\python.exe" experiments\02_mixed_precision\train.py --device cpu
```

Results log: `runs.csv` in this folder.

## Notes

- fp16 is refused on CPU (software-emulated, ~1000× slow — Phase 0 finding).
- On GPU, `--dtype fp16` is also available (RDNA4 runs it at bf16 speed);
  a `GradScaler` guards against fp16 gradient underflow. bf16 needs no scaler.
- The causal mask is `bool` here instead of the baseline's additive `-inf`
  float mask — numerically equivalent, but required for autocast dtype
  compatibility inside `nn.MultiheadAttention`.
