# Experiment 03 — Low-Rank Factorization of the MLP

**Hypothesis:** replacing each Block's MLP weight matrices W (in×out) with a
factorized pair A·B (in×r · r×out) cuts that layer's FLOPs from `in·out` to
`r·(in+out)` per token, trading model capacity for speed. Unlike experiment 02
this is NOT expected to be free — the deliverable is the **tradeoff curve**:
val loss vs wall-clock across ranks at the same 24.6M-token budget.

**One variable changed** vs `01_tiny_gpt_baseline`: the two MLP linears per
block (128→512 and 512→128) become `LowRankLinear` pairs. Attention stays
stock `nn.MultiheadAttention` (factorizing inside it means replacing the
module — a separate experiment). fp32, no autocast; control = `baseline-gpu`.

**Success criterion:** any rank that reaches baseline-level val loss in less
wall-clock wins outright. If none does, the finding is the quality-per-FLOP
curve — and whether the theoretical FLOP cut shows up in wall-clock at all on
this hardware at this tiny scale (it may be overhead-bound; that's a result).

## Runs

```powershell
# Rank sweep on GPU (tags auto: lowrank-r64-gpu, lowrank-r32-gpu, lowrank-r8-gpu)
& "modelEnvGpu\Scripts\python.exe" experiments\03_low_rank\train.py --rank 64
& "modelEnvGpu\Scripts\python.exe" experiments\03_low_rank\train.py --rank 32
& "modelEnvGpu\Scripts\python.exe" experiments\03_low_rank\train.py --rank 8

# Optional parity check: --rank 0 keeps dense MLPs (should reproduce baseline-gpu)
& "modelEnvGpu\Scripts\python.exe" experiments\03_low_rank\train.py --rank 0
```

Results log: `runs.csv` in this folder (includes a `rank` column and the
actual parameter count, which shrinks with rank).

## Notes

- Full rank for a 128×512 matrix is 128, so r=64 is half rank, r=8 is 1/16.
- Param counts drop with rank (the factorization has r·(in+out) weights
  instead of in·out), so this also probes the capacity axis; `est_flops`
  (6·N·T) scales with the reduced N accordingly.
- Foundation for LoRA (next in Phase 2): same factorization idea applied to
  weight *updates* on a frozen full-rank model instead of the weights themselves.
