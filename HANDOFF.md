# HANDOFF — localModels LLM Training-Efficiency Research

> Living document. Update in place; don't recreate. Last updated: **2026-07-02** (second
> session: finished Phase 1 with the GPU baseline, ran experiment 02 mixed-precision on
> both devices — Phase 2 has its first confirmed win).

## 1. Goal & current task

**Project goal:** find training/fine-tuning recipes on this consumer PC (Ryzen 9 9900X +
RX 9060 XT) that let small models approach enterprise-model quality on narrow tasks, by
attacking matrix-multiplication cost. Full plan: `research/ROADMAP.md`.

**Task in flight:** pick and run the next Phase 2 experiment — low-rank factorization
(`03_low_rank`) is the natural next one-variable step; LoRA on Qwen2.5-3B is the bigger
prize but needs `transformers`+`peft` installed into `modelEnvGpu` first.

## 2. Status

- ✅ Phase 0 done — hardware cost table in `experiments/00_matmul_bench/RESULTS.md`.
- ✅ **Phase 1 control complete on both devices** (`01_tiny_gpt_baseline/runs.csv`):
  `baseline-cpu` val 1.8541 / 1693 s; `baseline-gpu` val 1.8634 / **146.8 s** (11.5×).
  GPU baseline is the primary control for Phase 2+ (loss parity confirmed, Δ0.009 = noise).
- ✅ **Experiment 02 (mixed precision) done, hypothesis confirmed**
  (`02_mixed_precision/runs.csv`): `bf16-gpu` **1.74×** (84.6 s, val 1.8639);
  `bf16-cpu` **1.34×** (1264.6 s, val 1.8543). Loss parity everywhere.
- ✅ `train.py --ckpt` fix landed: runs save to `models/<tag>.pt`, no more clobbering.
  (The pre-fix checkpoint keeps its old name `tiny_gpt_baseline.pt` — pass `--ckpt` to load it.)
- ✅ Docs updated: ROADMAP checkboxes, RESULTS.md end-to-end table + takeaways.
- ⏸ Phase 1 scale-check item still open (larger corpus). Phase 2 items 2-5 not started.
- ⚠ Git: still **zero commits** — user hasn't asked for a commit; don't commit without asking.

## 3. Decisions + why

| Decision | Why |
|---|---|
| **User runs all training commands himself** in his terminal, pastes results back | He's learning the workflow hands-on; also keeps benchmarks off Claude's session and the machine quiet |
| GPU `baseline-gpu` = primary control for Phase 2+ | Loss parity with CPU confirmed; 11.5× faster iteration. CPU baseline kept for CPU-specific comparisons |
| Val loss always computed in fp32, even in mixed-precision experiments | Quality numbers stay apples-to-apples across all experiments |
| Experiment 02 uses a bool causal mask (baseline uses float −inf) | Float mask dtype clashes with bf16 activations in `nn.MultiheadAttention` under autocast; bool is numerically equivalent |
| Checkpoints default to `models/<tag>.pt` (`--ckpt` to override) | Each run keeps its artifact; runs stopped overwriting each other |
| Two venvs: `modelEnv/` (Py 3.13, torch-cpu) and `modelEnvGpu/` (Py 3.12, torch-rocm) | ROCm Windows wheels are cp312-only; untouched CPU env preserves clean comparisons |
| Control = fixed 24.6M-token budget, NOT convergence | Efficiency claims are comparisons at equal tokens; baseline deliberately undertrained |
| Wins measured in wall-clock at equal val loss, not theoretical FLOPs | Exp 02 proved it again: bf16's 8.5× matmul ceiling realized only 1.74× end-to-end (Amdahl) |

## 4. Ordered next steps

1. **Create `experiments/03_low_rank/`** — replace each `nn.Linear(d, d)`-ish projection
   with A·B factorization (rank r sweep, e.g. r ∈ {8, 32, 64}); copy of baseline loop,
   one variable changed. Compare vs `baseline-gpu` at 24.6M tokens; log to its own runs.csv.
2. Or (user's call) **install `transformers`+`peft` into `modelEnvGpu`** and start the LoRA
   experiment on Qwen2.5-3B (`inference/qwen_chat.py` proves the model runs locally).
3. Phase 1 scale check (larger corpus) — still open on the roadmap.
4. Consider the **initial git commit** (repo has zero commits; ask user first).
5. Keep `research/ROADMAP.md` checkboxes and memory file in sync as results land.

## 5. Files (touchpoints with line refs)

- `research/ROADMAP.md` — Phase 1 nearly done (scale check open); Phase 2 item 1 checked.
- `experiments/02_mixed_precision/train.py` — autocast experiment; `--device`, `--dtype`,
  auto-tag `<dtype>-<gpu|cpu>`; refuses fp16-on-CPU; GradScaler only for fp16+cuda.
- `experiments/02_mixed_precision/README.md` — hypothesis, method, success criterion.
- `experiments/02_mixed_precision/runs.csv` — smoke / bf16-gpu / bf16-cpu rows.
- `experiments/01_tiny_gpt_baseline/train.py` — now takes `--ckpt`; save path derives from
  `--tag`. Model at `:34-90`, argparse `:123-136`, save `:176-180`.
- `experiments/01_tiny_gpt_baseline/runs.csv` — 4 rows: smoke, baseline-cpu, smoke-gpu,
  baseline-gpu.
- `experiments/00_matmul_bench/RESULTS.md` — cost table + measured end-to-end table.
- `.claude/settings.json` — sets `worktree.bgIsolation: "none"` (see Gotcha #10).
- `models/` — `tiny_gpt_baseline.pt` (old CPU baseline), `baseline-gpu.pt`, `bf16-gpu.pt`,
  `bf16-cpu.pt`, `smoke.pt`.

## 6. Gotchas / constraints

1. **fp16 on CPU is software-emulated (~1000× slow).** CPU work must use bf16 or fp32.
   Experiment 02's script hard-refuses the combination.
2. **ROCm masquerades as CUDA:** `device="cuda"`, `torch.cuda.*` — correct for the AMD GPU.
3. **Float `-inf` attention masks break under autocast** in `nn.MultiheadAttention`
   (dtype mismatch with bf16). Use bool masks in any autocast experiment.
4. **Spurious console error** `Francisco: Unknown command line argument...` on every
   `modelEnvGpu` python start — ROCm's `offload-arch.exe` mishandles the space in
   `C:\Users\Francisco Soltero`. Cosmetic so far; suspect it first if `torch.compile` fails.
5. **`results_gpu.csv` CPU float32 rows are garbage** (benchmark ran while training occupied
   all cores). The table in `RESULTS.md` is authoritative. Never benchmark on a busy machine.
6. SDPA memory-efficient attention on RDNA4 is experimental — enable with env var
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (untested here).
7. `modelEnvGpu` has NO `transformers`/`requests`/`peft` — only torch stack. Tiny-GPT data
   is already at `data/tinyshakespeare.txt`.
8. Python 3.12 was installed user-scope via winget solely for ROCm wheels; system default
   remains 3.14/3.13. Always call venv pythons by explicit path.
9. PowerShell 5.1 on this box: no `&&` — chain with `;` or `if ($?)`.
10. **Background Claude sessions can't edit repo files directly** until a restart picks up
    `.claude/settings.json` (`bgIsolation: none`, created 2026-07-02 with user approval).
    Workaround used this session: Write to job tmp dir, then `Copy-Item` into the repo.
    Worktree isolation is impossible anyway until the repo has its first commit.
11. Smoke runs under-report GPU throughput (warmup unamortized): smoke-gpu logged 138k tok/s
    vs 167k sustained. Only full-budget runs are quotable.

## 7. Open questions

- **Initial git commit** — repo has zero commits; user hasn't said whether/when to commit or
  whether this should get a GitHub remote.
- **Next experiment: 03 low-rank (self-contained) or LoRA (needs installs)?** User's call.
- Phase 4 distillation will need an enterprise-model API budget — unasked.

## 8. Resume & verify

```powershell
# Verify GPU stack (expect: True + "AMD Radeon RX 9060 XT"; ignore the
# "Francisco: Unknown command line argument" line — Gotcha #4)
& "modelEnvGpu\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Results so far (expect 4 data rows: smoke / baseline-cpu / smoke-gpu / baseline-gpu)
Get-Content experiments\01_tiny_gpt_baseline\runs.csv

# (expect 3 data rows: smoke / bf16-gpu / bf16-cpu)
Get-Content experiments\02_mixed_precision\runs.csv
```

- Branch: `main`, no commits yet, no remote.
- **User preference: he runs training commands in his own terminal and pastes results.**
  Hand over ready-to-paste PowerShell 5.1 commands with explicit venv paths.
- Session memory: `~\.claude\projects\C--Users-Francisco-Soltero-Desktop-localModels\memory\`
  (`llm-training-efficiency-research.md`, `user-runs-commands-himself.md`) — updated 2026-07-02.
- Plans/specs: `research/ROADMAP.md` is the spec; this file is the resume packet.
