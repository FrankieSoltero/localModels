# Research Roadmap

**Mission:** find training/fine-tuning recipes that let small models on a consumer PC
approach enterprise-model quality, by attacking the dominant cost — matrix multiplication.

## Core research questions

1. **Where does the compute actually go?** For a transformer of size N trained on T tokens,
   training cost ≈ 6·N·T FLOPs, and >90% of that is GEMMs (QKV/output projections + MLP).
   Measure this empirically on our hardware, don't assume.
2. **Which matmul-reduction techniques survive contact with real hardware?** Low-bit
   quantization, low-rank factorization, and ternary/matmul-free architectures all cut
   theoretical FLOPs — but only some map to fast kernels on CPU/consumer GPU.
3. **Can fine-tuning + distillation close the gap to enterprise models on narrow tasks?**
   A 1–3B model will not match GPT/Claude broadly; the interesting question is whether it
   can match them on a *specific* task after distilling their outputs.

## Phases

### Phase 0 — Instrumentation ✅ (2026-07-02)
- [x] Run `experiments/00_matmul_bench` — GFLOP/s by matrix size and dtype on the 9900X.
- [x] Get the RX 9060 XT working — PyTorch 2.9.1+rocm7.2.1 in `modelEnvGpu/`, verified.
- [x] Result: cost table in `experiments/00_matmul_bench/RESULTS.md`. Headlines:
      GPU bf16 = 67 TFLOP/s (~20× CPU bf16); fp16 on CPU is emulated and unusable;
      GPU trains the tiny baseline 11.5× faster end-to-end in fp32 (measured).

### Phase 1 — Baseline
- [x] Train `01_tiny_gpt_baseline` on Tiny Shakespeare (fixed budget: 24.6M tokens):
      **val loss 1.8541, 14,516 tok/s, 28 min on CPU** (`runs.csv`, tag `baseline-cpu`).
      Loss still falling at 3000 iters — undertrained by design; all comparisons run at
      this same token budget.
- [x] Rerun the full baseline on GPU (tag `baseline-gpu`): **val loss 1.8634, 146.8 s,
      167,417 tok/s** — quality parity with CPU (Δ0.009 ≈ cross-device kernel noise),
      11.5× faster end-to-end. **This is the primary control for Phase 2+.**
- [ ] Scale check: same model on a slightly larger corpus to see where CPU/GPU tops out.

### Phase 2 — Cheaper matmuls, same architecture
Each gets an `experiments/NN_*` folder benchmarked against Phase 1:
- [x] **Mixed precision** (bf16) — confirmed the free lunch (2026-07-02):
      **GPU 1.74×** (84.6 s vs 146.8 s) and **CPU 1.34×** (1265 s vs 1693 s) at loss
      parity (`experiments/02_mixed_precision/runs.csv`). End-to-end gains lag the
      matmul-level ceilings (8.5×/3.5×) because the 0.84M model is overhead-dominated —
      expect the gap to close at larger scale.
- [ ] **Low-rank factorization** — replace W (d×d) with A·B (d×r · r×d); FLOPs drop ~r/d.
- [ ] **LoRA fine-tuning** — low-rank *updates*: full-model quality, tiny trainable footprint.
- [ ] **QLoRA** — 4-bit frozen base + LoRA; the standard way to fine-tune 3–7B models on
      consumer hardware.
- [ ] **Gradient checkpointing / fused optimizers** — memory tricks that enable bigger batches.

### Phase 3 — Replacing matmul itself
The speculative, most interesting phase:
- [ ] **BitNet b1.58** — ternary weights {-1,0,+1}; matmul becomes add/subtract. Reproduce at
      tiny scale, compare loss-vs-wallclock against Phase 1.
- [ ] **MatMul-free LM** (Zhu et al. 2024) — ternary layers + element-wise recurrence.
- [ ] **Structured matrices** (Monarch/butterfly) — O(n√n) instead of O(n²) per projection.
- [ ] Honest accounting: do these win on *our* hardware, or only on custom silicon?

### Phase 4 — Closing the quality gap
- [ ] Pick one narrow task (e.g., code explanation, structured extraction).
- [ ] Distill an enterprise model: generate training pairs via API, fine-tune Qwen2.5-3B
      with the cheapest recipe that survived Phases 2–3.
- [ ] Blind-evaluate distilled-small vs. enterprise on a held-out set.

## Ground rules

- One variable per experiment; always compare to `01_tiny_gpt_baseline` at equal token budget.
- Log every run: config, val loss, tokens/sec, wall-clock, notes. CSV per experiment folder.
- Negative results get written up too — "ternary nets train 3× slower on CPU" is a finding.
