# Style-Tuning Engine — Design

**Date:** 2026-08-06 · **Status:** approved by owner · **Supersedes:** nothing (extends `research/ROADMAP.md` Phase 2 LoRA/QLoRA items)

## 1. Goal

Build a reusable, fully-local fine-tuning **engine**: harvest → dataset → tune → eval → compare, driven by per-run YAML configs. First application: tune coding models toward the owner's personal coding style/stack. Later applications on the same engine: larger coding models (7B via QLoRA) and reasoning models.

"Better performance" is defined per Section 5's win condition — never by theoretical FLOPs, consistent with repo methodology.

## 2. Constraints & context

- Hardware: Ryzen 9 9900X + RX 9060 XT (16 GB, RDNA4), Windows 11. PyTorch 2.9.1 + ROCm 7.2.1 native (GPU appears as `device="cuda"`).
- **Fully local** — no paid API calls anywhere in the pipeline (owner decision).
- 4-bit QLoRA requires `bitsandbytes`, which on ROCm is Linux-only → 7B QLoRA requires WSL2 + ROCm (Stage 2). Native Windows supports bf16 LoRA today.
- bf16 autocast is the one confirmed training win (experiment 02); all training runs use it.
- Dev process: **code-by-hand mode** — the owner types every line of engine code; the agent navigates, verifies, and writes only `.code-by-hand.md`. Owner runs all training/pip commands in his own terminal.

## 3. Architecture

New top-level package `engine/` (infrastructure, not an experiment) + `configs/` + experiment folder `experiments/04_style_tune/` (hypothesis, runs.csv — keeps the numbered-experiment methodology). Each stage is an independently runnable script whose output is an inspectable file.

```
engine/
├── harvest.py    # extract code units from local repo clones
├── dataset.py    # backtranslation pairs, augmentation, splits
├── tune.py       # PEFT LoRA/QLoRA trainer, config-driven
├── eval.py       # local eval suite (Section 5)
└── report.py     # append runs.csv, compare adapter vs base
configs/
├── coder3b-lora.yaml    # Stage 1
└── coder7b-qlora.yaml   # Stage 2
```

Data flow: `local repo clones → data/style_corpus/raw.jsonl → {train,val,test}.jsonl → models/adapters/<tag>/ → experiments/04_style_tune/runs.csv`. All corpus data and adapters live under gitignored `data/` and `models/`.

A run config names: base model, adapter type (LoRA/QLoRA) + rank/alpha/dropout, dataset path + sampling weights, device, dtype, batch/seq-len/grad-accum/checkpointing, tag.

## 4. Dataset

**Sources.** The owner's local/GitHub repos (final list chosen at harvest time in a config; **`greystonedb` is explicitly denylisted**) plus a shortlist of well-maintained open-source repos in the same stack (Python + TypeScript; MIT/Apache-licensed only, license verified before harvest).

**Harvest** (`harvest.py`): extract complete logical units (functions, methods, classes with docstrings) plus file-level context. Filters: size bounds, no vendored/generated code, near-duplicate removal, and a **secrets scan** (regex for keys/tokens) — flagged units are dropped and reported, never written to the corpus.

**Backtranslation** (`dataset.py`): a local model (existing Qwen chat model or the Coder base) generates, for each harvested unit, the *instruction* that would have produced it. Instruction = prompt; the **unedited real code = completion**. Training targets are therefore always authentic owner code; the local model only generates the cheap side. Junk instructions filtered by length/relevance checks; a sample is human-reviewed before the first training run.

**Augmentation:** OSS-harvested pairs are restyled mechanically (owner's ruff/black/prettier/eslint configs) and sampled at **lower weight** than owner-repo pairs. Owner code is the style anchor.

**Splits:** held out by **whole repo**, never random rows — the test split contains only repos absent from training, so held-out loss measures style generalization, not memorization. `dataset.py` prints corpus stats (pairs per repo/language) before training.

## 5. Evaluation (fully local)

Per run, three signals appended to `runs.csv` alongside wall-clock, tokens/sec, VRAM peak:

1. **Held-out style loss (primary):** cross-entropy on the test split (owner code from unseen repos), tuned vs base.
2. **Capability regression guard:** pass@1 on a fixed ~100-problem local execution benchmark (MBPP/HumanEval+-style; generated code runs against tests in a sandboxed subprocess). Tuned model must stay within 2 absolute points of base pass@1.
3. **Style conformance (secondary):** completions for a fixed prompt set scored for mechanical conformance against the owner's lint/format configs.

**Win condition:** held-out style loss improves over base **and** the regression guard holds.

## 6. Staging

- **Stage 1 — validate the engine (native Windows):** Qwen2.5-Coder-3B, bf16 LoRA. Prereq: `transformers`, `peft`, `datasets` installed into `modelEnvGpu` (owner runs pip). Fast iteration loop for building the engine itself; smoke datasets keep code-by-hand cycles short.
- **Stage 2 — scale to 7B (WSL2):** WSL2 + ROCm + bitsandbytes venv; flip config to Qwen2.5-Coder-7B QLoRA. Runs as a parallel infra track — never gates Stage 1.
- **Stage 3 — reasoning models (later, out of scope for this spec's implementation plan):** same engine, new dataset/eval config (e.g. a DeepSeek-R1 distill). Requires its own eval design.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| bitsandbytes broken on WSL2 + RDNA4 | Engine already validated on 3B; fallback = 7B bf16 LoRA + aggressive grad checkpointing, or stay on 3B |
| Corpus too small / skewed to one repo | OSS augmentation, sampling weights, corpus stats reported before training |
| Low-quality backtranslated instructions | Filter pass + human review of a sample before first run |
| VRAM OOM at 7B | Batch/seq-len/grad-accum/checkpointing all in config; VRAM peak logged per run |
| Secrets or excluded code in corpus | Harvest-time secrets scan; explicit repo denylist; corpus is gitignored |

## 8. Out of scope

- Inference/serving engine (llama.cpp/GGUF export) — possible later phase, not part of this design.
- Test-time compute scaffolds (best-of-N, self-consistency).
- Any paid-API dataset generation or LLM-as-judge eval (revisit only if the owner changes the fully-local decision).
- Stage 3 reasoning-model specifics.
