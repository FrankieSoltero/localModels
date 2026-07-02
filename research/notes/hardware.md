# Hardware Notes — Getting Compute Out of This Machine

## What we have

- **CPU:** AMD Ryzen 9 9900X — 12 cores / 24 threads, Zen 5, full AVX-512. Genuinely good
  for small-model training; AVX-512 + bf16 paths in PyTorch CPU kernels are usable today.
- **GPU:** AMD Radeon RX 9060 XT — RDNA4 (gfx1200-family), 16 GB VRAM. Strong inference
  card; the training story depends entirely on the software stack below.
- **OS:** Windows 11 Home. WSL2 available.
- **Current venv:** Python 3.13, `torch 2.8.0+cpu` → **the GPU is not used at all today.**

## GPU enablement options (in rough order of preference)

### Option A — PyTorch + ROCm natively on Windows  ← CHOSEN (verified 2026-07-02)
**PyTorch 2.9.1 + ROCm 7.2.1 officially supports the RX 9060 XT on Windows 11.**
Confirmed via [AMD's Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
and [install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html).

Requirements and what we did:
- **Python 3.12 exactly** (wheels are cp312) → installed 3.12.10 via winget, created `modelEnvGpu/` venv.
  The original CPU venv `modelEnv/` (Python 3.13) stays untouched for CPU experiments.
- **Adrenalin driver ≥ 26.2.2** → our driver is dated 2026-06-27, newer. OK.
- Install = ROCm SDK wheels (`rocm_sdk_core`, `rocm_sdk_devel`, `rocm_sdk_libraries_custom`)
  then `torch-2.9.1+rocm7.2.1` / `torchvision` / `torchaudio` from repo.radeon.com.
- Caveat from AMD: only PyTorch's ROCm components ship on Windows — the full ROCm stack
  (profilers, bitsandbytes-rocm, etc.) is still Linux-only. If we hit that wall in Phase 2/3,
  fall back to Option B (WSL2).
- GPU shows up through the CUDA API surface: `torch.cuda.is_available()`, `device="cuda"`.

### Option B — WSL2 + ROCm (Linux userspace)
ROCm-on-WSL supports consumer Radeon cards; RDNA4 support landed after RDNA3. Gives the
mature Linux ROCm stack (and access to bitsandbytes-rocm, flash-attention ports, etc.)
without dual-booting. Slightly more friction, much better ecosystem for training.

### Option C — llama.cpp with Vulkan (inference + LoRA-scale finetunes)
Works on basically any GPU including RDNA4, today, on native Windows. Not a general
training platform, but: fast local inference of quantized models, and its ecosystem is
where BitNet-style ternary kernels actually exist (`bitnet.cpp`). Useful for Phase 3
wall-clock measurements of low-bit kernels.

### Option D — torch-directml
DirectML backend for PyTorch on any DX12 GPU. Historically laggy (pinned to old torch
versions, incomplete op coverage, weak training support). Last resort.

### Option E — CPU-only (the honest fallback)
The 9900X can train nanoGPT-scale models (1–20M params) in minutes-to-hours and can run
QLoRA on small models slowly. All Phase 0–2 experiments are designed to be feasible on CPU
so research is never blocked on GPU driver drama.

## Practical sizing (16 GB VRAM, once GPU works)

| Task | Feasible? |
|---|---|
| Pretrain 10–100M param model from scratch | Yes, comfortably |
| Full fine-tune 1–3B (bf16 + AdamW) | Tight — needs grad checkpointing / 8-bit optimizer |
| QLoRA fine-tune 3–8B | Yes — this is the sweet spot |
| QLoRA fine-tune 13B+ | Marginal; offloading required |

## Action items

- [ ] Check AMD docs for current PyTorch-on-Windows wheel status for RX 9060 XT (gfx1200).
- [ ] If native wheel exists: create a second venv (`modelEnvGpu`) so the CPU env stays clean.
- [ ] Else: set up WSL2 + ROCm and mirror the venv there.
- [ ] Rerun `experiments/00_matmul_bench` on GPU; add results to the cost table.
