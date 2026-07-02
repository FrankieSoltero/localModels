# localModels — Small-Model Training Efficiency Research

Research area for training and fine-tuning small language models on consumer PC hardware,
with the goal of approaching enterprise-model performance at a fraction of the compute cost.
The central research question: **can we reduce the dominant cost of training — matrix
multiplication — through quantization, low-rank methods, or matmul-free architectures?**

## Hardware profile

| Component | Spec | Status |
|---|---|---|
| CPU | AMD Ryzen 9 9900X (12C/24T) | PyTorch 2.8 (CPU), bf16 fast path via AVX-512 |
| GPU | AMD Radeon RX 9060 XT (RDNA4) | **Working** — PyTorch 2.9.1 + ROCm 7.2.1 on Windows |
| OS | Windows 11 Home | Install notes in `research/notes/hardware.md` |

## Results so far (2026-07-02)

Tiny GPT (0.84M params, char-level Shakespeare), fixed 24.6M-token budget:

| Run | dtype | wall-clock | val loss | speedup |
|---|---|---:|---:|---|
| CPU baseline | fp32 | 1,693 s | 1.8541 | — |
| CPU + autocast | bf16 | 1,265 s | 1.8543 | 1.34× |
| GPU baseline | fp32 | 147 s | 1.8634 | 11.5× vs CPU |
| GPU + autocast | bf16 | 85 s | 1.8639 | 1.74× vs GPU fp32 |

Full cost tables: `experiments/00_matmul_bench/RESULTS.md`. Headline lessons: fp16 on this
CPU is software-emulated (~1000× slow — never use it there), and end-to-end gains lag
matmul-level ceilings on tiny models because overhead dominates (Amdahl's law).

## Layout

```
localModels/
├── README.md                  ← you are here
├── HANDOFF.md                 ← living session-to-session resume packet
├── research/
│   ├── ROADMAP.md             ← research questions, phases, success metrics
│   └── notes/
│       ├── matmul-efficiency.md   ← survey: where FLOPs go + techniques to cut them
│       └── hardware.md            ← ROCm-on-Windows enablement for RDNA4
├── experiments/
│   ├── 00_matmul_bench/       ← measured GEMM throughput by device and dtype
│   ├── 01_tiny_gpt_baseline/  ← trainable small GPT (the control group for everything)
│   └── 02_mixed_precision/    ← bf16 autocast vs the fp32 baseline (first confirmed win)
├── inference/
│   └── qwen_chat.py           ← interactive chat with Qwen2.5-3B-Instruct
├── Docs/                      ← audit reports and project documents
├── data/                      ← datasets (gitignored)
├── models/                    ← checkpoints (gitignored)
├── modelEnv/                  ← Python 3.13 CPU venv (gitignored)
└── modelEnvGpu/               ← Python 3.12 ROCm venv (gitignored)
```

## Quickstart

Two virtual environments (ROCm Windows wheels require Python 3.12; the CPU env stays
untouched for clean comparisons):

```powershell
# CPU work
& "modelEnv\Scripts\python.exe" experiments\00_matmul_bench\matmul_bench.py
& "modelEnv\Scripts\python.exe" experiments\01_tiny_gpt_baseline\train.py

# GPU work (RX 9060 XT via ROCm — shows up as device="cuda")
& "modelEnvGpu\Scripts\python.exe" experiments\01_tiny_gpt_baseline\train.py --tag baseline-gpu
& "modelEnvGpu\Scripts\python.exe" experiments\02_mixed_precision\train.py

# Chat with a real small model (CPU env; downloads Qwen2.5-3B on first run)
& "modelEnv\Scripts\python.exe" inference\qwen_chat.py
```

The training script downloads ~1MB of Shakespeare on first run (pinned to a fixed upstream
commit). Each run appends to its experiment's `runs.csv` and saves its checkpoint to
`models/<tag>.pt`.

## Methodology

Every efficiency idea gets its own numbered folder under `experiments/`, compared against
`01_tiny_gpt_baseline` on the same data with the same token budget. Report at minimum:
**final validation loss, wall-clock time, tokens/sec, and estimated FLOPs**. An idea "wins"
if it reaches the same loss in less wall-clock time (or less energy), not just fewer
theoretical FLOPs — theoretical savings that hardware can't exploit don't count.

## License & attribution

Project code is released under the [MIT License](LICENSE).

Third-party assets (referenced at runtime, never redistributed in this repo):

- **Tiny Shakespeare corpus** — public-domain Shakespeare text, fetched from
  [karpathy/char-rnn](https://github.com/karpathy/char-rnn) (pinned commit).
- **Qwen2.5-3B-Instruct** — built with Qwen. The 3B variant is distributed by Alibaba
  under the [Qwen Research license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct);
  weights are downloaded from Hugging Face at runtime and are not included here.
