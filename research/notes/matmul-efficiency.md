# Where the FLOPs Go, and How to Cut Them

Working notes on reducing the matrix-multiplication cost of training transformers.

## 1. The cost structure of a transformer

For a decoder layer with hidden size `d`, sequence length `s`, batch `b`:

| Operation | FLOPs (fwd) | Share at d=768, s=1024 |
|---|---|---|
| QKV + output projections | 8·b·s·d² | ~25% |
| Attention scores + weighted sum | 4·b·s²·d | ~10% (grows with s²) |
| MLP (4× expansion) | 16·b·s·d² | ~50% |
| LayerNorm, softmax, residuals | O(b·s·d) | <5% |

Backward pass ≈ 2× forward. Rule of thumb: **training ≈ 6·N FLOPs per token** (N = params).
Almost all of it is GEMMs — so "make training cheaper" ≈ "make GEMMs cheaper or fewer."

Three levers, in increasing order of ambition:

## 2. Lever 1 — Make each matmul cheaper (precision)

- **Mixed precision (bf16/fp16):** same FLOP count, but 2× less memory traffic and access to
  faster hardware paths. On CPU (AVX-512 on Zen 5) and RDNA4 this is the first thing to try.
- **8-bit training (FP8 / INT8):** enterprise labs train in FP8 on H100s; consumer hardware
  mostly lacks fast FP8 GEMM, but INT8 inference kernels exist (llama.cpp, bitsandbytes).
- **4-bit (QLoRA):** freeze base weights in NF4, train LoRA adapters in bf16. Dettmers et al.
  2023. The workhorse of consumer-hardware fine-tuning — this is how a 9060 XT-class GPU
  fine-tunes a 3–7B model at all.

## 3. Lever 2 — Do fewer matmul FLOPs (structure)

- **LoRA (Hu et al. 2021):** W stays frozen; train ΔW = A·B with rank r ≪ d. Cuts *trainable*
  FLOPs and optimizer memory massively; forward cost unchanged. For fine-tuning, not pretraining.
- **Low-rank factorization of W itself:** replace d×d with (d×r)(r×d) → FLOPs scale by 2r/d.
  Quality drops if r is too small; interesting to measure the loss-vs-FLOPs frontier.
- **Structured matrices — Monarch / butterfly (Dao et al.):** O(n√n) matmuls with dense-like
  expressiveness. Caveat: needs good block-sparse kernels to realize the win on real hardware.
- **Mixture of Experts:** more parameters, same FLOPs/token — sparsity via routing. Probably
  overkill at our scale but worth a note.
- **FlashAttention:** reduces attention *memory traffic*, not FLOPs. Matters once s is large;
  irrelevant to the MLP-dominated regime of small models at short context.

## 4. Lever 3 — Eliminate multiplication (the research frontier)

- **BitNet b1.58 (Ma et al., Microsoft 2024, arXiv:2402.17764):** weights constrained to
  {-1, 0, +1} (log₂3 ≈ 1.58 bits). W·x becomes additions/subtractions — no multiplies.
  Claims parity with fp16 transformers from ~3B params when *trained from scratch* (doesn't
  work as post-hoc quantization). Training still keeps fp shadow weights, so the training-time
  win on stock hardware is limited — the big win is inference (see `bitnet.cpp`).
- **MatMul-free LM (Zhu et al. 2024, arXiv:2406.02528):** ternary "BitLinear" everywhere +
  replaces self-attention with an element-wise GRU-style recurrence (MLGRU). Reports
  competitive scaling up to 2.7B and large memory savings with a fused GPU implementation.
- **Honest framing:** on our hardware, ternary ops don't get magic kernels — PyTorch will
  still emulate them with float ops. The experiment is: (a) verify the *quality* claims at
  tiny scale, (b) measure whether any *wall-clock* win exists via integer kernels (CPU
  AVX2/AVX-512 int8, or llama.cpp-style kernels), (c) quantify the gap between theoretical
  and realized savings.

## 5. Lever 4 — Need fewer tokens/params in the first place

Orthogonal to matmul cost but multiplies with it:

- **Knowledge distillation:** train the small model on an enterprise model's outputs
  (sequence-level distillation). This is Phase 4 of the roadmap and likely the highest-leverage
  path to "enterprise-like performance on a narrow task."
- **Better data > more data:** phi-family results ("Textbooks Are All You Need") show curated
  synthetic data lets 1–3B models punch far above their weight.
- **Muon / second-order-ish optimizers:** recent optimizer work (Muon, used in NanoGPT
  speedruns and Kimi K2) reaches target loss in ~35–48% fewer tokens than AdamW at small
  scale — fewer steps means fewer matmuls, no architecture change. Cheap to test on our baseline.

## 6. Reading list

- Hu et al., *LoRA*, 2021 — arXiv:2106.09685
- Dettmers et al., *QLoRA*, 2023 — arXiv:2305.14314
- Ma et al., *The Era of 1-bit LLMs (BitNet b1.58)*, 2024 — arXiv:2402.17764
- Zhu et al., *Scalable MatMul-free Language Modeling*, 2024 — arXiv:2406.02528
- Dao et al., *Monarch: Expressive Structured Matrices*, 2022 — arXiv:2204.00595
- Jordan et al., *Muon optimizer* — github.com/KellerJordan/Muon
- Karpathy, *nanoGPT* — github.com/karpathy/nanoGPT (our baseline's ancestor)
