#!/usr/bin/env python3
"""
Experiment 03: low-rank factorization of the MLP weights vs the fp32 baseline.

Exactly one variable changed from 01_tiny_gpt_baseline: each Block's two MLP
linears (d -> 4d and 4d -> d) are replaced by factorized pairs A@B with rank r
(--rank). FLOPs for a factorized layer drop from in*out to r*(in+out) per
token. Attention stays stock nn.MultiheadAttention — factorizing inside it
would mean replacing the module, which is a separate future experiment.

This trades capacity for speed (unlike exp 02's free lunch), so the deliverable
is the tradeoff curve: val loss and wall-clock across ranks at the same
24.6M-token budget, vs baseline-gpu. --rank 0 keeps dense MLPs (parity check).

Run (GPU, tag auto=lowrank-r<r>-gpu):
  python experiments/03_low_rank/train.py --rank 64
  python experiments/03_low_rank/train.py --rank 32
  python experiments/03_low_rank/train.py --rank 8
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "tinyshakespeare.txt"
LOG_PATH = Path(__file__).parent / "runs.csv"


# ---------------- model (baseline + factorized MLP linears) ----------------

class LowRankLinear(nn.Module):
    """W (in x out) approximated as A (in x r) @ B (r x out); bias on the output."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features)

    def forward(self, x):
        return self.up(self.down(x))


def make_linear(in_features: int, out_features: int, rank: int) -> nn.Module:
    # rank 0 (or >= full rank) keeps the dense baseline layer
    if rank and rank < min(in_features, out_features):
        return LowRankLinear(in_features, out_features, rank)
    return nn.Linear(in_features, out_features)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, rank: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            make_linear(n_embd, 4 * n_embd, rank),
            nn.GELU(),
            make_linear(4 * n_embd, n_embd, rank),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, rank, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout, rank) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        mask = torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------- data (identical to 01_tiny_gpt_baseline) ----------------

def load_data():
    text = DATA_PATH.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], chars


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=20):
    model.eval()
    losses = [model(*get_batch(data, block_size, batch_size, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


# ---------------- training ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--rank", type=int, default=32,
                        help="rank r for the factorized MLP linears; 0 = dense (baseline parity)")
    parser.add_argument("--tag", type=str, default=None,
                        help="label for runs.csv (default: lowrank-r<rank>-<gpu|cpu>)")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="checkpoint path (default: models/<tag>.pt)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"lowrank-r{args.rank}-{'gpu' if device == 'cuda' else 'cpu'}"
    ckpt_path = args.ckpt or REPO_ROOT / "models" / f"{tag}.pt"
    torch.manual_seed(1337)

    train_data, val_data, chars = load_data()
    model = TinyGPT(len(chars), args.block_size, args.n_layer, args.n_head, args.n_embd,
                    args.rank).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} | rank={args.rank} | params={n_params/1e6:.2f}M | vocab={len(chars)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokens_per_iter = args.batch_size * args.block_size
    start = time.perf_counter()
    tokens_seen = 0

    for it in range(1, args.max_iters + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tokens_seen += tokens_per_iter

        if it % args.eval_every == 0 or it == args.max_iters:
            elapsed = time.perf_counter() - start
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, device)
            tps = tokens_seen / elapsed
            print(f"iter {it:5d} | train {loss.item():.4f} | val {val_loss:.4f} "
                  f"| {tps:,.0f} tok/s | {elapsed:,.0f}s elapsed")

    elapsed = time.perf_counter() - start
    val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, device)
    est_flops = 6 * n_params * tokens_seen

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nDone. val_loss={val_loss:.4f} | {tokens_seen:,} tokens in {elapsed:,.0f}s "
          f"({tokens_seen/elapsed:,.0f} tok/s) | est. {est_flops:.2e} training FLOPs")
    print(f"Checkpoint: {ckpt_path}")

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["tag", "device", "rank", "params", "iters", "tokens", "wallclock_s",
                             "tokens_per_s", "val_loss", "est_flops"])
        writer.writerow([tag, device, args.rank, n_params, args.max_iters, tokens_seen,
                         f"{elapsed:.1f}", f"{tokens_seen/elapsed:.0f}", f"{val_loss:.4f}",
                         f"{est_flops:.3e}"])
    print(f"Logged to {LOG_PATH}")


if __name__ == "__main__":
    main()
