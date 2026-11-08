"""Reading and changing what a transformer is doing, through the public API only.

Four things, in the order you would want them: look at an attention pattern,
check that the residual stream really is the sum of what was written into it,
ablate a head and see what it was worth, and confirm that nothing you did
left a mark on the model.

    python examples/01_hooks_and_patching.py
"""

from __future__ import annotations

import torch

from circuitry import ModelConfig, Transformer, capture, intervene, sites
from circuitry.model.hooks import replace_at, scale

# Nothing here needs gradients, and leaving them on makes every print below
# carry a tensor that is still attached to a graph.
torch.set_grad_enabled(False)

cfg = ModelConfig(n_layers=3, n_heads=4, d_model=32, d_mlp=64, d_vocab=50, n_ctx=16)
model = Transformer(cfg, generator=torch.Generator().manual_seed(0)).eval()
tokens = torch.randint(0, cfg.d_vocab, (2, 8), generator=torch.Generator().manual_seed(1))

print(f"a {cfg.n_layers}-layer model with {len(sites.all_sites(cfg))} addressable sites\n")

# ---------------------------------------------------------------------------
# 1. look at an attention pattern
# ---------------------------------------------------------------------------
with capture(model, "blocks.0.attn.pattern") as store:
    model(tokens)
pattern = store["blocks.0.attn.pattern"]
print(f"layer 0 patterns: {tuple(pattern.shape)}  [batch, head, query, key]")
print("head 0, final query, attention over keys:")
print("  " + "  ".join(f"{w:.2f}" for w in pattern[0, 0, -1].tolist()))
print(f"  rows sum to {float(pattern[0, 0, -1].sum()):.6f}, and nothing above the diagonal:"
      f" {float(pattern[:, :, 0, 1:].abs().max()):.1e}\n")

# ---------------------------------------------------------------------------
# 2. the residual stream is the sum of what was written into it
# ---------------------------------------------------------------------------
terms = model.residual_decomposition(tokens)
final = terms[f"blocks.{cfg.n_layers - 1}.resid_post"]
error = float((terms["sum"] - final).abs().max())
print(f"embed + pos + sum(attn.out) + sum(mlp.out) == resid_post, to {error:.1e}\n")

# ---------------------------------------------------------------------------
# 3. ablate one head and measure what it was worth
# ---------------------------------------------------------------------------
baseline = model(tokens)
metric = (baseline[:, -1, 0] - baseline[:, -1, 1]).mean()  # an arbitrary logit difference
print(f"baseline logit difference: {float(metric):+.4f}")
print("effect of zero-ablating each head at layer 1:")
for head in range(cfg.n_heads):
    hook = replace_at("blocks.1.attn.z", {"head": head}, 0.0)
    with intervene(model, {"blocks.1.attn.z": hook}):
        out = model(tokens)
    ablated = (out[:, -1, 0] - out[:, -1, 1]).mean()
    print(f"  head 1.{head}: {float(ablated):+.4f}   change {float(ablated - metric):+.4f}")

with intervene(model, {"blocks.2.mlp.out": scale(0.0)}):
    out = model(tokens)
print(f"  mlp 2   : {float((out[:, -1, 0] - out[:, -1, 1]).mean()):+.4f}\n")

# ---------------------------------------------------------------------------
# 4. none of that left a mark
# ---------------------------------------------------------------------------
print("model restored bitwise after every intervention:",
      torch.equal(model(tokens), baseline))
