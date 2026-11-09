"""Three ways to ask what a component is worth, and what the cheap ones cost.

Exact patching re-runs the model once per component and is the definition of
the effect.  Attribution patching estimates every component at once from a
single backward pass.  Integrated gradients averages the gradient along the
path between the two activations.

Running all three side by side at *two granularities* shows the thing that
matters, which no single number does: a first-order expansion is a statement
about small displacements, and how small the displacement is depends on how
much you patch at once.  Patching one head at one position moves the residual
stream a little and the estimate is decent.  Pooling a head's whole
contribution across every position moves it a lot, and the same estimate gets
nearly half the signs wrong.

    python examples/03_attribution_and_its_error.py
"""

from __future__ import annotations

import torch

from circuitry import ModelConfig, Transformer, patch
from circuitry.attrib.gradients import attribute
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.patching import default_granularity

torch.set_grad_enabled(False)

gen = torch.Generator().manual_seed(1)
cfg = ModelConfig(
    n_layers=4, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8, dtype="float64"
)
model = Transformer(cfg, generator=gen).eval()
clean = torch.randint(0, cfg.d_vocab, (8, 6), generator=gen)
corrupt = torch.randint(0, cfg.d_vocab, (8, 6), generator=gen)
pair = PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "demo")

METHODS = (
    ("first order", {"method": "linear"}),
    ("integrated, 8 steps", {"method": "integrated", "n_steps": 8}),
    ("integrated, 32 steps", {"method": "integrated", "n_steps": 32}),
)


def compare(granularity, title: str) -> None:
    shared = {
        "pair": pair, "sites": "blocks.*.attn.z", "direction": "denoise",
        "granularity": granularity,
    }
    exact = patch(model, **shared)
    scale = max(abs(v) for v in exact.effects.values())
    big = [k for k in exact.effects if abs(exact.effects[k]) > 0.05 * scale]
    print(f"{title}: {len(exact)} effects, largest {scale:.4f}")
    print(f"  {'method':22s} {'median err':>11s} {'worst err':>10s} {'sign wrong':>11s} {'cost':>7s}")
    rows = [("exact patching", exact.effects, exact.forward_passes * pair.batch)]
    for label, kwargs in METHODS:
        result = attribute(model, **shared, **kwargs)
        rows.append((label, result.effects, result.batch_items))
    for label, effects, cost in rows:
        errors = sorted(abs(effects[k] - exact.effects[k]) for k in exact.effects)
        median = errors[len(errors) // 2] / scale
        worst = errors[-1] / scale
        wrong = sum(
            1 for k in big if (effects[k] > 0) != (exact.effects[k] > 0)
        ) / max(len(big), 1)
        print(f"  {label:22s} {median:11.4f} {worst:10.4f} {wrong:11.1%} {cost:7d}")
    print()


compare(default_granularity(pos="each"), "one head at one position (a small displacement)")
compare(default_granularity(pos="all"), "a head's whole contribution (a large one)")

print("Errors are fractions of the largest true effect; cost is batch items processed.")
print("Two things worth taking from the tables. The first-order estimate degrades")
print("sharply as the displacement grows -- which is exactly the regime a circuit")
print("lives in, since a circuit is made of the components whose effects are large.")
print("And integrated gradients costs more than simply being exact, at every step")
print("count, because both need one evaluation per index and it needs n of them.")
