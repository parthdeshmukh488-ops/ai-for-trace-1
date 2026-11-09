"""Did the dictionary find the features, and do the usual proxies predict it?

Reconstruction loss and ``L0`` are what dictionaries are normally judged on,
and neither is the claim.  The claim is that the learned directions *are* the
network's features, and in the planted regime that is directly checkable.

Three ways of checking it, because the obvious one flatters the dictionary:

* **Mean max cosine, true to learned.**  For each true feature, how well does
  its best match among the learned directions align with it.  This is the
  usual number, and it is gameable: a dictionary that learns every true
  feature *and a hundred spurious ones* scores perfectly.
* **Mean max cosine, learned to true.**  The reverse direction, which the
  spurious features ruin.  Reporting only the first is how a bloated
  dictionary passes.
* **Optimally matched mean cosine.**  One learned direction per true feature,
  assigned to maximise the total -- so neither duplicates nor spares help.
  This is the strictest of the three and the one to quote if only one is
  quoted.  Computed by optimal assignment rather than greedily, because a
  greedy match is biased upward.

Recovery is measured only over the features the planted model *represents*.  A
feature the model gave up on -- a near-zero column -- has no direction in the
hidden space, so no dictionary can recover it, and counting it as a miss would
measure the planted model rather than the dictionary.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["downstream_kl", "frontier", "recovery"]


def _cosine_matrix(learned: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    a = learned / learned.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    b = true / true.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return a @ b.T  # [n_learned, n_true]


def recovery(
    learned: torch.Tensor,
    true: torch.Tensor,
    *,
    represented: torch.Tensor | None = None,
    signed: bool = False,
) -> dict:
    """Compare learned directions against a known feature basis.

    ``signed=False`` takes the absolute cosine, because a dictionary that
    learns the negative of a feature has still found it -- the sign is absorbed
    into the activation.  Set it true to require the sign as well.
    """
    if learned.ndim != 2 or true.ndim != 2:
        raise ValueError("both arguments must be [n_features, d]")
    if learned.shape[-1] != true.shape[-1]:
        raise ValueError(
            f"dimension mismatch: learned directions are {learned.shape[-1]}-d, "
            f"true ones {true.shape[-1]}-d"
        )
    if represented is not None:
        true = true[represented]
    cosine = _cosine_matrix(learned.double(), true.double())
    if not signed:
        cosine = cosine.abs()

    true_to_learned = cosine.max(dim=0).values
    learned_to_true = cosine.max(dim=1).values

    from scipy.optimize import linear_sum_assignment

    cost = -cosine.cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    matched = cosine.cpu().numpy()[rows, cols]

    return {
        "n_true": int(true.shape[0]),
        "n_learned": int(learned.shape[0]),
        "mmcs_true_to_learned": float(true_to_learned.mean()),
        "mmcs_learned_to_true": float(learned_to_true.mean()),
        "matched_mean_cosine": float(np.mean(matched)),
        "n_matched": len(matched),
        # How many true features have *some* good match at all.
        "fraction_true_above_0.9": float((true_to_learned > 0.9).double().mean()),
        "fraction_true_above_0.7": float((true_to_learned > 0.7).double().mean()),
        # And how many learned features correspond to nothing true.
        "fraction_learned_below_0.5": float((learned_to_true < 0.5).double().mean()),
    }


def frontier(reports) -> dict:
    """Reconstruction against measured L0, which is the only fair comparison.

    Architectures differ in how sparsity is obtained, so comparing their MSE
    at their default settings compares the settings.  The frontier -- MSE as a
    function of the sparsity actually achieved -- is what makes them
    commensurable.
    """
    points: dict[str, list] = {}
    for report in reports:
        points.setdefault(report.kind, []).append(
            {
                "l0": report.final_l0,
                "mse": report.final_mse,
                "explained_variance": report.final_explained_variance,
                "dead_fraction": report.dead_fraction,
                "expansion": report.config.expansion,
                "k": report.config.k,
                "lambda_sparsity": report.config.lambda_sparsity,
            }
        )
    for kind in points:
        points[kind].sort(key=lambda p: p["l0"])
    return points


def downstream_kl(sae, model, site: str, tokens: torch.Tensor, *, scale: float = 1.0) -> dict:
    """Splice the reconstruction back in and measure what the model notices.

    The metric that matters and the one most likely to disagree with MSE: a
    dictionary can reconstruct the residual stream well in Euclidean terms and
    still destroy the next-token distribution, because the directions the model
    reads are not the directions with the most variance.
    """
    from ..model.hooks import capture, intervene

    with torch.inference_mode():
        reference = model(tokens)
    with capture(model, [site]) as store, torch.inference_mode():
        model(tokens)
    original = store[site]

    def replace(tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(-1, tensor.shape[-1]) * scale
        with torch.inference_mode():
            rebuilt = sae(flat.to(next(sae.parameters()).dtype)).reconstruction
        return (rebuilt / scale).reshape(tensor.shape).to(tensor.dtype)

    with intervene(model, {site: replace}), torch.inference_mode():
        spliced = model(tokens)

    ref = reference.log_softmax(dim=-1)
    got = spliced.log_softmax(dim=-1)
    kl = (ref.exp() * (ref - got)).sum(dim=-1)
    variance = (original - original.mean(dim=(0, 1))).pow(2).sum(-1).mean()
    with torch.inference_mode():
        rebuilt = sae(original.reshape(-1, original.shape[-1]) * scale).reconstruction / scale
    mse = (original.reshape(-1, original.shape[-1]) - rebuilt).pow(2).sum(-1).mean()
    return {
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "mse": float(mse),
        "explained_variance": 1.0 - float(mse / variance.clamp(min=1e-12)),
        "site": site,
    }
