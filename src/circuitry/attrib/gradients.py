"""First-order estimates of a patching effect, and the two fixes for them.

Exact patching re-runs the forward pass once per index.  Attribution patching
estimates every index at once from a single backward pass, by taking the first
term of a Taylor expansion:

    effect(a) = M(base with a <- a') - M(base)
              ~= <a' - a, dM/da |_base>

The approximation is exactly as good as the metric is locally linear in that
activation, and the places it breaks are the interesting ones: saturated
softmaxes, where the gradient is near zero while the intervention moves the
argmax; LayerNorm's normalisation; and any node whose effect is large, which
is to say the nodes that end up in a circuit.

Three things in this module that are easy to get wrong.

**The gradient is taken on the base run, not on the clean run.**  Denoising
starts from the corrupt prompt, so its expansion point is the corrupt
activations and its gradient is ``dM/da`` evaluated there.  Using the clean
gradient for a denoising estimate is a plausible-looking error that produces
numbers of the right order and the wrong sign on about half the nodes.

**Integrated gradients is batched over its own steps.**  Averaging the
gradient along the straight line from the base activation to the patched one
needs ``n`` evaluations, but they are independent, so replicating the prompt
batch ``n`` times and interpolating differently in each replica gets all of
them from one forward-backward pair.  The cost is still ``n`` times the work --
which is why the compute axis in phase 4 counts *batch items processed* rather
than forward passes, since the latter would make this look free.

**Interpolation is per index, not per site.**  This is the detail that makes
or breaks the method.  A patching effect is defined for one index -- one head
at one position -- so the path integrated along must be the one where *only
that index* moves from its base value to its patched value.  Interpolating the
whole site while taking the dot product at one index integrates along a
different path entirely, and the result does not converge to the effect: it
converges, but to the wrong number, plateauing at a large error that more
steps will not reduce.  Measured during development: a plateau at 0.30 worst
error against effects of order 0.9, stable from 8 steps to 128.

Per-index interpolation also means the cost is ``n_steps`` times the number of
indices, which is the honest figure and the reason exact patching is often the
cheaper option outright.  Nothing about this method is free; the compute axis
in phase 4 exists to say so.

The fixes
---------

*Softmax recomputation.*  For ``q``, ``k`` and ``scores``, the first-order
term is evaluated on the wrong side of a saturating nonlinearity.  But the
pattern can be recomputed exactly from a patched score -- it is one softmax,
no forward pass -- so the expansion can be moved past it: take the *exact*
change in the attention pattern and expand from there.  This costs nothing and
is expected to help on exactly those three site families and nowhere else.

*Gradient dropout.*  Cancellation between downstream paths can make an
estimate small for the wrong reason: two routes with opposite signs sum to
nearly nothing at first order while either alone would matter.  So for each
layer, the gradient flowing back through that layer's component outputs is cut,
every attribution is recomputed, and the largest magnitude per site over those
variants is taken.  Costs one extra backward pass per layer.

What that does and does not detect is worth being precise about, because this
is a variant of a published idea rather than the published idea itself.  For a
site *downstream* of the cut layer the gradient is unchanged, so the variant
adds nothing.  For a site *upstream* of it the gradient becomes "the gradient
along every route that does not pass through that layer's components", which is
exactly the probe wanted: if the full gradient was small because two such
routes cancelled, one of these variants will be large.  For a site *in* the cut
layer the gradient is zeroed, so the variant is uninformative -- and harmlessly
ignored, since zero never beats a non-zero magnitude.

Both are implemented separately so phase 4 can report what each one
contributes, rather than reporting a combined method and leaving the reader to
guess which half did the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..model import sites as _sites
from ..model.hooks import intervene
from .metric import Baselines, Direction, Metric, PromptPair, measure_baselines
from .patching import Counterfactual, _index_list, default_granularity

__all__ = [
    "AttributionResult",
    "attribute",
]


@dataclass(frozen=True)
class AttributionResult:
    """Estimated effects, with the cost it took and which fixes were applied."""

    effects: dict[tuple[str, tuple[tuple[str, int], ...]], float]
    direction: str
    counterfactual: str
    method: str
    fixes: tuple[str, ...]
    forward_passes: int
    backward_passes: int
    batch_items: int
    n_steps: int = 1
    baselines: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.effects)

    def top(self, n: int = 10):
        items = [(site, index, value) for (site, index), value in self.effects.items()]
        return sorted(items, key=lambda t: -abs(t[2]))[:n]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "fixes": list(self.fixes),
            "direction": self.direction,
            "counterfactual": self.counterfactual,
            "n_steps": self.n_steps,
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "batch_items": self.batch_items,
            "baselines": self.baselines,
            "effects": {
                f"{site}|{'.'.join(f'{k}={v}' for k, v in index)}": value
                for (site, index), value in sorted(self.effects.items())
            },
        }


def _run_with_gradients(
    model,
    tokens: torch.Tensor,
    metric: Metric,
    names,
    *,
    patches: dict | None = None,
    drop_layer: int | None = None,
    replicas: int = 1,
    batch: int | None = None,
):
    """One forward-backward pass, returning the activations and their gradients.

    ``drop_layer`` suppresses the gradient arriving at that layer's component
    outputs, which is how gradient dropout probes for cancellation.

    ``replicas`` is for the integrated-gradient sweep, which stacks several
    interpolation points into one batch.  The metric is evaluated once per
    replica rather than once over the whole stack: a metric closes over
    per-item token ids of the original batch size, so handing it an ``n``-times
    larger batch is a shape error -- and the obvious fix of tiling the ids
    would push the problem into every metric anyone writes later.
    """
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def recorder(name: str):
        def hook(tensor: torch.Tensor):
            if patches is not None and name in patches:
                tensor = patches[name]
            activations[name] = tensor
            if tensor.requires_grad:
                tensor.register_hook(lambda grad, n=name: gradients.__setitem__(n, grad.detach()))
            return tensor if patches is not None and name in patches else None

        return hook

    hooks: dict[str, list] = {name: [recorder(name)] for name in names}
    if drop_layer is not None:
        for site in (f"blocks.{drop_layer}.attn.z", f"blocks.{drop_layer}.mlp.out"):
            # Appended *after* any recorder, not instead of it.  Skipping the
            # cut when the site happened to be one being measured -- which an
            # earlier version did -- meant that on a full sweep, where every
            # site is measured, gradient dropout cut nothing at all and the
            # method silently did nothing.
            hooks.setdefault(site, []).append(_gradient_blocker())
    was = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        with intervene(model, hooks):
            logits = model(tokens)
            if replicas == 1:
                total = metric(logits).sum()
            else:
                size = batch if batch is not None else logits.shape[0] // replicas
                total = sum(
                    metric(logits[k * size : (k + 1) * size]).sum() for k in range(replicas)
                )
            total.backward()
    finally:
        torch.set_grad_enabled(was)
        model.zero_grad(set_to_none=True)
    return activations, gradients


def _gradient_blocker():
    """Pass the value through unchanged while cutting the gradient that returns."""

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach() + 0.0 * tensor if tensor.requires_grad else tensor

    return hook


def _per_item_dot(delta: torch.Tensor, grad: torch.Tensor, index, site: str) -> torch.Tensor:
    """``<delta, grad>`` at one index, summed over the axes the index does not name."""
    dims = {axis: _sites.axis_dim(site, axis) for axis in index}
    selector = [slice(None)] * delta.ndim
    for axis, value in index.items():
        selector[dims[axis]] = value
    sel = tuple(selector)
    product = delta[sel] * grad[sel]
    return product.reshape(product.shape[0], -1).sum(dim=-1)


def _normalise_per_item(dots: torch.Tensor, baselines: Baselines) -> float:
    keep = baselines.keep
    if not baselines.kept:
        return float("nan")
    gap = baselines.clean[keep] - baselines.corrupt[keep]
    return float((dots[keep] / gap).mean())


def _recompute_softmax_delta(site: str, activations, deltas, index, model, layer: int):
    """The exact change in the attention pattern caused by patching q, k or scores.

    One softmax, no forward pass.  This is what moving the expansion point past
    the saturating nonlinearity costs: nothing.
    """
    scores = activations[f"blocks.{layer}.attn.scores"]
    pattern = activations[f"blocks.{layer}.attn.pattern"]
    leaf = site.rsplit(".", 1)[-1]
    new_scores = scores.clone()

    if leaf == "scores":
        dims = {axis: _sites.axis_dim(site, axis) for axis in index}
        selector = [slice(None)] * scores.ndim
        for axis, value in index.items():
            selector[dims[axis]] = value
        new_scores[tuple(selector)] = scores[tuple(selector)] + deltas[site][tuple(selector)]
    else:
        q = activations[f"blocks.{layer}.attn.q"]
        k = activations[f"blocks.{layer}.attn.k"]
        dq = deltas[f"blocks.{layer}.attn.q"] if leaf == "q" else torch.zeros_like(q)
        dk = deltas[f"blocks.{layer}.attn.k"] if leaf == "k" else torch.zeros_like(k)
        mask = torch.zeros_like(q if leaf == "q" else k)
        dims = {axis: _sites.axis_dim(site, axis) for axis in index}
        selector = [slice(None)] * mask.ndim
        for axis, value in index.items():
            selector[dims[axis]] = value
        mask[tuple(selector)] = 1.0
        if leaf == "q":
            q_new = q + mask * dq
            k_new = k
        else:
            q_new = q
            k_new = k + mask * dk
        import math

        new_scores = torch.einsum("bqhk,bshk->bhqs", q_new, k_new) / math.sqrt(model.cfg.d_head)
        if model.cfg.causal:
            seq = scores.shape[-1]
            causal = model.causal_mask[:seq, :seq]
            new_scores = new_scores.masked_fill(causal, torch.finfo(new_scores.dtype).min)
    return torch.softmax(new_scores, dim=-1) - pattern


def attribute(
    model,
    *,
    pair: PromptPair,
    sites,
    direction: Direction,
    counterfactual: Counterfactual | None = None,
    method: str = "linear",
    n_steps: int = 8,
    recompute_softmax: bool = False,
    gradient_dropout: bool = False,
    granularity: dict[str, str] | None = None,
    baselines: Baselines | None = None,
    metric: Metric | None = None,
) -> AttributionResult:
    """Estimate every indexed patching effect from gradients.

    ``method`` is ``"linear"`` for plain attribution patching or
    ``"integrated"`` for an ``n_steps`` average along the path from the base
    activation to the patched one.
    """
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")
    if direction == "denoise" and counterfactual is not None:
        raise ValueError("a counterfactual has no meaning when denoising")
    if direction == "noise" and counterfactual is None:
        raise ValueError("noising needs a counterfactual")
    if method not in ("linear", "integrated"):
        raise ValueError(f"method must be 'linear' or 'integrated', got {method!r}")
    if method == "integrated" and n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

    metric = metric or pair.metric
    granularity = granularity or default_granularity()
    cfg = model.cfg
    names = list(
        _sites.resolve(sites if isinstance(sites, str) else list(sites), cfg)
    )
    # Softmax recomputation needs the whole attention chain of any layer it
    # touches, even if the caller only asked about one site of it.
    needed = set(names)
    if recompute_softmax:
        for name in names:
            if ".attn." in name and name.rsplit(".", 1)[-1] in ("q", "k", "scores"):
                layer = name.split(".")[1]
                needed |= {
                    f"blocks.{layer}.attn.{leaf}" for leaf in ("q", "k", "scores", "pattern")
                }
    needed = list(_sites.validate(sorted(needed), cfg))

    if baselines is None:
        baselines = measure_baselines(model, pair)

    base_tokens = pair.corrupt if direction == "denoise" else pair.clean
    other_tokens = pair.clean if direction == "denoise" else pair.corrupt

    base_acts, base_grads = _run_with_gradients(model, base_tokens, metric, needed)
    forwards, backwards, items = 3, 1, 2 * pair.batch + pair.batch

    with torch.inference_mode():
        other_acts = {}
        from ..model.hooks import capture

        with capture(model, needed) as store:
            model(other_tokens)
        other_acts = {name: store[name] for name in needed}
    forwards += 1
    items += pair.batch

    # The patched-in value, and so the displacement the expansion is about.
    deltas: dict[str, torch.Tensor] = {}
    for name in needed:
        if direction == "denoise":
            target = other_acts[name]
        else:
            target = counterfactual.value(name, base_acts[name].detach(), other_acts[name])
        deltas[name] = (target - base_acts[name].detach()).detach()

    gradients: dict[str, torch.Tensor] = dict(base_grads)

    integrated_dots: dict[tuple[str, tuple], torch.Tensor] = {}
    if method == "integrated":
        for name in names:
            delta = deltas[name]
            base = base_acts[name].detach()
            big_tokens = base_tokens.repeat(n_steps, 1)
            fractions = torch.arange(1, n_steps + 1, dtype=delta.dtype) / n_steps
            shape = [n_steps * pair.batch] + [1] * (delta.ndim - 1)
            weights = fractions.repeat_interleave(pair.batch).reshape(shape)
            stacked_base = base.repeat(n_steps, *([1] * (base.ndim - 1)))
            for index in _index_list(name, cfg, pair.batch, pair.seq, granularity):
                # Only this index moves; everything else stays at the base value.
                mask = torch.zeros_like(delta)
                selector = [slice(None)] * delta.ndim
                for axis, value in index.items():
                    selector[_sites.axis_dim(name, axis)] = value
                mask[tuple(selector)] = 1.0
                masked_delta = (mask * delta).repeat(n_steps, *([1] * (delta.ndim - 1)))
                interpolated = (stacked_base + weights * masked_delta).requires_grad_(True)
                _, grads = _run_with_gradients(
                    model, big_tokens, metric, [name], patches={name: interpolated},
                    replicas=n_steps, batch=pair.batch,
                )
                forwards += 1
                backwards += 1
                items += n_steps * pair.batch
                grad = grads.get(name)
                key = (name, tuple(sorted(index.items())))
                if grad is None:  # pragma: no cover - no path from this site to the metric
                    integrated_dots[key] = torch.zeros(pair.batch, dtype=delta.dtype)
                    continue
                averaged = grad.reshape(n_steps, pair.batch, *grad.shape[1:]).mean(dim=0)
                integrated_dots[key] = _per_item_dot(delta, averaged, index, name)

    # Gradient dropout: one extra backward per layer, taking the largest
    # magnitude over the suppressed variants.
    dropout_grads: list[dict[str, torch.Tensor]] = []
    if gradient_dropout:
        for layer in range(cfg.n_layers):
            _, grads = _run_with_gradients(
                model, base_tokens, metric, needed, drop_layer=layer
            )
            forwards += 1
            backwards += 1
            items += pair.batch
            dropout_grads.append(grads)

    effects: dict[tuple[str, tuple], float] = {}
    for site in names:
        indices = _index_list(site, cfg, pair.batch, pair.seq, granularity)
        leaf = site.rsplit(".", 1)[-1]
        use_softmax_fix = recompute_softmax and leaf in ("q", "k", "scores")
        layer = int(site.split(".")[1]) if site.startswith("blocks.") else -1
        for index in indices:
            key = (site, tuple(sorted(index.items())))
            if method == "integrated" and key in integrated_dots:
                effects[key] = _normalise_per_item(integrated_dots[key], baselines)
                continue
            if use_softmax_fix:
                pattern_site = f"blocks.{layer}.attn.pattern"
                pattern_delta = _recompute_softmax_delta(
                    site, {k: v.detach() for k, v in base_acts.items()}, deltas, index, model,
                    layer,
                )
                grad = gradients.get(pattern_site)
                if grad is None:  # pragma: no cover
                    effects[key] = float("nan")
                    continue
                dots = (pattern_delta * grad).reshape(pair.batch, -1).sum(dim=-1)
            else:
                grad = gradients.get(site)
                if grad is None:  # pragma: no cover
                    effects[key] = float("nan")
                    continue
                dots = _per_item_dot(deltas[site], grad, index, site)
            value = _normalise_per_item(dots, baselines)
            if dropout_grads:
                best = value
                for variant in dropout_grads:
                    alt = variant.get(site)
                    if alt is None:
                        continue
                    candidate = _normalise_per_item(
                        _per_item_dot(deltas[site], alt, index, site), baselines
                    )
                    if abs(candidate) > abs(best):
                        best = candidate
                value = best
            effects[key] = value

    fixes = tuple(
        name
        for name, on in (("softmax_recomputation", recompute_softmax),
                         ("gradient_dropout", gradient_dropout))
        if on
    )
    return AttributionResult(
        effects=effects,
        direction=direction,
        counterfactual=counterfactual.name if counterfactual else "clean",
        method=method,
        fixes=fixes,
        forward_passes=forwards,
        backward_passes=backwards,
        batch_items=items,
        n_steps=n_steps if method == "integrated" else 1,
        baselines=baselines.report(),
    )
