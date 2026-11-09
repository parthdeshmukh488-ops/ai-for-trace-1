"""The effect of one *edge*, not of one component.

A node effect answers "does this head matter", which is usually not the
question.  The question is "does this head matter *to that one*", and through
which of its inputs -- because "head 9.9 reads head 7.3's output through its
query and not through its value" is the kind of statement published circuits
are made of, and a node measurement cannot express it.

Two procedures, and the difference between them is a correction
-----------------------------------------------------------------

The obvious procedure is **freeze and re-run**: replace the sender's output
with its counterfactual, freeze the components between sender and receiver so
the change cannot reach the receiver by any other route, freeze the receiver's
other ports so its attention pattern cannot move, and let everything
downstream recompute.  That is :func:`path_patch_by_freezing`, and it is
*wrong in general*.

The reason is that the sender's patched output stays in the residual stream
after the receiver, where every later component also reads it.  Freezing those
later components is not an option -- they have to stay live to respond to the
receiver's change, which is the thing being measured.  So the procedure
measures the sender's effect through the receiver *and* through every later
reader at once: a bundle of paths, not an edge.

It is exactly right in one case: when nothing reads the sender after the
receiver.  Measured on a small random model, the two procedures agree to
machine precision (2.8e-17) for every edge whose receiver is the final
read-out, and disagree by 0.24 to 0.56 of the clean-to-corrupt gap for edges
into intermediate components.  The phase-3 test that originally validated this
code used ``hist``, whose one head reaches the output through one MLP whose
only reader is the logits -- the single case in which the buggy procedure is
correct.

The right definition is arithmetic, and it is the one :func:`path_patch` now
uses.  The residual stream is a sum, so a receiver's input can be rewritten
term by term, and cutting an edge means swapping one term in *that receiver's*
input and leaving every other reader's view of the sender alone.  See
:mod:`circuitry.attrib.ablate`, which implements it, composes over sets of
edges, and recomputes each receiver's LayerNorm from its reconstructed input.

What edge effects do not do
---------------------------

They do not sum to the node effect.  The composition is nonlinear -- softmax
and the MLP activation -- so decomposing a node's influence into its outgoing
edges leaves a remainder, and the *size* of that remainder bounds what any
edge decomposition can mean.  :func:`path_decomposition_error` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.hooks import capture, intervene
from .circuit import Component
from .metric import (
    Baselines,
    Direction,
    PromptPair,
    effect_of,
    measure_baselines,
    normalise,
)
from .patching import Counterfactual, _all_indices, _multi_index_hook

__all__ = [
    "PathResult",
    "path_decomposition_error",
    "path_patch",
    "path_patch_by_freezing",
]

_PORT_SITES = {"q": "attn.q", "k": "attn.k", "v": "attn.v"}


@dataclass(frozen=True)
class PathResult:
    """The signed effect of one edge; see PatchResult for the level/effect split."""

    effect: float
    level: float
    sender: Component
    receiver: Component
    port: str
    direction: str
    counterfactual: str
    frozen_sites: tuple[str, ...]
    forward_passes: int

    def to_dict(self) -> dict:
        return {
            "effect": self.effect,
            "level": self.level,
            "sender": self.sender.label,
            "receiver": self.receiver.label,
            "port": self.port,
            "direction": self.direction,
            "counterfactual": self.counterfactual,
            "n_frozen_sites": len(self.frozen_sites),
            "forward_passes": self.forward_passes,
        }


def _component_sites(cfg, layer: int) -> list[str]:
    return [f"blocks.{layer}.attn.z", f"blocks.{layer}.mlp.out"]


def _frozen_plan(cfg, sender: Component, receiver: Component, port: str):
    """Which sites to freeze, and which indices of them, to isolate one edge.

    Returns ``[(site, keep_predicate)]`` where the predicate says which indices
    to leave alone; everything else at that site is pinned to the base run.
    """
    plan: list[tuple[str, object]] = []
    first = max(sender.layer, 0)
    last = receiver.layer if receiver.kind != "logits" else cfg.n_layers

    for layer in range(first, last):
        for site in _component_sites(cfg, layer):
            is_sender_site = (
                sender.kind != "embed"
                and site == sender.output_site
                and layer == sender.layer
            )
            if not is_sender_site:
                plan.append((site, None))
                continue
            if sender.head is None:
                continue  # the sender *is* this whole site; leave it to the patch
            # Freeze the sender's siblings, not the sender itself.
            head = sender.head
            plan.append((site, lambda index, head=head: index.get("head") == head))

    # The MLP at the receiver's own layer reads that layer's attention, so when
    # the receiver is an MLP its layer's heads are between the sender and it.
    if receiver.kind == "mlp":
        site = f"blocks.{receiver.layer}.attn.z"
        if sender.kind == "attn" and sender.layer == receiver.layer:
            head = sender.head
            plan.append((site, lambda index, head=head: index.get("head") == head))
        elif not any(s == site for s, _ in plan):
            plan.append((site, None))

    # The receiver's other ports, so its attention pattern cannot move.
    if receiver.kind == "attn":
        if port not in _PORT_SITES:
            raise ValueError(
                f"a head receiver needs port 'q', 'k' or 'v'; got {port!r}. Without it the "
                "measurement would include the edges into the other two inputs."
            )
        head = receiver.head
        for other, suffix in _PORT_SITES.items():
            if other == port:
                continue
            plan.append(
                (
                    f"blocks.{receiver.layer}.{suffix}",
                    lambda index, head=head: index.get("head") != head,
                )
            )
    elif port not in ("mlp_in", "resid"):
        raise ValueError(
            f"receiver {receiver.label} takes port 'mlp_in' or 'resid'; got {port!r}"
        )
    return plan


def path_patch_by_freezing(
    model,
    *,
    pair: PromptPair,
    sender: Component,
    receiver: Component,
    port: str,
    direction: Direction,
    counterfactual: Counterfactual | None = None,
    baselines: Baselines | None = None,
) -> PathResult:
    """Freeze-and-re-run edge patching.  Correct only in a special case.

    Kept because it is an independently written implementation of the same
    quantity *where the two agree*, which makes it the cross-check for
    :func:`path_patch`.  It is correct when nothing reads the sender after the
    receiver -- in practice, when the receiver is the final read-out -- and
    wrong otherwise, for the reason in the module docstring.  Prefer
    :func:`path_patch`.
    """
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")
    if direction == "denoise" and counterfactual is not None:
        raise ValueError("a counterfactual has no meaning when denoising")
    if direction == "noise" and counterfactual is None:
        raise ValueError("noising needs a counterfactual")
    if sender.kind == "logits":
        raise ValueError("the logit sink has no output to patch")
    if receiver.kind == "embed":
        raise ValueError("the embedding has no inputs")
    if receiver.kind != "logits" and sender.layer >= receiver.layer + (
        1 if receiver.kind == "mlp" else 0
    ):
        raise ValueError(
            f"{sender.label} cannot reach {receiver.label}: it is not upstream of it"
        )

    cfg = model.cfg
    plan = _frozen_plan(cfg, sender, receiver, port)
    needed = {sender.output_site, *(site for site, _ in plan)}
    if baselines is None:
        baselines = measure_baselines(model, pair)

    with capture(model, sorted(needed)) as clean_store, torch.inference_mode():
        model(pair.clean)
    with capture(model, sorted(needed)) as corrupt_store, torch.inference_mode():
        model(pair.corrupt)
    passes = 4

    base_tokens = pair.corrupt if direction == "denoise" else pair.clean
    base_store = corrupt_store if direction == "denoise" else clean_store
    if direction == "denoise":
        source = clean_store[sender.output_site]
    else:
        source = counterfactual.value(
            sender.output_site, clean_store[sender.output_site],
            corrupt_store[sender.output_site],
        )

    hooks: dict[str, list] = {}
    sender_indices = (
        [{"head": sender.head}] if sender.head is not None else [{}]
    )
    hooks.setdefault(sender.output_site, []).append(
        _multi_index_hook(sender.output_site, source, sender_indices)
    )
    for site, keep in plan:
        indices = [
            index
            for index in _all_indices(site, cfg, pair.batch, pair.seq)
            if keep is None or not keep(index)
        ]
        if indices:
            hooks.setdefault(site, []).append(
                _multi_index_hook(site, base_store[site], indices)
            )

    with intervene(model, hooks), torch.inference_mode():
        logits = model(base_tokens)
    passes += 1
    level = normalise(pair.metric(logits), baselines)
    return PathResult(
        effect=effect_of(level, direction),
        level=level,
        sender=sender,
        receiver=receiver,
        port=port,
        direction=direction,
        counterfactual=counterfactual.name if counterfactual else "clean",
        frozen_sites=tuple(sorted(site for site, _ in plan)),
        forward_passes=passes,
    )


def path_decomposition_error(
    model,
    *,
    pair: PromptPair,
    sender: Component,
    receivers,
    ports,
    direction: Direction,
    counterfactual: Counterfactual | None = None,
    node_effect: float | None = None,
    baselines: Baselines | None = None,
) -> dict:
    """How far the sender's edge effects are from its node effect.

    Not a bug to be fixed: the composition is nonlinear, so a node's influence
    does not decompose exactly into its outgoing edges.  The number is a bound
    on what any edge decomposition can mean, which is why it is measured and
    reported rather than assumed negligible.
    """
    from .patching import patch

    if baselines is None:
        baselines = measure_baselines(model, pair)
    if node_effect is None:
        single = patch(
            model,
            pair=pair,
            sites=[sender.output_site],
            direction=direction,
            counterfactual=counterfactual,
            granularity={"pos": "all", "head": "each"},
            baselines=baselines,
        )
        key = (sender.output_site, (("head", sender.head),) if sender.head is not None else ())
        node_effect = single.effects[key]

    edges = []
    for receiver, port in zip(receivers, ports, strict=True):
        result = path_patch(
            model,
            pair=pair,
            sender=sender,
            receiver=receiver,
            port=port,
            direction=direction,
            counterfactual=counterfactual,
            baselines=baselines,
        )
        edges.append(result)
    summed = sum(e.effect for e in edges)
    gap = summed - node_effect
    return {
        "node_effect": node_effect,
        "edge_effects": {f"{e.receiver.label}.{e.port}": e.effect for e in edges},
        "sum_of_edges": summed,
        "discrepancy": gap,
        "relative_discrepancy": gap / node_effect if node_effect else float("nan"),
        "n_edges": len(edges),
    }


def path_patch(
    model,
    *,
    pair: PromptPair,
    sender: Component,
    receiver: Component,
    port: str,
    direction: Direction,
    counterfactual: Counterfactual | None = None,
    baselines: Baselines | None = None,
    freeze_layernorm: bool = False,
) -> PathResult:
    """The effect on the metric of the single edge ``sender -> receiver.port``.

    Implemented by reconstructing the receiver's input term by term, so that
    cutting the edge changes only that receiver's view of the sender; see
    :mod:`circuitry.attrib.ablate`.  This is the definition the rest of the
    package uses, because it is the only one that composes: a *set* of edges
    is well defined under it, which is what discovery needs.
    """
    from .ablate import AblationEngine
    from .circuit import Edge

    engine = AblationEngine(
        model=model,
        pair=pair,
        direction=direction,
        counterfactual=counterfactual,
        baselines=baselines,
        freeze_layernorm=freeze_layernorm,
    )
    edge = Edge(sender, receiver, port)
    level = engine.level([edge])
    return PathResult(
        effect=effect_of(level, direction),
        level=level,
        sender=sender,
        receiver=receiver,
        port=port,
        direction=direction,
        counterfactual=counterfactual.name if counterfactual else "clean",
        frozen_sites=(),
        forward_passes=engine.forward_passes,
    )
