"""Path patching: the effect of one *edge*, not of one component.

A node effect answers "does this head matter", which is usually not the
question.  The question is "does this head matter *to that one*", and through
which of its inputs -- because "head 9.9 reads head 7.3's output through its
query and not through its value" is the kind of statement published circuits
are made of, and a node measurement cannot express it.

The procedure, and the part implementations get wrong
-----------------------------------------------------

To measure the edge from sender ``u`` to receiver ``v``, the change must reach
``v`` **only** along that edge.  Three things have to be arranged:

1. ``u``'s output is replaced by its counterfactual value.
2. Every component that sits between ``u`` and ``v`` is **frozen** at its
   base-run value.  Those components read ``u``, so without freezing them the
   change would also arrive at ``v`` along ``u -> w -> v``, and the
   measurement would be of a set of paths rather than of an edge.
3. If ``v`` is an attention head and the edge is into its *value* input, then
   ``v``'s own query and key are frozen too.  Otherwise the sender's change
   moves ``v``'s attention pattern, and what gets measured is the edge into
   ``q`` and ``k`` as well as the one into ``v``.

Everything downstream of ``v`` recomputes freely: that is what makes this the
effect of the edge *on the output*, rather than the effect on ``v``'s
activation.

Point 3 is the one worth labouring.  It is why :func:`path_patch` takes a
``port`` and why there is no default for it.

What edge effects do not do
---------------------------

They do not sum to the node effect.  The composition is nonlinear -- softmax
and the MLP activation -- so decomposing a node's influence into its outgoing
edges leaves a remainder, and the *size* of that remainder bounds what any
edge decomposition can mean.  :func:`path_decomposition_error` measures it, and
phase 3 records the number rather than assuming it is small.
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

__all__ = ["PathResult", "path_decomposition_error", "path_patch"]

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
) -> PathResult:
    """The effect on the metric of the single edge ``sender -> receiver.port``."""
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
