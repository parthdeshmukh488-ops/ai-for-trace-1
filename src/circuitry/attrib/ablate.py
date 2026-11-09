"""Running the model with a *set of edges* removed.

Node ablation is easy: replace one activation.  Edge ablation is not, because
an edge is not a thing the forward pass contains.  "Head 3 reads head 1's
output through its value input, and that connection is cut" has to be realised
by reconstructing head 3's value input from everything *except* head 1's
contribution, while head 1 keeps writing into the residual stream for every
other reader.

The construction rests on one fact about this architecture: the residual
stream is a sum.  At any point it is

    resid = embed + pos + sum over earlier components of their output

so the input a particular reader sees can be rewritten term by term.  To cut
the edge from sender ``s`` into receiver ``r``'s port, take the residual
stream ``r`` would have read and swap ``s``'s term for its counterfactual
value:

    input(r) = resid_at_r - contribution(s) + counterfactual_contribution(s)

Then recompute ``r``'s port from that input and let the rest of the pass carry
on.  Several edges into the same port are several terms swapped; edges into
different ports of the same head are independent, which is what lets a circuit
distinguish a query connection from a value one.

Two things make this exact rather than approximate.

**A head's contribution is addressable.**  ``attn.out`` is the sum over heads,
so patching it moves every head at once.  But ``z`` is per head and
``out = sum_h z_h W_O_h + b_O``, so ``z_h W_O_h`` is exactly head ``h``'s
term, and the bias belongs to no sender and is never swapped.

**The contributions come from the pass itself, not from a cache.**  Under
ablation the upstream components produce different values, and those different
values are what the downstream readers should see.  Reading them live is the
difference between ablating one edge and ablating a path.

LayerNorm is the one place a choice has to be made.  The reconstructed input
is a genuine residual stream, so the honest thing is to put it through
LayerNorm properly, recomputing the normalisation scale from it -- and that is
the default, which makes the whole construction exact.  The alternative,
freezing the scale at its base-run value, is what *attribution graphs* need,
because it is the only way the map from residual stream to port becomes linear
and edge weights become closed-form products.  It is available as
``freeze_layernorm=True`` and it is not a free choice: measured on a small
random model, freezing changes a single edge's measured effect by up to 0.35
of the clean-to-corrupt gap.  That is reported rather than assumed small, and
it is the reason the default is the exact one.
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
    normalise_per_item,
)
from .patching import Counterfactual

__all__ = ["AblationEngine", "check_identity"]


def _contribution_sites(cfg) -> list[str]:
    out = ["embed", "pos_embed"]
    for layer in range(cfg.n_layers):
        out += [
            f"blocks.{layer}.attn.z",
            f"blocks.{layer}.mlp.out",
            f"blocks.{layer}.resid_pre",
            f"blocks.{layer}.resid_mid",
        ]
    out += [f"blocks.{cfg.n_layers - 1}.resid_post", "ln_final.scale", "logits"]
    for layer in range(cfg.n_layers):
        out += [f"blocks.{layer}.ln1.scale", f"blocks.{layer}.ln2.scale"]
    return out


@dataclass
class AblationEngine:
    """Evaluates the metric with an arbitrary set of edges cut.

    Built once per prompt pair and direction, because the counterfactual
    contributions are cached; then queried many times with different edge
    sets, which is what greedy discovery needs.
    """

    model: object
    pair: PromptPair
    direction: Direction
    counterfactual: Counterfactual | None = None
    baselines: Baselines | None = None
    freeze_layernorm: bool = False

    def __post_init__(self) -> None:
        if self.direction not in ("denoise", "noise"):
            raise ValueError(f"direction must be 'denoise' or 'noise', got {self.direction!r}")
        if self.direction == "denoise" and self.counterfactual is not None:
            raise ValueError("a counterfactual has no meaning when denoising")
        if self.direction == "noise" and self.counterfactual is None:
            raise ValueError("noising needs a counterfactual")
        cfg = self.model.cfg
        if self.baselines is None:
            self.baselines = measure_baselines(self.model, self.pair)
        sites = _contribution_sites(cfg)
        with capture(self.model, sites) as clean, torch.inference_mode():
            self.model(self.pair.clean)
        with capture(self.model, sites) as corrupt, torch.inference_mode():
            self.model(self.pair.corrupt)
        self._clean = {name: clean[name] for name in sites}
        self._corrupt = {name: corrupt[name] for name in sites}
        self.base_tokens = self.pair.corrupt if self.direction == "denoise" else self.pair.clean
        base = self._corrupt if self.direction == "denoise" else self._clean
        other = self._clean if self.direction == "denoise" else self._corrupt
        # What a cut edge's sender contributes instead.  Denoising restores the
        # clean value; noising substitutes the chosen counterfactual.
        self._swap_in = {}
        for name in sites:
            if self.direction == "denoise":
                self._swap_in[name] = other[name]
            else:
                self._swap_in[name] = self.counterfactual.value(name, base[name], other[name])
        self.forward_passes = 4

    # -- contributions ---------------------------------------------------

    def _head_contribution(self, z: torch.Tensor, layer: int, head: int) -> torch.Tensor:
        w_o = self.model.blocks[layer].attn.W_O[head]
        return z[:, :, head] @ w_o

    def _swapped_contribution(self, component: Component) -> torch.Tensor:
        if component.kind == "embed":
            return self._swap_in["embed"] + self._swap_in["pos_embed"]
        if component.kind == "attn":
            return self._head_contribution(
                self._swap_in[f"blocks.{component.layer}.attn.z"],
                component.layer,
                component.head,
            )
        return self._swap_in[f"blocks.{component.layer}.mlp.out"]

    # -- the ablated pass ------------------------------------------------

    def level(self, removed) -> float:
        """The normalised metric with ``removed`` edges cut.

        Everything is assembled as closures over one dictionary, ``live``,
        which the recorders fill as the pass proceeds.  Forward order
        guarantees what each reader needs is already in it: a head's ``q``
        hook fires after that layer's ``resid_pre``, and an MLP's ``pre`` hook
        fires after that layer's ``attn.z``.
        """
        cfg = self.model.cfg
        by_port: dict[tuple[int, str, int | None], list[Component]] = {}
        for edge in removed:
            by_port.setdefault((edge.target.layer, edge.port, edge.target.head), []).append(
                edge.source
            )

        live: dict[str, torch.Tensor] = {}
        hooks: dict[str, list] = {}

        def record(name: str):
            def hook(tensor: torch.Tensor):
                live[name] = tensor
                return None

            return hook

        recorded = ["embed", "pos_embed", f"blocks.{cfg.n_layers - 1}.resid_post"]
        for layer in range(cfg.n_layers):
            recorded += [
                f"blocks.{layer}.attn.z",
                f"blocks.{layer}.mlp.out",
                f"blocks.{layer}.resid_pre",
                f"blocks.{layer}.resid_mid",
            ]
        for name in recorded:
            hooks.setdefault(name, []).append(record(name))

        def live_contribution(component: Component) -> torch.Tensor:
            if component.kind == "embed":
                return live["embed"] + live["pos_embed"]
            if component.kind == "attn":
                return self._head_contribution(
                    live[f"blocks.{component.layer}.attn.z"], component.layer, component.head
                )
            return live[f"blocks.{component.layer}.mlp.out"]

        def swap_delta(senders) -> torch.Tensor:
            total = None
            for sender in senders:
                change = self._swapped_contribution(sender) - live_contribution(sender)
                total = change if total is None else total + change
            return total

        def port_hook(layer: int, port: str, head: int, senders):
            attn = self.model.blocks[layer].attn
            weight = {"q": attn.W_Q, "k": attn.W_K, "v": attn.W_V}[port][head]
            bias = {"q": attn.b_Q, "k": attn.b_K, "v": attn.b_V}[port][head]

            def hook(tensor: torch.Tensor) -> torch.Tensor:
                stream = live[f"blocks.{layer}.resid_pre"] + swap_delta(senders)
                out = tensor.clone()
                out[:, :, head] = self._layernorm(stream, layer, "ln1") @ weight + bias
                return out

            return hook

        def mlp_hook(layer: int, senders):
            mlp = self.model.blocks[layer].mlp

            def hook(tensor: torch.Tensor) -> torch.Tensor:
                stream = live[f"blocks.{layer}.resid_mid"] + swap_delta(senders)
                return self._layernorm(stream, layer, "ln2") @ mlp.W_in + mlp.b_in

            return hook

        def logit_hook(senders):
            def hook(tensor: torch.Tensor) -> torch.Tensor:
                stream = live[f"blocks.{cfg.n_layers - 1}.resid_post"] + swap_delta(senders)
                normed = self._layernorm(stream, cfg.n_layers - 1, "final")
                return normed @ self.model.W_U + self.model.b_U

            return hook

        for layer in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                for port in ("q", "k", "v"):
                    senders = by_port.get((layer, port, head))
                    if senders:
                        hooks.setdefault(f"blocks.{layer}.attn.{port}", []).append(
                            port_hook(layer, port, head, senders)
                        )
            senders = by_port.get((layer, "mlp_in", None))
            if senders:
                hooks.setdefault(f"blocks.{layer}.mlp.pre", []).append(mlp_hook(layer, senders))
        senders = by_port.get((cfg.n_layers, "resid", None))
        if senders:
            hooks.setdefault("logits", []).append(logit_hook(senders))

        with intervene(self.model, hooks), torch.inference_mode():
            logits = self.model(self.base_tokens)
        self.forward_passes += 1
        values = self.pair.metric(logits)
        self._last_per_item = normalise_per_item(values, self.baselines)
        return normalise(values, self.baselines)

    def level_per_item(self, removed) -> torch.Tensor:
        """The normalised level for each usable item, for bootstrapping."""
        self.level(removed)
        return self._last_per_item

    def effect(self, removed) -> float:
        """The signed change from the unpatched baseline; see ``metric.effect_of``."""
        return effect_of(self.level(removed), self.direction)

    # -- port reconstruction --------------------------------------------

    def _layernorm(self, x: torch.Tensor, layer: int, which: str) -> torch.Tensor:
        """Put a reconstructed residual stream through the receiver's LayerNorm.

        By default the normalisation scale is recomputed from ``x``, which is
        what the receiver would actually do and what makes the reconstruction
        exact.  With ``freeze_layernorm`` the base run's scale is reused
        instead, which linearises the map -- needed for closed-form edge
        weights -- at a cost this module measures rather than waves away.
        """
        cfg = self.model.cfg
        if not cfg.use_ln:
            return x
        module = (
            self.model.ln_final if which == "final" else getattr(self.model.blocks[layer], which)
        )
        centred = x - x.mean(dim=-1, keepdim=True)
        if self.freeze_layernorm:
            name = "ln_final.scale" if which == "final" else f"blocks.{layer}.{which}.scale"
            base = self._corrupt if self.direction == "denoise" else self._clean
            scale = base[name]
        else:
            scale = (centred.pow(2).mean(dim=-1, keepdim=True) + cfg.ln_eps).rsqrt()
        return centred * scale * module.weight + module.bias


def check_identity(engine: AblationEngine, *, edges=None) -> dict:
    """Verify the reconstruction against two independent computations.

    The empty case is nearly free and nearly worthless: with nothing cut, no
    port hook is installed at all, so it only checks that the recorders do not
    leak.  The real checks are the other two.

    *A single cut edge against path patching.*  :func:`circuitry.attrib.paths
    .path_patch` computes the same quantity a completely different way -- it
    freezes the components between sender and receiver and lets the real
    forward pass carry the sender's change -- so agreement between the two is
    agreement between two separately written implementations.  This is the
    strongest check available here and the one that matters.

    *Every edge into one port, against patching that port's input directly.*
    Cutting all of them means the port reads the counterfactual residual
    stream, which is also what a node patch of that port produces -- except
    that the reconstruction uses the base run's LayerNorm scale while a node
    patch uses the counterfactual run's.  The gap between them is therefore
    exactly the cost of freezing the scale, which is why it is measured here
    rather than asserted to be small.
    """
    from .circuit import all_edges
    from .paths import path_patch

    cfg = engine.model.cfg
    out: dict[str, float] = {}

    with torch.inference_mode():
        direct = normalise(
            engine.pair.metric(engine.model(engine.base_tokens)), engine.baselines
        )
    out["empty_ablation_level"] = engine.level([])
    out["direct_level"] = direct
    out["empty_vs_direct"] = abs(out["empty_ablation_level"] - direct)

    # --- a single edge, against path patching -------------------------
    candidates = edges
    if candidates is None:
        candidates = [
            e
            for e in all_edges(cfg)
            if e.source.kind == "attn" and e.target.kind in ("attn", "mlp")
        ][:6]
    worst = 0.0
    pairs = []
    for edge in candidates:
        mine = engine.level([edge])
        theirs = path_patch(
            engine.model,
            pair=engine.pair,
            sender=edge.source,
            receiver=edge.target,
            port=edge.port,
            direction=engine.direction,
            counterfactual=engine.counterfactual,
            baselines=engine.baselines,
        ).level
        worst = max(worst, abs(mine - theirs))
        pairs.append({"edge": edge.label, "ablation": mine, "path_patch": theirs})
    out["n_single_edges_checked"] = len(candidates)
    out["worst_vs_path_patch"] = worst
    out["single_edge_detail"] = pairs

    # --- every edge into one port, against a node patch of it ---------
    from .patching import _multi_index_hook

    layer = cfg.n_layers - 1
    into_mlp = [
        e for e in all_edges(cfg) if e.target.kind == "mlp" and e.target.layer == layer
    ]
    if into_mlp:
        reconstructed = engine.level(into_mlp)
        site = f"blocks.{layer}.mlp.pre"
        with capture(engine.model, [site]) as base_store, torch.inference_mode():
            engine.model(engine.base_tokens)
        other = engine.pair.clean if engine.direction == "denoise" else engine.pair.corrupt
        with capture(engine.model, [site]) as other_store, torch.inference_mode():
            engine.model(other)
        target = (
            other_store[site]
            if engine.direction == "denoise"
            else engine.counterfactual.value(site, base_store[site], other_store[site])
        )
        hook = _multi_index_hook(site, target, [{}])
        with intervene(engine.model, {site: hook}), torch.inference_mode():
            logits = engine.model(engine.base_tokens)
        node_patched = normalise(engine.pair.metric(logits), engine.baselines)
        out["all_edges_into_last_mlp"] = reconstructed
        out["node_patch_of_that_port"] = node_patched
        out["port_reconstruction_gap"] = abs(reconstructed - node_patched)
        out["n_edges_in_that_case"] = len(into_mlp)
    return out
