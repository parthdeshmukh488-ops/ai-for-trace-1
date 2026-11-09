"""Automated circuit discovery by greedy edge pruning.

The procedure: start with the whole model on the clean prompt, walk the edges
in reverse topological order, and cut each one whose removal costs less than a
threshold.  What is left uncut is the circuit.

Two properties distinguish this from ranking edges by their individual effect,
and both matter.

**The removals are committed and persistent.**  An edge is judged in the
context of the ones already gone, which is the only way redundancy is handled
correctly: two paths that each look dispensable on their own are not both
dispensable, and a procedure that scored every edge against the full model
would remove both.  The cost is that the result depends on the traversal
order, which is why the order is fixed and stated rather than left to
whichever way a dictionary happened to iterate.

**Discovery is a noising procedure.**  Nothing is cut at the start, so the
model runs on the clean prompt at a level of 1, and cutting edges moves it
toward the corrupt behaviour.  Denoising would answer a different question --
which set *restores* the behaviour from a corrupt run -- and the two can
disagree, so the direction is not a free parameter here.

**The threshold has no default**, because choosing it *is* the question.  It
should be stated relative to the metric's own sampling error, which
:func:`noise_floor` measures: a threshold below that floor returns the whole
model, and that is a fact about the measurement rather than about the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .ablate import AblationEngine
from .circuit import Circuit, Component, Edge, all_edges
from .metric import Baselines, Direction, PromptPair
from .patching import Counterfactual, ResampleAblation

__all__ = ["DiscoveryResult", "discover", "noise_floor", "topological_rank"]


def topological_rank(component: Component) -> int:
    """Position in the forward order: embedding first, logit sink last.

    Attention precedes the MLP within a layer because the MLP reads that
    layer's attention output, so the pair occupies two consecutive ranks.
    """
    if component.kind == "embed":
        return -1
    if component.kind == "logits":
        return 2 * component.layer
    return 2 * component.layer + (0 if component.kind == "attn" else 1)


def _traversal_order(edges) -> list[Edge]:
    """Reverse topological: the output end first, and deterministic."""
    return sorted(
        edges,
        key=lambda e: (
            -topological_rank(e.target),
            -topological_rank(e.source),
            e.port,
            e.target.head if e.target.head is not None else -1,
            e.source.head if e.source.head is not None else -1,
        ),
    )


@dataclass
class DiscoveryResult:
    circuit: Circuit
    removed: tuple[Edge, ...]
    threshold: float
    direction: str
    counterfactual: str
    forward_passes: int
    final_level: float
    noise_floor: float | None = None
    trace: list = field(default_factory=list)

    @property
    def n_kept(self) -> int:
        return len(self.circuit.edges)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "direction": self.direction,
            "counterfactual": self.counterfactual,
            "n_edges_kept": self.n_kept,
            "n_edges_removed": len(self.removed),
            "final_level": self.final_level,
            "noise_floor": self.noise_floor,
            "forward_passes": self.forward_passes,
            "kept": sorted(e.label for e in self.circuit.edges),
        }


def noise_floor(
    model,
    *,
    pair: PromptPair,
    direction: Direction = "noise",
    counterfactual: Counterfactual | None = None,
    n_subsets: int = 4,
    n_boot: int = 200,
    generator: torch.Generator | None = None,
    baselines: Baselines | None = None,
) -> dict:
    """How precisely the metric is known, given the prompt sample.

    Not the model's determinism -- the model is exactly reproducible.  The
    uncertainty is in the *prompts*: a different sample of pairs would give a
    different level, and a discovery threshold below that spread is picking up
    sampling noise rather than circuit structure.

    Measured by bootstrapping over items, on a few random edge subsets rather
    than on the empty set, because the empty set's level is pinned to the
    baseline by construction and has no spread at all.
    """
    counterfactual = counterfactual or (
        ResampleAblation() if direction == "noise" else None
    )
    generator = generator or torch.Generator().manual_seed(0)
    engine = AblationEngine(model, pair, direction, counterfactual, baselines)
    edges = all_edges(model.cfg)
    spreads = []
    for _ in range(n_subsets):
        mask = torch.rand(len(edges), generator=generator) < 0.5
        subset = [e for e, take in zip(edges, mask.tolist(), strict=True) if take]
        per_item = engine.level_per_item(subset)
        n = per_item.numel()
        if n < 2:
            continue
        draws = torch.randint(0, n, (n_boot, n), generator=generator)
        spreads.append(float(per_item[draws].mean(dim=1).std()))
    return {
        "noise_floor": float(torch.tensor(spreads).median()) if spreads else float("nan"),
        "per_subset": spreads,
        "n_subsets": n_subsets,
        "n_boot": n_boot,
        "n_items": int(engine.baselines.kept),
    }


def discover(
    model,
    *,
    pair: PromptPair,
    threshold: float,
    direction: Direction = "noise",
    counterfactual: Counterfactual | None = None,
    edges=None,
    baselines: Baselines | None = None,
    scorer: str = "exact",
    keep_trace: bool = False,
) -> DiscoveryResult:
    """Greedy edge pruning.  Returns the edges that could *not* be cut."""
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")
    if scorer not in ("exact",):
        raise ValueError(
            f"scorer must be 'exact'; got {scorer!r}. A gradient-scored variant would "
            "need the dead-gradient problem of phase 4 addressed first."
        )
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    if direction == "noise" and counterfactual is None:
        counterfactual = ResampleAblation()
    if direction == "denoise" and counterfactual is not None:
        raise ValueError("a counterfactual has no meaning when denoising")

    engine = AblationEngine(model, pair, direction, counterfactual, baselines)
    candidates = _traversal_order(all_edges(model.cfg) if edges is None else edges)

    removed: list[Edge] = []
    current = engine.level([])
    trace: list = []
    for edge in candidates:
        proposed = [*removed, edge]
        level = engine.level(proposed)
        # Noising starts at 1 and falls; denoising starts at 0 and rises.  In
        # both cases the cost of a cut is the move away from where we were.
        cost = abs(current - level)
        cut = cost < threshold
        if keep_trace:
            trace.append(
                {
                    "edge": edge.label,
                    "current_before": current,
                    "level": level,
                    "cost": cost,
                    "cut": cut,
                }
            )
        if cut:
            removed = proposed
            current = level

    kept = [e for e in candidates if e not in set(removed)]
    return DiscoveryResult(
        circuit=Circuit.of(edges=kept, label=f"discovered@{threshold:g}"),
        removed=tuple(removed),
        threshold=threshold,
        direction=direction,
        counterfactual=counterfactual.name if counterfactual else "clean",
        forward_passes=engine.forward_passes,
        final_level=current,
        trace=trace,
    )
