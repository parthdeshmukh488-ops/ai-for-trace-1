"""Scoring a circuit: faithfulness, completeness, minimality, and the ceiling.

The first three are the standard family, and each is implemented with its own
Monte Carlo error reported rather than as a bare number -- completeness over
twenty sampled subsets has an error bar, and quoting it without one is the
kind of thing this repository exists not to do.

The fourth is the one that makes the others interpretable here, and it is not
standard.  A planted circuit is the *program's* dependency graph: every edge
the program needs for some input.  A particular pair of prompts exercises only
part of it.  An edge whose removal changes nothing *on these prompts* cannot
be recovered by any causal method at all, because there is no causal signal to
recover -- so recall against the full planted graph has a ceiling below one,
and that ceiling is a property of the prompts rather than a failure of the
method.  :func:`recoverable` measures it, and every recall number in phase 5
is reported against both the full graph and the recoverable subset.

Without that, a discovery method looks worse the more thorough the ground
truth is, which is exactly backwards.
"""

from __future__ import annotations

import torch

from .ablate import AblationEngine
from .circuit import Circuit, Edge

__all__ = ["completeness", "faithfulness", "minimality", "recoverable", "score"]


def faithfulness(engine: AblationEngine, circuit: Circuit, *, universe=None) -> dict:
    """Cut everything outside the circuit and see whether the behaviour survives.

    One number, and the weakest member of the family: a circuit can be
    faithful and still be the wrong circuit, which is what completeness and
    minimality are for.
    """
    from .circuit import all_edges

    universe = list(all_edges(engine.model.cfg) if universe is None else universe)
    outside = [e for e in universe if e not in circuit.edges]
    level = engine.level(outside)
    unpatched = 0.0 if engine.direction == "denoise" else 1.0
    return {
        # The level is the faithfulness: 1 means the circuit alone reproduces
        # the full model's behaviour, in either direction, because the
        # normalisation already puts clean at 1 and corrupt at 0.
        "faithfulness": level,
        "unpatched_level": unpatched,
        "n_edges_in_circuit": len(circuit.edges),
        "n_edges_cut": len(outside),
    }


def completeness(
    engine: AblationEngine,
    circuit: Circuit,
    *,
    n_subsets: int = 20,
    max_subset: int | None = None,
    generator: torch.Generator | None = None,
    universe=None,
) -> dict:
    """Does the circuit respond to knockouts the way the whole model does?

    For sampled subsets ``K`` of the circuit, compare the circuit with ``K``
    also removed against the whole model with only ``K`` removed.  A circuit
    that is faithful but incomplete agrees with the model until you start
    breaking it and then diverges, which is the signature of having left out
    something the circuit was relying on.

    The maximum over subsets is reported alongside the mean, because
    completeness fails on *specific* subsets and an average hides exactly
    those.
    """
    from .circuit import all_edges

    generator = generator or torch.Generator().manual_seed(0)
    universe = list(all_edges(engine.model.cfg) if universe is None else universe)
    inside = sorted(circuit.edges)
    outside = [e for e in universe if e not in circuit.edges]
    if not inside:
        return {"n_subsets": 0, "mean_discrepancy": float("nan")}
    cap = max_subset or max(1, len(inside) // 2)

    discrepancies = []
    for _ in range(n_subsets):
        size = int(torch.randint(1, cap + 1, (1,), generator=generator))
        picks = torch.randperm(len(inside), generator=generator)[:size].tolist()
        knocked = [inside[i] for i in picks]
        circuit_level = engine.level([*outside, *knocked])
        model_level = engine.level(knocked)
        discrepancies.append(abs(circuit_level - model_level))
    values = torch.tensor(discrepancies)
    return {
        "n_subsets": n_subsets,
        "mean_discrepancy": float(values.mean()),
        "max_discrepancy": float(values.max()),
        # The Monte Carlo error of the mean, so the number can be read.
        "mean_standard_error": float(values.std() / (len(values) ** 0.5)),
        "max_subset_size": cap,
    }


def minimality(
    engine: AblationEngine,
    circuit: Circuit,
    *,
    n_subsets: int = 8,
    generator: torch.Generator | None = None,
    universe=None,
) -> dict:
    """Does every member of the circuit earn its place under some knockout?

    For each edge, the largest degradation its removal causes over sampled
    knockouts of the rest.  An edge whose best score is zero is dead weight:
    the circuit would be smaller and no worse without it.
    """
    from .circuit import all_edges

    generator = generator or torch.Generator().manual_seed(0)
    universe = list(all_edges(engine.model.cfg) if universe is None else universe)
    inside = sorted(circuit.edges)
    outside = [e for e in universe if e not in circuit.edges]
    if not inside:
        return {"n_edges": 0, "min_score": float("nan")}

    scores: dict[str, float] = {}
    for edge in inside:
        others = [e for e in inside if e != edge]
        best = 0.0
        for _ in range(n_subsets):
            size = int(torch.randint(0, max(len(others), 1) + 1, (1,), generator=generator))
            picks = torch.randperm(len(others), generator=generator)[:size].tolist()
            knocked = [others[i] for i in picks]
            with_edge = engine.level([*outside, *knocked])
            without = engine.level([*outside, *knocked, edge])
            best = max(best, abs(with_edge - without))
        scores[edge.label] = best
    values = torch.tensor(list(scores.values()))
    return {
        "n_edges": len(inside),
        "per_edge": scores,
        "min_score": float(values.min()),
        "median_score": float(values.median()),
        "n_dead_weight": int((values < 1e-9).sum()),
        "n_subsets": n_subsets,
    }


def recoverable(
    engine: AblationEngine, truth: Circuit, *, threshold: float
) -> dict:
    """Which planted edges have any causal signal on *these* prompts.

    An edge whose individual removal moves the metric by less than
    ``threshold`` -- which should be the measured noise floor -- cannot be
    found by any causal method, because there is nothing to find.  Recall
    against the full planted graph is therefore capped, and this is the cap.
    """
    effects: dict[str, float] = {}
    live: set[Edge] = set()
    unpatched = 0.0 if engine.direction == "denoise" else 1.0
    for edge in sorted(truth.edges):
        cost = abs(engine.level([edge]) - unpatched)
        effects[edge.label] = cost
        if cost >= threshold:
            live.add(edge)
    return {
        "threshold": threshold,
        "n_planted": len(truth.edges),
        "n_recoverable": len(live),
        "ceiling_on_recall": len(live) / len(truth.edges) if truth.edges else float("nan"),
        "per_edge_effect": effects,
        "recoverable": Circuit.of(edges=live, label="recoverable"),
    }


def score(
    engine: AblationEngine,
    found: Circuit,
    truth: Circuit,
    *,
    recall_threshold: float = 0.0,
    with_family: bool = False,
    generator: torch.Generator | None = None,
) -> dict:
    """Precision and recall against the planted graph and against its ceiling."""
    out = dict(found.score_against(truth))
    if recall_threshold > 0:
        bound = recoverable(engine, truth, threshold=recall_threshold)
        reachable = bound["recoverable"]
        against = found.score_against(reachable)
        out["ceiling_on_recall"] = bound["ceiling_on_recall"]
        out["n_recoverable"] = bound["n_recoverable"]
        out["edge_recall_vs_recoverable"] = against["edge_recall"]
        out["edge_f1_vs_recoverable"] = against["edge_f1"]
        out["edge_precision_vs_recoverable"] = against["edge_precision"]
    if with_family:
        out["faithfulness"] = faithfulness(engine, found)
        out["completeness"] = completeness(engine, found, generator=generator)
        out["minimality"] = minimality(engine, found, generator=generator)
    return out
