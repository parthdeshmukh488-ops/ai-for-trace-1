"""Greedy edge pruning.

The two properties worth testing are the ones that distinguish this from
ranking edges by their individual effects: the removals are *committed*, so
each edge is judged against the model as already pruned, and the traversal
order is fixed rather than whatever a set happened to iterate.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig, Transformer, compile_program
from circuitry.attrib.acdc import discover, noise_floor, topological_rank
from circuitry.attrib.circuit import Circuit, Component, all_edges
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.patching import ResampleAblation, ZeroAblation
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs


@pytest.fixture
def compiled():
    program, vocab = programs.build("hist")
    model = compile_program(program, vocab, 4, verify=False)
    pair = compiled_pairs(
        model, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    ).pair
    return model, pair


@pytest.fixture
def smooth():
    g = torch.Generator().manual_seed(1)
    cfg = ModelConfig(
        n_layers=2, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8, dtype="float64"
    )
    model = Transformer(cfg, generator=g).eval()
    clean = torch.randint(0, cfg.d_vocab, (8, 5), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (8, 5), generator=g)
    return model, PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "s")


# ---------------------------------------------------------------------------
# the traversal
# ---------------------------------------------------------------------------


def test_topological_rank_orders_the_forward_pass():
    assert topological_rank(Component.embed()) == -1
    assert topological_rank(Component.attn(0, 0)) < topological_rank(Component.mlp(0))
    assert topological_rank(Component.mlp(0)) < topological_rank(Component.attn(1, 0))
    assert topological_rank(Component.logits(3)) == 6


def test_traversal_is_reverse_topological_and_deterministic(smooth):
    from circuitry.attrib.acdc import _traversal_order

    edges = all_edges(smooth[0].cfg)
    order = _traversal_order(edges)
    ranks = [topological_rank(e.target) for e in order]
    assert ranks == sorted(ranks, reverse=True)
    # And stable: a shuffled input gives the same order out.
    shuffled = list(reversed(edges))
    assert _traversal_order(shuffled) == order


# ---------------------------------------------------------------------------
# the thresholds
# ---------------------------------------------------------------------------


def test_zero_threshold_removes_nothing(smooth):
    """A strict inequality, so a cost of exactly zero is still not below zero."""
    model, pair = smooth
    result = discover(model, pair=pair, threshold=0.0)
    assert result.removed == ()
    assert result.n_kept == len(all_edges(model.cfg))


def test_a_huge_threshold_removes_everything(smooth):
    model, pair = smooth
    result = discover(model, pair=pair, threshold=10.0)
    assert result.n_kept == 0
    assert len(result.removed) == len(all_edges(model.cfg))


def test_the_circuit_shrinks_monotonically_in_the_threshold(smooth):
    model, pair = smooth
    sizes = [
        discover(model, pair=pair, threshold=t).n_kept
        for t in (0.0, 0.01, 0.05, 0.2, 1.0, 10.0)
    ]
    assert sizes == sorted(sizes, reverse=True), sizes


def test_negative_threshold_is_rejected(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="non-negative"):
        discover(model, pair=pair, threshold=-0.1)


# ---------------------------------------------------------------------------
# the mechanism
# ---------------------------------------------------------------------------


def test_removals_are_committed_not_scored_independently(smooth):
    """Each edge is judged against the model as already pruned.

    Visible in the trace: after a cut is committed, the next candidate's
    starting level is the post-cut level, not the original one.  A procedure
    that scored every edge against the full model would show a constant
    starting level throughout, and would remove both halves of a redundant
    pair.
    """
    model, pair = smooth
    result = discover(model, pair=pair, threshold=0.05, keep_trace=True)
    assert result.trace
    starts = [entry["current_before"] for entry in result.trace]
    assert len({round(s, 9) for s in starts}) > 1, "the starting level never moved"
    for previous, entry in zip(result.trace, result.trace[1:], strict=False):
        expected = previous["level"] if previous["cut"] else previous["current_before"]
        assert entry["current_before"] == pytest.approx(expected, abs=1e-12)


def test_discovery_is_a_noising_procedure_by_default(smooth):
    """Nothing cut means the clean prompt, which is level 1."""
    model, pair = smooth
    result = discover(model, pair=pair, threshold=0.0)
    assert result.direction == "noise"
    assert result.counterfactual == "resample"
    assert result.final_level == pytest.approx(1.0, abs=1e-9)


def test_a_counterfactual_can_be_chosen(smooth):
    model, pair = smooth
    result = discover(model, pair=pair, threshold=0.05, counterfactual=ZeroAblation())
    assert result.counterfactual == "zero"


def test_counterfactual_with_denoise_is_refused(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="no meaning when denoising"):
        discover(
            model, pair=pair, threshold=0.05, direction="denoise",
            counterfactual=ResampleAblation(),
        )


def test_an_unimplemented_scorer_says_why(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="dead-gradient"):
        discover(model, pair=pair, threshold=0.05, scorer="attribution")


def test_unknown_direction_is_rejected(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="direction"):
        discover(model, pair=pair, threshold=0.05, direction="both")


# ---------------------------------------------------------------------------
# against a planted circuit
# ---------------------------------------------------------------------------


def test_recovers_the_planted_circuit_with_high_precision(compiled):
    """Precision, not recall: recall is capped by what the prompts exercise.

    See ``metrics.recoverable`` and the test below it.
    """
    model, pair = compiled
    truth = Circuit.from_ground_truth(model.graph, model.cfg)
    found = discover(model.model, pair=pair, threshold=0.02)
    scores = found.circuit.score_against(truth)
    assert scores["edge_precision"] >= 0.5, scores
    assert scores["node_recall"] == pytest.approx(1.0), scores


def test_recall_reaches_its_ceiling(compiled):
    from circuitry.attrib.ablate import AblationEngine
    from circuitry.attrib.metrics import score

    model, pair = compiled
    truth = Circuit.from_ground_truth(model.graph, model.cfg)
    engine = AblationEngine(model.model, pair, "noise", ResampleAblation())
    found = discover(model.model, pair=pair, threshold=0.02)
    scored = score(engine, found.circuit, truth, recall_threshold=1e-9)
    assert scored["edge_recall_vs_recoverable"] == pytest.approx(1.0), scored


def test_cost_is_one_pass_per_edge_plus_the_caches(smooth):
    model, pair = smooth
    result = discover(model, pair=pair, threshold=0.05)
    assert result.forward_passes == len(all_edges(model.cfg)) + 5


# ---------------------------------------------------------------------------
# the noise floor
# ---------------------------------------------------------------------------


def test_noise_floor_is_measured_over_resampled_prompts(smooth):
    """Not the model's determinism -- the model is exactly reproducible.

    The uncertainty is in the prompt sample, so the floor is a bootstrap over
    items on non-trivial edge subsets; the empty set is pinned to the baseline
    and has no spread at all.
    """
    model, pair = smooth
    report = noise_floor(
        model, pair=pair, n_subsets=3, n_boot=50,
        generator=torch.Generator().manual_seed(0),
    )
    assert report["noise_floor"] > 0
    assert report["n_items"] == 8
    assert len(report["per_subset"]) == 3


def test_noise_floor_is_reproducible(smooth):
    model, pair = smooth
    a = noise_floor(model, pair=pair, n_subsets=2, n_boot=50,
                    generator=torch.Generator().manual_seed(3))
    b = noise_floor(model, pair=pair, n_subsets=2, n_boot=50,
                    generator=torch.Generator().manual_seed(3))
    assert a["noise_floor"] == b["noise_floor"]


def test_result_serialises(smooth):
    model, pair = smooth
    payload = discover(model, pair=pair, threshold=0.05).to_dict()
    assert payload["direction"] == "noise"
    assert payload["n_edges_kept"] + payload["n_edges_removed"] == len(
        all_edges(model.cfg)
    )
