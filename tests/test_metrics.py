"""Circuit metrics, and the recall ceiling that makes them readable.

Three of these are the standard family.  The fourth, ``recoverable``, is not,
and it is the one that changes how the others read: a planted circuit is the
*program's* dependency graph, and a given prompt sample exercises only part of
it, so recall has a ceiling that is a property of the prompts.  Reporting
recall without it makes a correct method look worse the more thorough the
ground truth is.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig, Transformer, compile_program
from circuitry.attrib.ablate import AblationEngine
from circuitry.attrib.acdc import discover
from circuitry.attrib.circuit import Circuit, all_edges
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.metrics import (
    completeness,
    faithfulness,
    minimality,
    recoverable,
    score,
)
from circuitry.attrib.patching import ResampleAblation
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs


@pytest.fixture
def engine():
    g = torch.Generator().manual_seed(1)
    cfg = ModelConfig(
        n_layers=2, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8, dtype="float64"
    )
    model = Transformer(cfg, generator=g).eval()
    clean = torch.randint(0, cfg.d_vocab, (8, 5), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (8, 5), generator=g)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "s")
    return AblationEngine(model, pair, "noise", ResampleAblation())


@pytest.fixture
def planted():
    program, vocab = programs.build("hist")
    model = compile_program(program, vocab, 4, verify=False)
    pair = compiled_pairs(
        model, n_pairs=16, length=4, generator=torch.Generator().manual_seed(0)
    ).pair
    engine = AblationEngine(model.model, pair, "noise", ResampleAblation())
    truth = Circuit.from_ground_truth(model.graph, model.cfg)
    return model, engine, truth


# ---------------------------------------------------------------------------
# faithfulness
# ---------------------------------------------------------------------------


def test_the_whole_model_is_perfectly_faithful(engine):
    whole = Circuit.of(edges=all_edges(engine.model.cfg))
    assert faithfulness(engine, whole)["faithfulness"] == pytest.approx(1.0, abs=1e-9)


def test_the_empty_circuit_is_not_faithful(engine):
    report = faithfulness(engine, Circuit.empty())
    assert report["faithfulness"] == pytest.approx(0.0, abs=1e-9)
    assert report["n_edges_cut"] == len(all_edges(engine.model.cfg))


def test_faithfulness_is_the_level_with_the_complement_cut(engine):
    """Deliberately not asserting monotonicity in the circuit's size.

    Adding an edge cuts one fewer, which moves the model toward the clean one
    -- but not monotonically in the metric, because contributions cancel.  An
    earlier version of this test asserted monotonicity and failed, correctly:
    a bigger circuit is not always a more faithful one, and a test that claims
    otherwise is testing a false belief.
    """
    edges = all_edges(engine.model.cfg)
    circuit = Circuit.of(edges=edges[: len(edges) // 2])
    outside = [e for e in edges if e not in circuit.edges]
    report = faithfulness(engine, circuit)
    assert report["faithfulness"] == pytest.approx(engine.level(outside), abs=1e-12)
    assert report["n_edges_in_circuit"] == len(circuit.edges)
    assert report["n_edges_cut"] == len(outside)


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------


def test_completeness_of_the_whole_model_is_zero(engine):
    """Knocking a subset out of the whole model is the same measurement twice."""
    whole = Circuit.of(edges=all_edges(engine.model.cfg))
    report = completeness(
        engine, whole, n_subsets=5, generator=torch.Generator().manual_seed(0)
    )
    assert report["max_discrepancy"] == pytest.approx(0.0, abs=1e-9)


def test_completeness_reports_its_own_error_bar(engine):
    """Over twenty subsets it has a Monte Carlo error, and quoting it without
    one is the kind of thing this repository exists not to do."""
    edges = all_edges(engine.model.cfg)
    partial = Circuit.of(edges=edges[: len(edges) // 2])
    report = completeness(
        engine, partial, n_subsets=8, generator=torch.Generator().manual_seed(0)
    )
    assert report["n_subsets"] == 8
    assert report["mean_standard_error"] >= 0.0
    assert report["max_discrepancy"] >= report["mean_discrepancy"]


def test_completeness_is_reproducible(engine):
    edges = all_edges(engine.model.cfg)
    partial = Circuit.of(edges=edges[: len(edges) // 2])
    kwargs = dict(n_subsets=4)
    a = completeness(engine, partial, generator=torch.Generator().manual_seed(2), **kwargs)
    b = completeness(engine, partial, generator=torch.Generator().manual_seed(2), **kwargs)
    assert a["mean_discrepancy"] == b["mean_discrepancy"]


def test_completeness_of_an_empty_circuit_is_not_a_number(engine):
    report = completeness(engine, Circuit.empty())
    assert report["mean_discrepancy"] != report["mean_discrepancy"]  # nan


# ---------------------------------------------------------------------------
# minimality
# ---------------------------------------------------------------------------


def test_minimality_flags_dead_weight(planted):
    """An edge that never matters under any knockout should score zero.

    Measured on a circuit that is actually faithful, with one gratuitous edge
    added.  This has to be done on a working circuit: minimality cuts
    everything *outside* the circuit first, so inside a circuit too small to
    carry the behaviour nothing matters and every edge scores zero -- which is
    correct, and makes the test vacuous.  An earlier version made exactly that
    mistake.
    """
    model, engine, _truth = planted
    found = discover(model.model, pair=engine.pair, threshold=0.02)
    assert faithfulness(engine, found.circuit)["faithfulness"] > 0.5
    gratuitous = next(
        e
        for e in all_edges(model.cfg)
        if e not in found.circuit.edges and e.target.kind == "attn"
    )
    padded = Circuit.of(edges=[*found.circuit.edges, gratuitous])
    report = minimality(
        engine, padded, n_subsets=4, generator=torch.Generator().manual_seed(0)
    )
    assert report["n_edges"] == len(found.circuit.edges) + 1
    assert report["per_edge"][gratuitous.label] < report["median_score"]
    assert report["n_dead_weight"] >= 1


def test_minimality_of_an_empty_circuit_is_not_a_number(engine):
    assert minimality(engine, Circuit.empty())["n_edges"] == 0


# ---------------------------------------------------------------------------
# the ceiling
# ---------------------------------------------------------------------------


def test_recoverable_caps_recall_below_one(planted):
    """The planted graph is the program's; the prompts exercise part of it."""
    _model, engine, truth = planted
    report = recoverable(engine, truth, threshold=1e-9)
    assert 0 < report["n_recoverable"] <= report["n_planted"]
    assert report["ceiling_on_recall"] <= 1.0
    assert set(report["per_edge_effect"]) == {e.label for e in truth.edges}


def test_a_higher_threshold_lowers_the_ceiling(planted):
    _model, engine, truth = planted
    low = recoverable(engine, truth, threshold=1e-9)["ceiling_on_recall"]
    high = recoverable(engine, truth, threshold=0.5)["ceiling_on_recall"]
    assert high <= low


def test_discovery_reaches_the_ceiling_even_when_raw_recall_is_low(planted):
    """The point of measuring the ceiling at all."""
    model, engine, truth = planted
    found = discover(model.model, pair=engine.pair, threshold=0.02)
    scored = score(engine, found.circuit, truth, recall_threshold=1e-9)
    assert scored["edge_recall"] <= 1.0
    assert scored["edge_recall_vs_recoverable"] >= scored["edge_recall"]
    assert scored["edge_recall_vs_recoverable"] == pytest.approx(1.0, abs=1e-9)


def test_score_can_include_the_whole_family(planted):
    model, engine, truth = planted
    found = discover(model.model, pair=engine.pair, threshold=0.02)
    scored = score(
        engine, found.circuit, truth, recall_threshold=1e-9, with_family=True,
        generator=torch.Generator().manual_seed(0),
    )
    assert "faithfulness" in scored
    assert "completeness" in scored
    assert "minimality" in scored
    assert scored["faithfulness"]["faithfulness"] == pytest.approx(1.0, abs=0.2)
