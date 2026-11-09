"""Edge effects, and the two ways of measuring them that do not agree.

The correction this file records is the most consequential one in the
repository so far.  The obvious procedure for patching an edge -- replace the
sender's output, freeze what lies between it and the receiver, re-run -- is
wrong in general, because the sender's patched output stays in the residual
stream after the receiver where every later component also reads it.  Freezing
those is not an option: they have to stay live to respond to the receiver,
which is the thing being measured.

So the procedure measures a bundle of paths.  It is exactly right only when
nothing reads the sender after the receiver, and the test that originally
validated it used ``hist`` -- one head, one MLP, the logits -- which is
precisely that case.

The right definition is arithmetic, in :mod:`circuitry.attrib.ablate`.  The two
are checked against each other here in the regime where they must agree, which
makes the freeze-and-re-run code an independent implementation rather than
dead weight.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig, Transformer, compile_program
from circuitry.attrib.ablate import AblationEngine, check_identity
from circuitry.attrib.circuit import Component, Edge, all_edges
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.patching import ResampleAblation, default_granularity, patch
from circuitry.attrib.paths import (
    path_decomposition_error,
    path_patch,
    path_patch_by_freezing,
)
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs


@pytest.fixture
def smooth():
    """An untrained model: dense connectivity and unsaturated nonlinearities."""
    g = torch.Generator().manual_seed(1)
    cfg = ModelConfig(
        n_layers=3, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8, dtype="float64"
    )
    model = Transformer(cfg, generator=g).eval()
    clean = torch.randint(0, cfg.d_vocab, (8, 6), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (8, 6), generator=g)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "smooth")
    return model, pair


@pytest.fixture
def compiled():
    program, vocab = programs.build("hist")
    model = compile_program(program, vocab, 4, verify=False)
    pair = compiled_pairs(
        model, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    ).pair
    return model, pair


# ---------------------------------------------------------------------------
# the two procedures
# ---------------------------------------------------------------------------


def test_the_two_procedures_agree_where_both_are_valid(smooth):
    """Nothing reads the sender after the final read-out, so they must coincide.

    This is the cross-check: two separately written implementations of the
    same quantity, agreeing to machine precision.
    """
    model, pair = smooth
    sink = Component.logits(model.cfg.n_layers)
    for sender in (Component.mlp(2), Component.attn(2, 0), Component.attn(0, 0)):
        arithmetic = path_patch(
            model, pair=pair, sender=sender, receiver=sink, port="resid", direction="denoise"
        )
        by_freezing = path_patch_by_freezing(
            model, pair=pair, sender=sender, receiver=sink, port="resid", direction="denoise"
        )
        assert arithmetic.effect == pytest.approx(by_freezing.effect, abs=1e-12), sender


def test_the_freezing_procedure_is_wrong_for_an_intermediate_receiver(smooth):
    """And wrong by a lot, which is why the default changed.

    The sender's patched output reaches every live component after the
    receiver, so what gets measured is a bundle of paths.  Measured here at
    0.2 to 0.6 of the clean-to-corrupt gap.
    """
    model, pair = smooth
    gaps = []
    for sender, receiver, port in (
        (Component.attn(0, 0), Component.mlp(1), "mlp_in"),
        (Component.attn(1, 0), Component.mlp(2), "mlp_in"),
    ):
        kwargs = dict(
            pair=pair, sender=sender, receiver=receiver, port=port, direction="denoise"
        )
        gaps.append(
            abs(
                path_patch(model, **kwargs).effect
                - path_patch_by_freezing(model, **kwargs).effect
            )
        )
    assert min(gaps) > 0.1, gaps


# ---------------------------------------------------------------------------
# the ablation engine, which is the definition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["denoise", "noise"])
def test_reconstruction_matches_the_untouched_model_and_path_patching(smooth, direction):
    model, pair = smooth
    cf = None if direction == "denoise" else ResampleAblation()
    report = check_identity(AblationEngine(model, pair, direction, cf))
    assert report["empty_vs_direct"] == pytest.approx(0.0, abs=1e-12)
    # check_identity compares single edges against the freezing procedure, so
    # restrict it to receivers where that procedure is valid.
    sink = Component.logits(model.cfg.n_layers)
    edges = [Edge(Component.attn(0, h), sink, "resid") for h in range(model.cfg.n_heads)]
    report = check_identity(
        AblationEngine(model, pair, direction, cf), edges=edges
    )
    assert report["worst_vs_path_patch"] == pytest.approx(0.0, abs=1e-12)


def test_reconstruction_is_exact_on_a_model_without_layernorm(compiled):
    model, pair = compiled
    report = check_identity(AblationEngine(model.model, pair, "denoise", None))
    assert report["worst_vs_path_patch"] == pytest.approx(0.0, abs=1e-12)
    assert report["port_reconstruction_gap"] == pytest.approx(0.0, abs=1e-12)


def test_freezing_the_layernorm_scale_is_not_free(smooth):
    """Measured, because attribution graphs need it and it changes the answer."""
    model, pair = smooth
    exact = AblationEngine(model, pair, "denoise", None)
    frozen = AblationEngine(model, pair, "denoise", None, freeze_layernorm=True)
    sink = Component.logits(model.cfg.n_layers)
    gaps = [
        abs(exact.level([Edge(s, sink, "resid")]) - frozen.level([Edge(s, sink, "resid")]))
        for s in (Component.mlp(2), Component.attn(2, 0), Component.attn(0, 0))
    ]
    assert max(gaps) > 0.01, gaps


def test_cutting_nothing_gives_the_direction_s_baseline(smooth):
    model, pair = smooth
    assert AblationEngine(model, pair, "denoise", None).level([]) == pytest.approx(0.0, abs=1e-9)
    noise = AblationEngine(model, pair, "noise", ResampleAblation())
    assert noise.level([]) == pytest.approx(1.0, abs=1e-9)


def test_edges_compose(smooth):
    """Cutting a set is not the same as cutting each separately, and must be defined."""
    model, pair = smooth
    engine = AblationEngine(model, pair, "denoise", None)
    sink = Component.logits(model.cfg.n_layers)
    a = Edge(Component.attn(0, 0), sink, "resid")
    b = Edge(Component.attn(0, 1), sink, "resid")
    both = engine.level([a, b])
    assert both != pytest.approx(engine.level([a]), abs=1e-6)
    assert both != pytest.approx(engine.level([b]), abs=1e-6)


def test_ports_of_one_head_are_independent(smooth):
    """The point of carrying a port on an edge at all."""
    model, pair = smooth
    engine = AblationEngine(model, pair, "denoise", None)
    sender, receiver = Component.attn(0, 0), Component.attn(2, 1)
    levels = {
        port: engine.level([Edge(sender, receiver, port)]) for port in ("q", "k", "v")
    }
    assert len({round(v, 9) for v in levels.values()}) == 3, levels


def test_cutting_every_edge_into_a_port_matches_patching_its_input(compiled):
    model, pair = compiled
    report = check_identity(AblationEngine(model.model, pair, "denoise", None))
    assert report["n_edges_in_that_case"] > 0
    assert report["port_reconstruction_gap"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------


def test_single_path_edge_equals_node_effect(compiled):
    """`hist` is one head reaching the output through one MLP, and nothing else."""
    model, pair = compiled
    node = patch(
        model.model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
        granularity=default_granularity(pos="all"),
    ).effects[("blocks.0.attn.z", (("head", 0),))]
    edge = path_patch(
        model.model, pair=pair, sender=Component.attn(0, 0), receiver=Component.mlp(0),
        port="mlp_in", direction="denoise",
    )
    assert edge.effect == pytest.approx(node, abs=1e-9)


def test_edge_effects_do_not_sum_to_the_node_effect(smooth):
    """Measured over *every* edge the sender has, not a sample of them.

    An earlier version enumerated four of forty-one and used the buggy
    freezing procedure, and reported a discrepancy of 95% to 273%.  With the
    correct definition and the full edge set it is a few percent when
    denoising and about a third when noising -- still large enough to bound
    what an edge decomposition can mean, and not what was first reported.
    """
    model, pair = smooth
    sender = Component.attn(0, 0)
    edges = [e for e in all_edges(model.cfg) if e.source == sender]
    assert len(edges) > 10
    out = path_decomposition_error(
        model, pair=pair, sender=sender,
        receivers=[e.target for e in edges], ports=[e.port for e in edges],
        direction="noise", counterfactual=ResampleAblation(),
    )
    assert out["n_edges"] == len(edges)
    assert abs(out["relative_discrepancy"]) > 0.05, out


def test_the_compiled_regime_cannot_show_that_discrepancy():
    """A caveat on the regime, not a result about decomposition.

    Compiled lookup tables are piecewise constant, so a patch either flips a
    switch or does nothing, and every nonlinear interaction vanishes by
    construction.  Reporting the compiled number as evidence that edge
    decomposition is sound would be exactly backwards.
    """
    program, vocab = programs.build("dyck1")
    model = compile_program(program, vocab, 5, verify=False)
    pair = compiled_pairs(
        model, n_pairs=16, length=5, generator=torch.Generator().manual_seed(0)
    ).pair
    sender = Component.mlp(2)
    edges = [e for e in all_edges(model.cfg) if e.source == sender]
    out = path_decomposition_error(
        model.model, pair=pair, sender=sender,
        receivers=[e.target for e in edges], ports=[e.port for e in edges],
        direction="denoise",
    )
    assert out["discrepancy"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_a_head_receiver_requires_one_of_its_own_ports(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="ports are"):
        path_patch(
            model, pair=pair, sender=Component.attn(0, 0), receiver=Component.attn(1, 0),
            port="mlp_in", direction="denoise",
        )


def test_direction_and_counterfactual_are_checked(smooth):
    model, pair = smooth
    kwargs = dict(
        pair=pair, sender=Component.attn(0, 0), receiver=Component.mlp(1), port="mlp_in"
    )
    with pytest.raises(ValueError, match="no meaning when denoising"):
        path_patch(model, direction="denoise", counterfactual=ResampleAblation(), **kwargs)
    with pytest.raises(ValueError, match="needs a counterfactual"):
        path_patch(model, direction="noise", **kwargs)
    with pytest.raises(ValueError, match="direction"):
        path_patch(model, direction="sideways", **kwargs)


def test_the_freezing_procedure_still_validates_its_ports(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="needs port"):
        path_patch_by_freezing(
            model, pair=pair, sender=Component.attn(0, 0), receiver=Component.attn(1, 0),
            port="mlp_in", direction="denoise",
        )


def test_result_serialises(smooth):
    model, pair = smooth
    payload = path_patch(
        model, pair=pair, sender=Component.attn(0, 0), receiver=Component.mlp(1),
        port="mlp_in", direction="denoise",
    ).to_dict()
    assert payload["sender"] == "a0.h0"
    assert payload["receiver"] == "m1"
    assert payload["port"] == "mlp_in"
