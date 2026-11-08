"""Path patching: the effect of one edge, and what an edge decomposition costs.

``test_single_path_edge_equals_node_effect`` is the correctness test: on a
program where a component reaches the output through exactly one route, the
edge effect and the node effect are the same measurement and must agree
exactly.  Everything else about the freezing procedure is a way to get that
wrong.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig, Transformer, compile_program
from circuitry.attrib.circuit import Component
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.patching import ResampleAblation, default_granularity, patch
from circuitry.attrib.paths import path_decomposition_error, path_patch
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs


@pytest.fixture
def smooth():
    """An untrained model: its nonlinearities are not saturated.

    Which matters.  A compiled model's lookup tables are piecewise constant, so
    *every* nonlinear effect vanishes there and an edge decomposition comes out
    exact for a reason that has nothing to do with decomposition being valid.
    """
    g = torch.Generator().manual_seed(7)
    cfg = ModelConfig(
        n_layers=4, n_heads=4, d_model=32, d_mlp=64, d_vocab=40, n_ctx=16, dtype="float64"
    )
    model = Transformer(cfg, generator=g).eval()
    clean = torch.randint(0, cfg.d_vocab, (16, 8), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (16, 8), generator=g)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 16, [1] * 16, position=-1), "smooth")
    return model, pair


def test_single_path_edge_equals_node_effect():
    """`hist` is one head reaching the output through one MLP, and nothing else."""
    program, vocab = programs.build("hist")
    compiled = compile_program(program, vocab, 4, verify=False)
    pair = compiled_pairs(
        compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    ).pair
    node = patch(
        compiled.model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
        granularity=default_granularity(pos="all"),
    ).effects[("blocks.0.attn.z", (("head", 0),))]
    edge = path_patch(
        compiled.model, pair=pair, sender=Component.attn(0, 0),
        receiver=Component.mlp(0), port="mlp_in", direction="denoise",
    )
    assert edge.effect == pytest.approx(node, abs=1e-9)


def test_both_directions_agree_on_a_single_path():
    program, vocab = programs.build("hist")
    compiled = compile_program(program, vocab, 4, verify=False)
    pair = compiled_pairs(
        compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    ).pair
    kwargs = dict(
        pair=pair, sender=Component.attn(0, 0), receiver=Component.mlp(0), port="mlp_in"
    )
    denoise = path_patch(compiled.model, direction="denoise", **kwargs)
    noise = path_patch(
        compiled.model, direction="noise", counterfactual=ResampleAblation(), **kwargs
    )
    # One path means the edge carries everything, so restoring it recovers the
    # clean answer and destroying it loses the same amount.
    assert denoise.effect == pytest.approx(1.0, abs=1e-9)
    assert noise.effect == pytest.approx(-1.0, abs=1e-9)


def test_edge_effects_do_not_sum_to_the_node_effect_on_a_smooth_model(smooth):
    """The reportable number of phase 3, measured rather than assumed small.

    Softmax and the MLP activation make a component's influence non-additive
    over its outgoing edges, and the remainder is of the same order as the
    effect itself -- which bounds what any edge decomposition can mean.
    """
    model, pair = smooth
    out = path_decomposition_error(
        model,
        pair=pair,
        sender=Component.attn(0, 0),
        receivers=[
            Component.attn(1, 0), Component.attn(2, 0), Component.mlp(1),
            Component.logits(4),
        ],
        ports=["v", "v", "mlp_in", "resid"],
        direction="noise",
        counterfactual=ResampleAblation(),
    )
    assert out["n_edges"] == 4
    assert abs(out["relative_discrepancy"]) > 0.1, out


def test_the_compiled_regime_cannot_show_that_discrepancy():
    """A caveat on the regime, not a result about decomposition.

    Compiled lookup tables are piecewise constant, so patching either saturates
    a switch or does nothing, and the remainder is identically zero.  Reporting
    the compiled number as evidence that edge decomposition is sound would be
    exactly backwards.
    """
    program, vocab = programs.build("dyck1")
    compiled = compile_program(program, vocab, 5, verify=False)
    pair = compiled_pairs(
        compiled, n_pairs=16, length=5, generator=torch.Generator().manual_seed(0)
    ).pair
    out = path_decomposition_error(
        compiled.model, pair=pair, sender=Component.mlp(2),
        receivers=[Component.mlp(3), Component.mlp(5)], ports=["mlp_in", "mlp_in"],
        direction="denoise",
    )
    assert out["discrepancy"] == pytest.approx(0.0, abs=1e-9)


def test_freezing_the_other_ports_is_what_isolates_the_value_edge(smooth):
    """Without it, the measurement includes the edges into q and k as well."""
    model, pair = smooth
    kwargs = dict(
        pair=pair, sender=Component.attn(0, 0), receiver=Component.attn(2, 1),
        direction="noise", counterfactual=ResampleAblation(),
    )
    effects = {
        port: path_patch(model, port=port, **kwargs).effect for port in ("q", "k", "v")
    }
    assert len({round(v, 9) for v in effects.values()}) == 3, effects
    for port in ("q", "k", "v"):
        frozen = path_patch(model, port=port, **kwargs).frozen_sites
        others = {p for p in ("q", "k", "v") if p != port}
        for other in others:
            assert any(f"attn.{other}" in site for site in frozen), (port, frozen)


def test_a_head_receiver_requires_a_port(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="needs port"):
        path_patch(
            model, pair=pair, sender=Component.attn(0, 0), receiver=Component.attn(1, 0),
            port="mlp_in", direction="denoise",
        )


def test_an_mlp_receiver_rejects_an_attention_port(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="mlp_in"):
        path_patch(
            model, pair=pair, sender=Component.attn(0, 0), receiver=Component.mlp(1),
            port="q", direction="denoise",
        )


def test_a_downstream_sender_is_rejected(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="not upstream"):
        path_patch(
            model, pair=pair, sender=Component.attn(3, 0), receiver=Component.attn(1, 0),
            port="v", direction="denoise",
        )


def test_the_logit_sink_has_no_output_and_the_embedding_no_inputs(smooth):
    model, pair = smooth
    with pytest.raises(ValueError, match="no output"):
        path_patch(
            model, pair=pair, sender=Component.logits(4), receiver=Component.mlp(1),
            port="mlp_in", direction="denoise",
        )
    with pytest.raises(ValueError, match="no inputs"):
        path_patch(
            model, pair=pair, sender=Component.attn(0, 0), receiver=Component.embed(),
            port="resid", direction="denoise",
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


def test_result_serialises(smooth):
    model, pair = smooth
    payload = path_patch(
        model, pair=pair, sender=Component.attn(0, 0), receiver=Component.mlp(1),
        port="mlp_in", direction="denoise",
    ).to_dict()
    assert payload["sender"] == "a0.h0"
    assert payload["receiver"] == "m1"
    assert payload["port"] == "mlp_in"
    assert payload["forward_passes"] == 5
