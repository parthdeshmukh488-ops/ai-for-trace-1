"""The cost model, checked against arithmetic done by hand.

Every comparison in this repository that has compute on an axis is downstream
of this module, so an error here rescales results rather than breaking them --
which is the kind of error a test has to catch because nothing else will.
"""

from __future__ import annotations

import pytest

from circuitry import ModelConfig
from circuitry.util import flops


def test_forward_matches_hand_computation(tiny_config):
    """Worked out by hand for L=2, H=2, d_head=8, d=16, d_mlp=32, V=20, S=4.

    projections  S * L * 4 * 2 * d * d_attn  = 4 * 2 * 4 * 2 * 16 * 16 = 16384
    attention    L * 2 * 2 * sum(1..4) * d_attn = 2 * 4 * 10 * 16      =  1280
    mlp          S * L * 2 * 2 * d * d_mlp  = 4 * 2 * 4 * 16 * 32      = 16384
    unembed      S * 2 * d * V              = 4 * 2 * 16 * 20          =  2560
    """
    f = flops.forward_flops(tiny_config, 4)
    assert f.projections == 16384
    assert f.attention == 1280
    assert f.mlp == 16384
    assert f.unembed == 2560
    assert f.total == 36608
    assert int(f) == f.total


def test_attention_term_is_triangular_not_quadratic(tiny_config):
    """A causal mask means sum(1..S) keys, not S*S; the difference is a factor of ~2."""
    s = 8
    f = flops.forward_flops(tiny_config, s)
    triangular = s * (s + 1) // 2
    expected = tiny_config.n_layers * 2 * 2 * triangular * tiny_config.d_attn
    assert f.attention == expected
    naive_square = tiny_config.n_layers * 2 * 2 * s * s * tiny_config.d_attn
    assert f.attention < naive_square


def test_unembedding_is_a_large_fraction_for_gpt2_small():
    """Not a rounding error: the term most often dropped is a third of the total.

    The assertion is deliberately loose -- it is checking that the term is of
    the same order as the layer stack, which is the claim the module docstring
    makes, not a specific ratio.
    """
    cfg = ModelConfig(n_layers=12, n_heads=12, d_model=768, d_mlp=3072, d_vocab=50257, n_ctx=1024)
    f = flops.forward_flops(cfg, 1)
    assert f.unembed / f.total > 0.25


def test_decode_excludes_prefill_and_grows_with_context(tiny_config):
    one = flops.decode_flops(tiny_config, prefix_len=10, n_new=1)
    later = flops.decode_flops(tiny_config, prefix_len=100, n_new=1)
    # Only the attention term depends on the context length.
    assert later.attention > one.attention
    assert later.projections == one.projections
    assert later.mlp == one.mlp


def test_decode_is_additive_over_steps(tiny_config):
    """Decoding n tokens at once must cost what decoding them one at a time does."""
    bulk = flops.decode_flops(tiny_config, prefix_len=5, n_new=4).total
    stepwise = sum(
        flops.decode_flops(tiny_config, prefix_len=5 + i, n_new=1).total for i in range(4)
    )
    assert bulk == stepwise


def test_backward_is_twice_forward_and_says_it_is_approximate(tiny_config):
    fwd = flops.forward_flops(tiny_config, 4)
    bwd = flops.backward_flops(tiny_config, 4)
    assert bwd.total == 2 * fwd.total
    assert "approximation" in bwd.excludes


def test_count_names_what_it_excludes(tiny_config):
    f = flops.forward_flops(tiny_config, 4)
    assert "layernorm" in f.excludes
    assert "softmax" in f.excludes


def test_batch_scales_linearly(tiny_config):
    one = flops.forward_flops(tiny_config, 6, batch=1).total
    five = flops.forward_flops(tiny_config, 6, batch=5).total
    assert five == 5 * one


def test_unembed_all_false_counts_only_the_final_position(tiny_config):
    full = flops.forward_flops(tiny_config, 8, unembed_all=True)
    last = flops.forward_flops(tiny_config, 8, unembed_all=False)
    assert last.unembed * 8 == full.unembed
    assert last.projections == full.projections


@pytest.mark.parametrize(
    "kwargs",
    [dict(seq_len=0), dict(seq_len=-1), dict(seq_len=4, batch=0)],
)
def test_rejects_degenerate_shapes(tiny_config, kwargs):
    with pytest.raises(ValueError):
        flops.forward_flops(tiny_config, **kwargs)


def test_decode_rejects_degenerate_shapes(tiny_config):
    with pytest.raises(ValueError):
        flops.decode_flops(tiny_config, prefix_len=-1, n_new=1)
    with pytest.raises(ValueError):
        flops.decode_flops(tiny_config, prefix_len=1, n_new=0)
