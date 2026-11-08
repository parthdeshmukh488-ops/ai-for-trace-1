"""The config validates at construction, because everything caches from it."""

from __future__ import annotations

import dataclasses

import pytest

from circuitry import ModelConfig
from circuitry.model.config import ACT_FNS

TINY = dict(n_layers=2, n_heads=2, d_model=16, d_mlp=32, d_vocab=20, n_ctx=16)


def test_d_head_defaults_to_even_split():
    cfg = ModelConfig(**TINY)
    assert cfg.d_head == 8
    assert cfg.d_attn == 16


def test_d_head_may_be_independent_of_d_model():
    """Compiled models need this: their width is whatever their variables sum to."""
    cfg = ModelConfig(**{**TINY, "d_model": 17}, d_head=5)
    assert cfg.d_attn == 10 != cfg.d_model


def test_indivisible_width_without_explicit_d_head_raises():
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(**{**TINY, "d_model": 17})


@pytest.mark.parametrize("field", ["n_layers", "n_heads", "d_model", "d_mlp", "d_vocab", "n_ctx"])
def test_non_positive_dimensions_raise(field):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{**TINY, field: 0})


def test_unknown_act_fn_raises():
    with pytest.raises(ValueError, match="act_fn"):
        ModelConfig(**TINY, act_fn="swiglu")


def test_every_declared_act_fn_is_accepted():
    for name in ACT_FNS:
        assert ModelConfig(**TINY, act_fn=name).act_fn == name


def test_frozen():
    cfg = ModelConfig(**TINY)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_layers = 3


def test_with_recomputes_d_head_when_head_count_changes():
    cfg = ModelConfig(**TINY)
    assert cfg.d_head == 8
    assert cfg.with_(n_heads=4).d_head == 4


def test_with_revalidates():
    with pytest.raises(ValueError):
        ModelConfig(**TINY).with_(act_fn="nope")


def test_non_positive_explicit_d_head_raises():
    with pytest.raises(ValueError, match="d_head"):
        ModelConfig(**TINY, d_head=0)


def test_unknown_positional_scheme_raises():
    with pytest.raises(ValueError, match="positional"):
        ModelConfig(**TINY, positional="rotary")


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="dtype"):
        ModelConfig(**TINY, dtype="int8")


def test_non_positive_ln_eps_raises():
    with pytest.raises(ValueError, match="ln_eps"):
        ModelConfig(**TINY, ln_eps=0.0)
