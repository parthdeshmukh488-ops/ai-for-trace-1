"""Shared fixtures.

The tiny config is the workhorse: small enough that exhaustive tests over its
whole input space finish in milliseconds, large enough to have more than one
head, more than one layer and a non-trivial MLP, which is where the
interesting bugs live.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig

TINY = dict(n_layers=2, n_heads=2, d_model=16, d_mlp=32, d_vocab=20, n_ctx=16)


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(**TINY, dtype="float64")


@pytest.fixture
def tiny_config_f32() -> ModelConfig:
    return ModelConfig(**TINY)


@pytest.fixture
def linear_config() -> ModelConfig:
    """No LayerNorm and an identity activation: the model is linear in its inputs.

    This is the configuration in which a first-order attribution estimate is
    *exact*, so it is what validates the estimator before phase 4 measures the
    estimator's error on a curved model.
    """
    return ModelConfig(**TINY, act_fn="identity", use_ln=False, dtype="float64")


@pytest.fixture
def gen() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(1234)
    return g


@pytest.fixture
def tmp_results(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    return out
