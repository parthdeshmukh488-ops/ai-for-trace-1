"""Export to the independently written implementation, and the round trip back.

Marked ``oracle`` because it needs the reference library.  It does *not* need
the network: a randomly initialised model is exported and compared, which
tests the forward pass with the weight mapping out of the way.  The arm that
loads a real checkpoint is marked ``network`` and is excluded from CI, because
a test that fails when a download fails is not testing this code.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig
from circuitry.model.loading import (
    from_reference_state_dict,
    reference_config,
    to_reference,
)
from circuitry.model.transformer import Transformer

pytest.importorskip("transformers")
pytestmark = pytest.mark.oracle

EXPORTABLE = dict(n_layers=2, n_heads=2, d_model=16, d_mlp=32, d_vocab=20, n_ctx=16)


@pytest.fixture
def model():
    cfg = ModelConfig(**EXPORTABLE)
    return Transformer(cfg, generator=torch.Generator().manual_seed(3)).eval()


def test_logits_agree_with_the_reference(model):
    reference = to_reference(model)
    tokens = torch.randint(0, model.cfg.d_vocab, (2, 6),
                           generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        mine = model(tokens)
        theirs = reference(tokens).logits
    deviation = float((mine - theirs).abs().max())
    # float32 reassociation only; the benchmark reports the achieved figure.
    assert deviation < 1e-5, deviation


def test_export_then_import_recovers_every_weight(model):
    """The two mappings are inverses, which is what makes either one checkable."""
    reference = to_reference(model)
    back = from_reference_state_dict(model.cfg, reference.state_dict())
    for (name, a), (_, b) in zip(
        model.named_parameters(), back.named_parameters(), strict=True
    ):
        assert torch.allclose(a, b, atol=1e-7, rtol=0), name


def test_reference_config_maps_the_activation_name():
    hf_config = to_reference(Transformer(ModelConfig(**EXPORTABLE))).config
    assert reference_config(hf_config).act_fn == "gelu_tanh"
    assert reference_config(hf_config).n_layers == EXPORTABLE["n_layers"]


def test_export_refuses_an_activation_the_reference_does_not_have():
    cfg = ModelConfig(**EXPORTABLE, act_fn="relu")
    with pytest.raises(ValueError, match="gelu_new"):
        to_reference(Transformer(cfg))


def test_export_refuses_a_model_without_layernorm():
    cfg = ModelConfig(**EXPORTABLE, use_ln=False)
    with pytest.raises(ValueError, match="LayerNorm"):
        to_reference(Transformer(cfg))


def test_export_refuses_sinusoidal_positions():
    cfg = ModelConfig(**EXPORTABLE, positional="sinusoidal")
    with pytest.raises(ValueError, match="learned positions"):
        to_reference(Transformer(cfg))


def test_export_refuses_a_head_width_mismatch():
    cfg = ModelConfig(**{**EXPORTABLE, "d_model": 17}, d_head=5)
    with pytest.raises(ValueError, match="d_head == d_model"):
        to_reference(Transformer(cfg))


def test_export_refuses_a_nonzero_unembedding_bias(model):
    with torch.no_grad():
        model.b_U.fill_(0.1)
    with pytest.raises(ValueError, match="unembedding bias"):
        to_reference(model)


def test_import_names_a_missing_key():
    cfg = ModelConfig(**EXPORTABLE)
    with pytest.raises(KeyError, match=r"wte\.weight"):
        from_reference_state_dict(cfg, {"nonsense": torch.zeros(1)})


@pytest.mark.network
@pytest.mark.slow
def test_real_checkpoint_round_trip():
    """The weight mapping against a real checkpoint.  Needs a download."""
    from circuitry.model.loading import load_gpt2

    model = load_gpt2("gpt2", fold_ln=False, center_writing=False, center_unembedding=False)
    assert model.cfg.n_layers == 12
    assert model.cfg.d_model == 768
    assert model.cfg.act_fn == "gelu_tanh"
