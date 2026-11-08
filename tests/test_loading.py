"""The three weight folds, and the claim that each is exact.

"Exactly behaviour-preserving" is a claim, so each fold has a test asserting
the model's function is unchanged.  What counts as "the function" differs by
fold, and getting that wrong is how the cross-check first failed on a correct
implementation:

* ``fold_layernorm`` leaves the logits alone, and leaves the residual stream
  alone, but changes ``ln_final.normalized`` -- the final gain and bias now
  live in the unembedding.
* ``center_writing_weights`` leaves the logits alone and shifts the residual
  stream by a multiple of the all-ones vector, which the next LayerNorm
  removes.
* ``center_unembed`` shifts every logit at a position by the same constant, so
  the log-softmax and the argmax are preserved and the raw logits are not.

The tests run in float64: these are algebraic identities, and checking them in
float32 measures the identity and the accumulation noise together.
"""

from __future__ import annotations

import pytest
import torch

from circuitry.model.hooks import capture
from circuitry.model.loading import (
    FoldingNotApplicable,
    center_unembed,
    center_writing_weights,
    fold_layernorm,
)
from circuitry.model.transformer import Transformer

ATOL = 1e-11


@pytest.fixture
def pair(tiny_config, gen):
    """Two identical models, so one can be folded and compared against the other."""
    seed = 20260108
    a = Transformer(tiny_config, generator=torch.Generator().manual_seed(seed)).eval()
    b = Transformer(tiny_config, generator=torch.Generator().manual_seed(seed)).eval()
    # A fold is only interesting if the gains and biases are not already trivial.
    with torch.no_grad():
        for model in (a, b):
            for block in model.blocks:
                block.ln1.weight.copy_(torch.linspace(0.5, 1.5, model.cfg.d_model))
                block.ln1.bias.copy_(torch.linspace(-0.2, 0.2, model.cfg.d_model))
                block.ln2.weight.copy_(torch.linspace(1.5, 0.5, model.cfg.d_model))
                block.ln2.bias.copy_(torch.linspace(0.2, -0.2, model.cfg.d_model))
            model.ln_final.weight.copy_(torch.linspace(0.8, 1.2, model.cfg.d_model))
            model.ln_final.bias.copy_(torch.linspace(0.1, -0.1, model.cfg.d_model))
    assert torch.equal(a.blocks[0].ln1.weight, b.blocks[0].ln1.weight)
    return a, b


@pytest.fixture
def tokens(tiny_config, gen):
    return torch.randint(0, tiny_config.d_vocab, (2, 6), generator=gen)


def test_fold_ln_preserves_logits(pair, tokens):
    plain, folded = pair
    before = plain(tokens)
    fold_layernorm(folded)
    assert torch.allclose(folded(tokens), before, atol=ATOL, rtol=0)


def test_fold_ln_actually_changed_the_weights(pair, tokens):
    """Guard against a fold that preserves logits by doing nothing."""
    plain, folded = pair
    fold_layernorm(folded)
    assert torch.equal(folded.blocks[0].ln1.weight, torch.ones_like(folded.blocks[0].ln1.weight))
    assert float(folded.blocks[0].ln1.bias.detach().abs().max()) == 0.0
    assert not torch.allclose(folded.blocks[0].attn.W_Q, plain.blocks[0].attn.W_Q)


def test_fold_ln_leaves_the_residual_stream_alone(pair, tokens):
    plain, folded = pair
    fold_layernorm(folded)
    with capture(plain, "blocks.1.resid_pre") as a, torch.no_grad():
        plain(tokens)
    with capture(folded, "blocks.1.resid_pre") as b, torch.no_grad():
        folded(tokens)
    assert torch.allclose(a["blocks.1.resid_pre"], b["blocks.1.resid_pre"], atol=ATOL, rtol=0)


def test_center_writing_weights_preserves_logits(pair, tokens):
    plain, centred = pair
    before = plain(tokens)
    center_writing_weights(centred)
    assert torch.allclose(centred(tokens), before, atol=ATOL, rtol=0)


def test_center_writing_weights_shifts_the_stream_by_a_constant(pair, tokens):
    """The measured fact behind the cross-check's fold-invariant comparison."""
    plain, centred = pair
    center_writing_weights(centred)
    with capture(plain, "blocks.1.resid_pre") as a, torch.no_grad():
        plain(tokens)
    with capture(centred, "blocks.1.resid_pre") as b, torch.no_grad():
        centred(tokens)
    x, y = a["blocks.1.resid_pre"], b["blocks.1.resid_pre"]
    assert float((x - y).abs().max()) > 1e-3  # the raw streams differ
    demeaned = (x - x.mean(-1, keepdim=True)) - (y - y.mean(-1, keepdim=True))
    assert float(demeaned.abs().max()) < ATOL  # and differ only by a constant


def test_center_unembed_preserves_argmax_and_log_softmax(pair, tokens):
    plain, centred = pair
    before = plain(tokens)
    center_unembed(centred)
    after = centred(tokens)
    assert torch.equal(before.argmax(-1), after.argmax(-1))
    assert torch.allclose(before.log_softmax(-1), after.log_softmax(-1), atol=ATOL, rtol=0)


def test_center_unembed_does_change_the_raw_logits(pair, tokens):
    plain, centred = pair
    before = plain(tokens)
    center_unembed(centred)
    assert not torch.allclose(centred(tokens), before, atol=1e-6, rtol=0)


def test_all_three_folds_together_preserve_the_log_softmax(pair, tokens):
    plain, folded = pair
    before = plain(tokens).log_softmax(-1)
    center_unembed(center_writing_weights(fold_layernorm(folded)))
    assert torch.allclose(folded(tokens).log_softmax(-1), before, atol=ATOL, rtol=0)


def test_folds_that_need_layernorm_refuse_without_it(tiny_config, gen):
    model = Transformer(tiny_config.with_(use_ln=False), generator=gen)
    with pytest.raises(FoldingNotApplicable, match="LayerNorm"):
        fold_layernorm(model)
    with pytest.raises(FoldingNotApplicable, match="behaviour-preserving"):
        center_writing_weights(model)


def test_center_unembed_works_without_layernorm(tiny_config, gen):
    """It is a statement about the softmax, not about the stream."""
    model = Transformer(tiny_config.with_(use_ln=False), generator=gen).eval()
    tokens = torch.zeros(1, 4, dtype=torch.long)
    before = model(tokens).log_softmax(-1)
    center_unembed(model)
    assert torch.allclose(model(tokens).log_softmax(-1), before, atol=ATOL, rtol=0)
