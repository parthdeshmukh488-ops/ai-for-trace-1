"""The engine.

The most valuable test here is ``test_resid_decomposition_is_exact``: the
residual stream at any point must equal the sum of everything written into it
so far, and that identity touches the embedding, every head, every MLP and the
order the block composes them in.  It catches more engine bugs than the rest of
this file put together.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig
from circuitry.model import sites
from circuitry.model.hooks import capture, intervene, replace, replace_at
from circuitry.model.transformer import Transformer, act_fn


@pytest.fixture
def model(tiny_config, gen):
    return Transformer(tiny_config, generator=gen).eval()


@pytest.fixture
def tokens(tiny_config, gen):
    return torch.randint(0, tiny_config.d_vocab, (3, 7), generator=gen)


# ---------------------------------------------------------------------------
# shapes and the residual identity
# ---------------------------------------------------------------------------


def test_shapes_at_every_site(model, tokens, tiny_config):
    names = sites.all_sites(tiny_config, include_expensive=True)
    with capture(model, list(names), include_expensive=True) as store:
        model(tokens)
    for name in names:
        expected = sites.shape(name, tiny_config, 3, 7)
        assert tuple(store[name].shape) == expected, name


def test_resid_decomposition_is_exact(model, tokens):
    """resid_post[last] == embed + pos_embed + sum(attn.out) + sum(mlp.out)."""
    terms = model.residual_decomposition(tokens)
    final = terms[f"blocks.{model.cfg.n_layers - 1}.resid_post"]
    assert torch.allclose(terms["sum"], final, atol=1e-13, rtol=0)


def test_attn_result_sums_to_attn_out(model, tokens):
    """The per-head decomposition must reconstitute the thing it decomposes."""
    with capture(model, ["blocks.0.attn.result", "blocks.0.attn.out"],
                 include_expensive=True) as store:
        model(tokens)
    summed = store["blocks.0.attn.result"].sum(dim=2) + model.blocks[0].attn.b_O
    assert torch.allclose(summed, store["blocks.0.attn.out"], atol=1e-13, rtol=0)


def test_attn_result_is_not_computed_unless_wanted(model, tokens):
    """The opt-in is real, not just documented."""
    seen = []
    with intervene(model, {"blocks.0.attn.out": lambda t: seen.append("out") or None}):
        model(tokens)
    assert seen == ["out"]
    assert not model.hooks.wanted("blocks.0.attn.result")


def test_pattern_rows_are_distributions(model, tokens):
    with capture(model, "blocks.*.attn.pattern") as store:
        model(tokens)
    for name in store.keys():
        rows = store[name].sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# the causal mask
# ---------------------------------------------------------------------------


def test_causal_mask_zeroes_the_upper_triangle(model, tokens):
    with capture(model, "blocks.0.attn.pattern") as store:
        model(tokens)
    pattern = store["blocks.0.attn.pattern"]
    upper = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    assert float(pattern[:, :, upper].abs().max()) == 0.0


def test_causal_mask_blocks_future(model, tokens):
    """Patching a masked score entry must be a no-op, not a behaviour change.

    If it were not, an experiment sweeping every entry of the attention pattern
    would spend half its measurements letting the model see the future, and
    report the resulting effects as circuit structure.
    """
    base = model(tokens)
    huge = torch.full((3, 2, 7, 7), 1e4, dtype=base.dtype)
    with intervene(model, {"blocks.0.attn.scores": replace(huge)}):
        patched = model(tokens)
    # Every unmasked entry was set to the same value, so attention becomes
    # uniform over the allowed keys and the output changes -- but the masked
    # entries must still contribute nothing, so the pattern stays triangular.
    with intervene(model, {"blocks.0.attn.scores": replace(huge)}), capture(
        model, "blocks.0.attn.pattern"
    ) as store:
        model(tokens)
    pattern = store["blocks.0.attn.pattern"]
    upper = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    assert float(pattern[:, :, upper].abs().max()) == 0.0
    assert not torch.equal(patched, base)


def test_earlier_positions_do_not_see_later_tokens(model, tiny_config, gen):
    """The structural version of the same claim, without touching a hook."""
    a = torch.randint(0, tiny_config.d_vocab, (1, 7), generator=gen)
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % tiny_config.d_vocab
    out_a, out_b = model(a), model(b)
    assert torch.equal(out_a[:, :-1], out_b[:, :-1])
    assert not torch.equal(out_a[:, -1], out_b[:, -1])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_rejects_sequences_longer_than_the_context(model, tiny_config):
    too_long = torch.zeros(1, tiny_config.n_ctx + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="n_ctx"):
        model(too_long)


def test_rejects_out_of_range_tokens(model, tiny_config):
    with pytest.raises(ValueError, match="token ids"):
        model(torch.tensor([[tiny_config.d_vocab]]))
    with pytest.raises(ValueError, match="token ids"):
        model(torch.tensor([[-1]]))


def test_rejects_wrong_rank(model):
    with pytest.raises(ValueError, match=r"\[batch, pos\]"):
        model(torch.zeros(1, 2, 3, dtype=torch.long))


def test_accepts_an_unbatched_sequence(model, tiny_config):
    out = model(torch.zeros(5, dtype=torch.long))
    assert out.shape == (1, 5, tiny_config.d_vocab)


# ---------------------------------------------------------------------------
# configuration variants
# ---------------------------------------------------------------------------


def test_layernorm_sites_exist_even_without_layernorm(tiny_config, gen, tokens):
    cfg = tiny_config.with_(use_ln=False)
    model = Transformer(cfg, generator=gen).eval()
    with capture(model, ["blocks.0.ln1.scale", "blocks.0.ln1.normalized",
                         "blocks.0.resid_pre"]) as store:
        model(tokens)
    assert torch.equal(store["blocks.0.ln1.scale"], torch.ones(3, 7, 1, dtype=model.dtype))
    assert torch.equal(store["blocks.0.ln1.normalized"], store["blocks.0.resid_pre"])


def test_layernorm_scale_is_the_reciprocal_root_variance(model, tokens):
    with capture(model, ["blocks.0.resid_pre", "blocks.0.ln1.scale"]) as store:
        model(tokens)
    x = store["blocks.0.resid_pre"]
    centred = x - x.mean(dim=-1, keepdim=True)
    expected = (centred.pow(2).mean(dim=-1, keepdim=True) + model.cfg.ln_eps).rsqrt()
    assert torch.allclose(store["blocks.0.ln1.scale"], expected, atol=1e-14, rtol=0)


def test_no_positional_embedding_makes_positions_identical(tiny_config, gen):
    cfg = tiny_config.with_(positional="none")
    model = Transformer(cfg, generator=gen).eval()
    with capture(model, "pos_embed") as store:
        model(torch.zeros(1, 4, dtype=torch.long))
    assert float(store["pos_embed"].abs().max()) == 0.0


def test_sinusoidal_positions_are_bounded_and_distinct(tiny_config, gen):
    cfg = tiny_config.with_(positional="sinusoidal")
    model = Transformer(cfg, generator=gen).eval()
    with capture(model, "pos_embed") as store:
        model(torch.zeros(1, 6, dtype=torch.long))
    pos = store["pos_embed"][0]
    assert float(pos.abs().max()) <= 1.0
    assert not torch.allclose(pos[0], pos[1])


@pytest.mark.parametrize("name", ["gelu_tanh", "gelu_exact", "relu", "identity", "solu"])
def test_every_activation_runs_and_is_shape_preserving(name):
    x = torch.linspace(-3, 3, 11, dtype=torch.float64)
    assert act_fn(name, x).shape == x.shape


def test_gelu_variants_differ_by_about_a_thousandth():
    """Which is why importing weights under the wrong one fails the cross-check."""
    x = torch.linspace(-4, 4, 201, dtype=torch.float64)
    gap = float((act_fn("gelu_tanh", x) - act_fn("gelu_exact", x)).abs().max())
    assert 1e-5 < gap < 1e-2


def test_identity_activation_makes_the_mlp_linear():
    """The configuration phase 4's validation test needs."""
    cfg = ModelConfig(
        n_layers=1, n_heads=1, d_model=8, d_mlp=16, d_vocab=10, n_ctx=8,
        act_fn="identity", use_ln=False, dtype="float64",
    )
    gen = torch.Generator()
    gen.manual_seed(0)
    model = Transformer(cfg, generator=gen).eval()
    block = model.blocks[0]
    x = torch.randn(2, 3, 8, dtype=torch.float64, generator=gen)
    out = block.mlp(x, model.hooks)
    doubled = block.mlp(2 * x, model.hooks)
    assert torch.allclose(doubled - block.mlp.b_out, 2 * (out - block.mlp.b_out), atol=1e-13)


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="unknown activation"):
        act_fn("swiglu", torch.zeros(2))


def test_tied_unembedding_matches_the_embedding(tiny_config, gen):
    model = Transformer(tiny_config, generator=gen)
    assert torch.equal(model.W_U.data, model.W_E.data.T)


def test_untied_unembedding_is_independent(tiny_config, gen):
    model = Transformer(tiny_config.with_(tie_unembed=False), generator=gen)
    assert not torch.equal(model.W_U.data, model.W_E.data.T)


def test_replace_at_writes_one_head_only(model, tokens):
    with capture(model, "blocks.0.attn.z") as store:
        model(tokens)
    before = store["blocks.0.attn.z"]
    hook = replace_at("blocks.0.attn.z", {"head": 1}, 0.0)
    with intervene(model, {"blocks.0.attn.z": hook}), capture(model, "blocks.0.attn.z") as after:
        model(tokens)
    got = after["blocks.0.attn.z"]
    assert torch.equal(got[:, :, 0], before[:, :, 0])
    assert float(got[:, :, 1].abs().max()) == 0.0


def test_device_and_dtype_are_reported(model, tiny_config):
    assert model.dtype == torch.float64
    assert model.device.type == "cpu"
