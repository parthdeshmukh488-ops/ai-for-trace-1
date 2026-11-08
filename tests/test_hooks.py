"""The hook system, and the contract that every number in the repository rests on.

``test_intervention_is_exactly_reversible`` compares with ``torch.equal``, not
``allclose``.  A leaked hook that perturbs the output slightly is precisely the
failure this exists to catch, and an approximate comparison would let it
through.
"""

from __future__ import annotations

import pytest
import torch

from circuitry.model import sites
from circuitry.model.hooks import (
    HookRegistry,
    add,
    capture,
    freeze,
    intervene,
    project_out,
    replace,
    replace_at,
    scale,
)
from circuitry.model.transformer import Transformer
from circuitry.util.store import ActivationBudgetExceeded


@pytest.fixture
def model(tiny_config, gen):
    return Transformer(tiny_config, generator=gen).eval()


@pytest.fixture
def tokens(tiny_config, gen):
    return torch.randint(0, tiny_config.d_vocab, (2, 5), generator=gen)


# ---------------------------------------------------------------------------
# the reversibility contract
# ---------------------------------------------------------------------------


def test_intervention_is_exactly_reversible(model, tokens):
    before = model(tokens)
    with intervene(model, {"blocks.0.resid_post": scale(0.0)}):
        model(tokens)
    after = model(tokens)
    assert torch.equal(before, after)


def test_reversible_even_when_the_hook_raises(model, tokens):
    before = model(tokens)

    def boom(_tensor):
        raise RuntimeError("deliberate")

    with pytest.raises(RuntimeError, match="deliberate"):
        with intervene(model, {"blocks.0.mlp.post": boom}):
            model(tokens)
    assert torch.equal(before, model(tokens))
    assert len(model.hooks.snapshot()) == 0


def test_capture_is_reversible(model, tokens):
    before = model(tokens)
    with capture(model, "blocks.*.resid_post") as store:
        model(tokens)
    assert len(store) == model.cfg.n_layers
    assert torch.equal(before, model(tokens))


def test_nested_interventions_restore_the_outer_state(model, tokens):
    with intervene(model, {"blocks.0.resid_post": scale(2.0)}):
        outer = model(tokens)
        with intervene(model, {"blocks.1.resid_post": scale(3.0)}):
            inner = model(tokens)
        assert not torch.equal(outer, inner)
        assert torch.equal(outer, model(tokens))


def test_hooks_on_one_site_compose_in_order():
    registry = HookRegistry()
    registry.add("s", lambda t: t + 1)
    registry.add("s", lambda t: t * 10)
    assert float(registry("s", torch.zeros(1))) == 10.0


def test_observe_only_hooks_return_none_and_change_nothing():
    registry = HookRegistry()
    seen = []
    registry.add("s", lambda t: seen.append(float(t)) or None)
    out = registry("s", torch.tensor([3.0]))
    assert seen == [3.0]
    assert float(out) == 3.0


def test_unhooked_sites_pass_through_untouched():
    registry = HookRegistry()
    tensor = torch.zeros(2)
    assert registry("unhooked", tensor) is tensor
    assert not registry.wanted("unhooked")


def test_intervene_rejects_an_unknown_site(model):
    with pytest.raises(sites.SiteNotFound):
        with intervene(model, {"blocks.0.attn.probs": scale(0.0)}):
            pass


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_capture_matches_the_value_actually_used(model, tokens):
    """A captured activation must be the one the forward pass consumed.

    Checked by capturing the MLP's input and output and recomputing the output
    from the input by hand: if the capture were taken from a different place in
    the pass -- before a hook, say, or from a stale buffer -- these would not
    agree.
    """
    with capture(model, ["blocks.0.ln2.normalized", "blocks.0.mlp.out"]) as store:
        model(tokens)
    mlp = model.blocks[0].mlp
    from circuitry.model.transformer import act_fn

    recomputed = act_fn(model.cfg.act_fn, store["blocks.0.ln2.normalized"] @ mlp.W_in + mlp.b_in)
    recomputed = recomputed @ mlp.W_out + mlp.b_out
    assert torch.allclose(recomputed, store["blocks.0.mlp.out"], atol=1e-14, rtol=0)


def test_capture_sees_the_patched_value_not_the_original(model, tokens):
    """Hook order matters: a capture registered after a patch records the patch."""
    zeros = torch.zeros(2, 5, model.cfg.d_model, dtype=model.dtype)
    with intervene(model, {"blocks.0.resid_post": replace(zeros)}):
        with capture(model, "blocks.0.resid_post") as store:
            model(tokens)
    assert float(store["blocks.0.resid_post"].abs().max()) == 0.0


def test_capture_detaches_by_default(model, tokens):
    with capture(model, "logits") as store:
        model(tokens)
    assert not store["logits"].requires_grad


def test_capture_can_keep_the_graph(model, tokens):
    model.W_E.requires_grad_(True)
    try:
        with capture(model, "logits", detach=False) as store:
            model(tokens)
        assert store["logits"].requires_grad
    finally:
        model.W_E.requires_grad_(False)


def test_budget_raises_before_allocating(model, tokens):
    """expect=(batch, seq) turns the budget into a pre-flight check."""
    with pytest.raises(ActivationBudgetExceeded, match="over by"):
        with capture(model, "blocks.*.resid_post", budget_bytes=16, expect=(2, 5)):
            pass


def test_budget_raises_during_capture_without_expect(model, tokens):
    with pytest.raises(ActivationBudgetExceeded):
        with capture(model, "blocks.*.resid_post", budget_bytes=16):
            model(tokens)


def test_capture_accepts_concrete_names_and_patterns(model, tokens):
    with capture(model, ["blocks.0.resid_post", "blocks.1.resid_post"]) as store:
        model(tokens)
    assert len(store) == 2


# ---------------------------------------------------------------------------
# combinators
# ---------------------------------------------------------------------------


def test_replace_checks_the_shape(model, tokens):
    with pytest.raises(ValueError, match="shape"):
        with intervene(model, {"logits": replace(torch.zeros(1))}):
            model(tokens)


def test_scale_zero_is_zero_ablation(model, tokens):
    with intervene(model, {"blocks.0.mlp.out": scale(0.0)}), capture(
        model, "blocks.0.mlp.out"
    ) as store:
        model(tokens)
    assert float(store["blocks.0.mlp.out"].abs().max()) == 0.0


def test_add_steers_and_is_linear_in_strength(model, tokens):
    direction = torch.zeros(model.cfg.d_model, dtype=model.dtype)
    direction[0] = 1.0
    with capture(model, "blocks.1.resid_pre") as base:
        model(tokens)
    with intervene(model, {"blocks.1.resid_pre": add(direction, strength=2.5)}), capture(
        model, "blocks.1.resid_pre"
    ) as store:
        model(tokens)
    delta = store["blocks.1.resid_pre"] - base["blocks.1.resid_pre"]
    assert torch.allclose(delta[..., 0], torch.full_like(delta[..., 0], 2.5), atol=1e-14)
    assert float(delta[..., 1:].abs().max()) == 0.0


def test_add_at_selected_positions_only(model, tokens):
    direction = torch.ones(model.cfg.d_model, dtype=model.dtype)
    with capture(model, "blocks.1.resid_pre") as base:
        model(tokens)
    with intervene(model, {"blocks.1.resid_pre": add(direction, positions=[0])}), capture(
        model, "blocks.1.resid_pre"
    ) as store:
        model(tokens)
    delta = store["blocks.1.resid_pre"] - base["blocks.1.resid_pre"]
    assert float(delta[:, 0].min()) == pytest.approx(1.0)
    assert float(delta[:, 1:].abs().max()) == 0.0


def test_project_out_removes_the_component(model, tokens):
    direction = torch.zeros(model.cfg.d_model, dtype=model.dtype)
    direction[3] = 1.0
    with intervene(model, {"blocks.1.resid_pre": project_out(direction)}), capture(
        model, "blocks.1.resid_pre"
    ) as store:
        model(tokens)
    assert float(store["blocks.1.resid_pre"][..., 3].abs().max()) < 1e-13


def test_project_out_orthonormalises_before_subtracting():
    """Two near-parallel directions must not remove the shared part twice."""
    d = 6
    a = torch.zeros(d, dtype=torch.float64)
    a[0] = 1.0
    b = a.clone()
    b[1] = 1e-3
    hook = project_out(torch.stack([a, b]))
    x = torch.ones(1, 1, d, dtype=torch.float64)
    out = hook(x)
    # The span of {a, b} is the first two coordinates, so exactly those go.
    assert float(out[..., :2].abs().max()) < 1e-12
    assert torch.allclose(out[..., 2:], x[..., 2:], atol=1e-14)


def test_freeze_pins_a_site_to_a_cached_value(model, tokens, tiny_config, gen):
    other = torch.randint(0, tiny_config.d_vocab, (2, 5), generator=gen)
    with capture(model, "blocks.0.attn.out") as cached:
        model(other)
    with intervene(model, {"blocks.0.attn.out": freeze(cached, "blocks.0.attn.out")}), capture(
        model, "blocks.0.attn.out"
    ) as store:
        model(tokens)
    assert torch.equal(store["blocks.0.attn.out"], cached["blocks.0.attn.out"])


def test_freeze_rejects_a_cache_from_a_different_shape(model, tokens):
    with capture(model, "blocks.0.attn.out") as cached:
        model(tokens[:, :3])
    with pytest.raises(ValueError, match="different batch or sequence length"):
        with intervene(model, {"blocks.0.attn.out": freeze(cached, "blocks.0.attn.out")}):
            model(tokens)


def test_replace_at_on_an_attention_pattern_uses_the_right_axes(model, tokens):
    """The same combinator must work on [B,H,T,T] as on [B,T,H,dh]."""
    hook = replace_at("blocks.0.attn.pattern", {"head": 0, "query": 2}, 0.0)
    with intervene(model, {"blocks.0.attn.pattern": hook}), capture(
        model, "blocks.0.attn.pattern"
    ) as store:
        model(tokens)
    pattern = store["blocks.0.attn.pattern"]
    assert float(pattern[:, 0, 2].abs().max()) == 0.0
    assert float(pattern[:, 1, 2].abs().max()) > 0.0


def test_registry_repr():
    registry = HookRegistry()
    registry.add("s", lambda t: None)
    assert "1 sites" in repr(registry)


def test_any_wanted_reports_across_a_group():
    registry = HookRegistry()
    registry.add("b", lambda t: None)
    assert registry.any_wanted(["a", "b"])
    assert not registry.any_wanted(["a", "c"])


def test_clear_removes_everything():
    registry = HookRegistry()
    registry.add("a", lambda t: None)
    registry.clear()
    assert not registry.wanted("a")


def test_replace_at_accepts_a_tensor_value(model, tokens):
    """The combinator must move a tensor onto the site's dtype and device."""
    value = torch.zeros(2, 5, model.cfg.d_head, dtype=torch.float32)
    hook = replace_at("blocks.0.attn.z", {"head": 0}, value)
    with intervene(model, {"blocks.0.attn.z": hook}), capture(
        model, "blocks.0.attn.z"
    ) as store:
        model(tokens)
    assert float(store["blocks.0.attn.z"][:, :, 0].abs().max()) == 0.0


def test_intervene_accepts_a_list_of_hooks_for_one_site(model, tokens):
    with intervene(model, {"blocks.0.resid_post": [scale(0.0), add(
        torch.ones(model.cfg.d_model, dtype=model.dtype)
    )]}), capture(model, "blocks.0.resid_post") as store:
        model(tokens)
    got = store["blocks.0.resid_post"]
    assert torch.allclose(got, torch.ones_like(got), atol=1e-14)
