"""Gradient approximations to a patching effect.

``test_attribution_patching_is_exact_in_a_linear_model`` is the test that
licenses phase 4 to exist.  Until it passes, any error the experiment measures
is indistinguishable from a sign bug in the estimator, and the whole
measurement is void.  It is checked at machine precision, not at a tolerance.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import ModelConfig, Transformer
from circuitry.attrib.gradients import attribute
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.patching import ResampleAblation, ZeroAblation, default_granularity, patch

SITES = "blocks.*.attn.z"


@pytest.fixture
def linear():
    """Identity activation, no LayerNorm, one position: the model is linear.

    One position matters as much as the other two: with a single key the
    softmax is identically 1 and contributes no curvature, so the only
    nonlinearity left is none.
    """
    g = torch.Generator().manual_seed(0)
    cfg = ModelConfig(
        n_layers=3, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8,
        act_fn="identity", use_ln=False, dtype="float64",
    )
    model = Transformer(cfg, generator=g).eval()
    clean = torch.randint(0, cfg.d_vocab, (8, 1), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (8, 1), generator=g)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "linear")
    return model, pair


@pytest.fixture
def curved():
    """A normal model: softmax, GELU and LayerNorm all contributing curvature."""
    g = torch.Generator().manual_seed(1)
    cfg = ModelConfig(
        n_layers=3, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8, dtype="float64"
    )
    model = Transformer(cfg, generator=g).eval()
    clean = torch.randint(0, cfg.d_vocab, (8, 6), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (8, 6), generator=g)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "curved")
    return model, pair


# ---------------------------------------------------------------------------
# the validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["denoise", "noise"])
def test_attribution_patching_is_exact_in_a_linear_model(linear, direction):
    model, pair = linear
    cf = None if direction == "denoise" else ResampleAblation()
    exact = patch(model, pair=pair, sites=SITES, direction=direction, counterfactual=cf)
    estimate = attribute(model, pair=pair, sites=SITES, direction=direction, counterfactual=cf)
    assert set(exact.effects) == set(estimate.effects)
    for key, value in exact.effects.items():
        assert estimate.effects[key] == pytest.approx(value, abs=1e-12), key


def test_exactness_holds_on_the_residual_stream_too(linear):
    model, pair = linear
    kwargs = dict(pair=pair, sites="blocks.*.resid_post", direction="denoise")
    exact = patch(model, **kwargs)
    estimate = attribute(model, **kwargs)
    for key, value in exact.effects.items():
        assert estimate.effects[key] == pytest.approx(value, abs=1e-12), key


def test_the_linear_case_has_real_signal(linear):
    """Exactness would be uninformative if every effect were zero."""
    model, pair = linear
    exact = patch(model, pair=pair, sites=SITES, direction="denoise")
    assert max(abs(v) for v in exact.effects.values()) > 0.1


# ---------------------------------------------------------------------------
# on a curved model: the estimate is wrong, and knowably so
# ---------------------------------------------------------------------------


def test_the_estimate_is_not_exact_on_a_curved_model(curved):
    model, pair = curved
    exact = patch(model, pair=pair, sites=SITES, direction="denoise")
    estimate = attribute(model, pair=pair, sites=SITES, direction="denoise")
    worst = max(abs(exact.effects[k] - estimate.effects[k]) for k in exact.effects)
    assert worst > 1e-6, "curvature should produce a measurable error; is the model linear?"


def test_integrated_gradients_reduces_the_error(curved):
    """Averaging along the path should beat expanding at one end of it."""
    model, pair = curved
    kwargs = dict(pair=pair, sites=SITES, direction="denoise",
                  granularity=default_granularity(pos="all"))
    exact = patch(model, **kwargs)

    def worst(result):
        return max(abs(exact.effects[k] - result.effects[k]) for k in exact.effects)

    linear_error = worst(attribute(model, method="linear", **kwargs))
    ig_error = worst(attribute(model, method="integrated", n_steps=32, **kwargs))
    assert ig_error < linear_error, (linear_error, ig_error)


def test_integrated_gradients_converges_at_the_riemann_rate(curved):
    """The error must halve with each doubling of the step count.

    A right-endpoint Riemann sum is ``O(1/n)``, and seeing that rate is the
    evidence that the method converges to the *exact* effect rather than to
    something else.  An earlier version interpolated the whole site while
    taking the dot product at one index, which integrates along a different
    path: it still converged, but to a plateau at 0.30 against effects of
    order 0.9, stable from 8 steps to 128.  Only the rate distinguishes the
    two, which is why this test checks the rate and not merely that more steps
    help.
    """
    model, pair = curved
    kwargs = dict(pair=pair, sites=SITES, direction="denoise",
                  granularity=default_granularity(pos="all"))
    exact = patch(model, **kwargs)
    errors = {}
    for n in (8, 16, 32, 64):
        result = attribute(model, method="integrated", n_steps=n, **kwargs)
        errors[n] = max(abs(exact.effects[k] - result.effects[k]) for k in exact.effects)
    for small, large in ((8, 16), (16, 32), (32, 64)):
        assert errors[small] / errors[large] == pytest.approx(2.0, rel=0.15), errors


def test_integrated_gradients_is_dominated_by_exact_patching(curved):
    """A result, not a sanity check, and the reason the compute axis exists.

    Both methods need one evaluation per index; integrated gradients needs
    ``n_steps`` of them.  So it costs ``n_steps`` times what exact patching
    costs and is less accurate at every step count.  The only gradient method
    that buys anything is the plain linear one, whose cost does not grow with
    the number of indices at all.
    """
    model, pair = curved
    kwargs = dict(pair=pair, sites=SITES, direction="denoise",
                  granularity=default_granularity(pos="all"))
    exact = patch(model, **kwargs)
    exact_items = exact.forward_passes * pair.batch
    ig = attribute(model, method="integrated", n_steps=16, **kwargs)
    linear = attribute(model, method="linear", **kwargs)
    assert ig.batch_items > exact_items
    assert linear.batch_items < exact_items


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------


def test_cost_is_counted_in_batch_items_not_just_passes(curved):
    """Integrated gradients batches its own steps, so passes alone understate it.

    The cost is ``n_steps`` times the number of indices, times the batch.  One
    forward pass per index hides the step count entirely, which is why the
    compute axis in phase 4 is batch items.
    """
    model, pair = curved
    kwargs = dict(pair=pair, sites=["blocks.0.attn.z"], direction="denoise")
    one = attribute(model, method="integrated", n_steps=1, **kwargs)
    many = attribute(model, method="integrated", n_steps=16, **kwargs)
    n_indices = model.cfg.n_heads * pair.seq
    assert len(one) == n_indices
    assert many.forward_passes == one.forward_passes  # one pass per index, either way
    assert many.batch_items - one.batch_items == 15 * pair.batch * n_indices


def test_gradient_dropout_costs_one_backward_per_layer(curved):
    model, pair = curved
    kwargs = dict(pair=pair, sites=["blocks.0.attn.z"], direction="denoise")
    plain = attribute(model, **kwargs)
    dropped = attribute(model, gradient_dropout=True, **kwargs)
    assert dropped.backward_passes == plain.backward_passes + model.cfg.n_layers
    assert dropped.fixes == ("gradient_dropout",)


# ---------------------------------------------------------------------------
# the fixes
# ---------------------------------------------------------------------------


def test_softmax_recomputation_changes_the_q_k_and_scores_estimates(curved):
    model, pair = curved
    kwargs = dict(pair=pair, direction="denoise",
                  granularity=default_granularity(pos="all"))
    for site in ("blocks.1.attn.q", "blocks.1.attn.k", "blocks.1.attn.scores"):
        plain = attribute(model, sites=[site], **kwargs)
        fixed = attribute(model, sites=[site], recompute_softmax=True, **kwargs)
        assert plain.effects != fixed.effects, site
        assert fixed.fixes == ("softmax_recomputation",)


def test_softmax_recomputation_leaves_other_sites_alone(curved):
    """It is expected to help on three site families and nowhere else."""
    model, pair = curved
    kwargs = dict(pair=pair, sites=["blocks.1.attn.z", "blocks.1.mlp.out"],
                  direction="denoise")
    plain = attribute(model, **kwargs)
    fixed = attribute(model, recompute_softmax=True, **kwargs)
    for key in plain.effects:
        assert fixed.effects[key] == pytest.approx(plain.effects[key], abs=1e-12)


def test_softmax_recomputation_helps_on_a_saturated_pattern():
    """The regime the fix exists for: a hard softmax, where the gradient vanishes.

    Built by scaling up the query and key projections of an otherwise ordinary
    model, which saturates the attention while leaving the MLP's GELU smooth --
    so there *is* a gradient to compare against, which is not true of a
    compiled model (see the test below).
    """
    g = torch.Generator().manual_seed(2)
    cfg = ModelConfig(
        n_layers=2, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=8, dtype="float64"
    )
    model = Transformer(cfg, generator=g).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.attn.W_Q *= 40.0
            block.attn.W_K *= 40.0
    clean = torch.randint(0, cfg.d_vocab, (8, 6), generator=g)
    corrupt = torch.randint(0, cfg.d_vocab, (8, 6), generator=g)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 8, [1] * 8, position=-1), "saturated")
    kwargs = dict(pair=pair, sites=["blocks.1.attn.scores"], direction="denoise",
                  granularity=default_granularity(pos="all"))
    exact = patch(model, **kwargs)
    plain = attribute(model, **kwargs)
    fixed = attribute(model, recompute_softmax=True, **kwargs)

    def worst(result):
        return max(abs(exact.effects[k] - result.effects[k]) for k in exact.effects)

    assert worst(fixed) < worst(plain), (worst(plain), worst(fixed))


def test_compiled_models_have_no_gradient_at_most_sites():
    """Why phase 4 cannot be run on compiled models, measured rather than argued.

    A compiled model is exact *because* its lookup tables are piecewise
    constant, and the derivative of a piecewise-constant function is zero.  So
    the gradient is identically zero at a large fraction of sites -- measured
    at a third to two thirds across the suite -- and every gradient-based
    estimate there is exactly 0 while the true effect is large.  An "error"
    measured in that regime is measuring the artefact, not curvature.
    """
    from circuitry import compile_program
    from circuitry.attrib.gradients import _run_with_gradients
    from circuitry.symbolic import programs
    from circuitry.toy.tasks import compiled_pairs

    dead = {}
    for name in ("hist", "sort", "dyck1"):
        program, vocab = programs.build(name)
        compiled = compile_program(program, vocab, 4, verify=False)
        pair = compiled_pairs(
            compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
        ).pair
        sites = [
            f"blocks.{i}.{leaf}"
            for i in range(compiled.cfg.n_layers)
            for leaf in ("attn.z", "mlp.out", "attn.pattern")
        ]
        _, grads = _run_with_gradients(compiled.model, pair.clean, pair.metric, sites)
        alive = sum(1 for value in grads.values() if float(value.abs().max()) > 1e-12)
        dead[name] = 1.0 - alive / len(grads)
    assert min(dead.values()) > 0.3, dead
    assert max(dead.values()) > 0.6, dead


# ---------------------------------------------------------------------------
# validation of arguments
# ---------------------------------------------------------------------------


def test_direction_and_counterfactual_are_checked(curved):
    model, pair = curved
    with pytest.raises(ValueError, match="direction"):
        attribute(model, pair=pair, sites=SITES, direction="both")
    with pytest.raises(ValueError, match="no meaning when denoising"):
        attribute(model, pair=pair, sites=SITES, direction="denoise",
                  counterfactual=ZeroAblation())
    with pytest.raises(ValueError, match="needs a counterfactual"):
        attribute(model, pair=pair, sites=SITES, direction="noise")


def test_unknown_method_and_bad_step_count(curved):
    model, pair = curved
    with pytest.raises(ValueError, match="method"):
        attribute(model, pair=pair, sites=SITES, direction="denoise", method="quadratic")
    with pytest.raises(ValueError, match="n_steps"):
        attribute(model, pair=pair, sites=SITES, direction="denoise",
                  method="integrated", n_steps=0)


def test_result_serialises(curved):
    model, pair = curved
    payload = attribute(
        model, pair=pair, sites=["blocks.0.attn.z"], direction="noise",
        counterfactual=ResampleAblation(), method="integrated", n_steps=4,
        recompute_softmax=True, gradient_dropout=True,
    ).to_dict()
    assert payload["method"] == "integrated"
    assert payload["n_steps"] == 4
    assert set(payload["fixes"]) == {"softmax_recomputation", "gradient_dropout"}
    assert payload["batch_items"] > 0
