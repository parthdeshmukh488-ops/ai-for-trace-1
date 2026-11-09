"""Sparse autoencoders, their training, and the details that silently break them.

``test_recovers_orthogonal_features_exactly`` is the load-bearing one: with no
superposition to undo, recovery must be essentially perfect, and a dictionary
that fails it is broken in a way no loss curve will reveal.  Everything else
here guards one specific footgun named in ``sae.py``'s docstring.
"""

from __future__ import annotations

import pytest
import torch

from circuitry.dictlearn.evaluate import frontier, recovery
from circuitry.dictlearn.sae import SaeConfig, make_sae
from circuitry.dictlearn.train import normalisation_scale, train_sae
from circuitry.toy.superposition import SuperpositionSpec, train_superposition

KINDS = ("relu", "topk", "batchtopk", "jumprelu")


def _cfg(kind, **overrides):
    base = dict(d_in=8, expansion=4, dtype="float64")
    if kind in ("topk", "batchtopk"):
        base["k"] = 3
    else:
        base["lambda_sparsity"] = 1e-3
    return SaeConfig(**{**base, **overrides})


@pytest.fixture
def activations():
    g = torch.Generator()
    g.manual_seed(0)
    return torch.randn(2048, 8, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_every_variant_trains_and_reconstructs(kind, activations):
    sae, report = train_sae(kind, _cfg(kind), activations, steps=200, batch_size=128, seed=0)
    assert report.final_explained_variance > 0.0
    assert 0 < report.final_l0 <= sae.cfg.n_features


@pytest.mark.parametrize("kind", KINDS)
def test_the_decoder_bias_is_subtracted_before_encoding(kind):
    """So the model's claim is `x ~ b_d + sum f_i W_d[:,i]`, a baseline plus features.

    Without it a "feature" can be a constant offset, and the bias is absorbed
    into the dictionary.
    """
    sae = make_sae(kind, _cfg(kind), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        sae.b_d.fill_(3.0)
    x = sae.b_d.detach()[None, :].clone()
    # At exactly the baseline, every pre-activation is just the encoder bias.
    assert torch.allclose(sae.pre_activation(x), sae.b_e.detach()[None, :], atol=1e-12)


def test_topk_l0_is_exactly_k(activations):
    for k in (1, 3, 7):
        sae = make_sae("topk", _cfg("topk", k=k), generator=torch.Generator().manual_seed(0))
        out = sae(activations[:64])
        assert float(out.l0) == pytest.approx(float(k))


def test_topk_rejects_an_impossible_k():
    with pytest.raises(ValueError, match="k must be in"):
        make_sae("topk", _cfg("topk", k=999))
    with pytest.raises(ValueError, match="needs k"):
        make_sae("topk", SaeConfig(d_in=8, expansion=2))


def test_batchtopk_varies_l0_per_example_but_averages_to_k(activations):
    """The point of the architecture: easy examples lend their budget to hard ones."""
    sae = make_sae("batchtopk", _cfg("batchtopk", k=3),
                   generator=torch.Generator().manual_seed(0))
    sae.train()
    out = sae(activations[:128])
    per_item = (out.features > 0).sum(dim=-1)
    assert float(per_item.to(torch.float64).mean()) == pytest.approx(3.0, abs=0.01)
    assert int(per_item.min()) != int(per_item.max()), "L0 did not vary across examples"


def test_batchtopk_inference_threshold_is_estimated_and_required(activations):
    """Train and inference behaviour differ, which is the architecture's one trap."""
    sae = make_sae("batchtopk", _cfg("batchtopk", k=3),
                   generator=torch.Generator().manual_seed(0))
    sae.eval()
    with pytest.raises(RuntimeError, match="no inference threshold"):
        sae(activations[:16])
    sae.train()
    sae(activations[:128])
    sae.eval()
    out = sae(activations[:128])
    # In eval the global threshold stands in for the batch, so L0 is close to
    # k but not pinned to it.
    assert 0 < float(out.l0) < sae.cfg.n_features


def test_jumprelu_gradient_reaches_the_threshold(activations):
    """Both the Heaviside and the L0 count have zero derivative almost everywhere.

    Without the straight-through estimator the thresholds would never move,
    and the architecture would be a ReLU with a frozen offset.
    """
    sae = make_sae("jumprelu", _cfg("jumprelu"), generator=torch.Generator().manual_seed(0))
    out = sae(activations[:128])
    out.loss.backward()
    assert sae.log_theta.grad is not None
    assert float(sae.log_theta.grad.abs().max()) > 0


def test_jumprelu_gradient_pushes_thresholds_up_under_a_sparsity_penalty(activations):
    """The sign matters: the penalty must make thresholds rise, not fall."""
    sae = make_sae(
        "jumprelu", _cfg("jumprelu", lambda_sparsity=1.0),
        generator=torch.Generator().manual_seed(0),
    )
    out = sae(activations[:256])
    sae.sparsity(out.features).backward()
    # Gradient descent subtracts the gradient, so a negative gradient on
    # log_theta is an increase in theta.
    assert float(sae.log_theta.grad.sum()) <= 0


def test_decoder_norm_constraint_holds_for_relu_and_not_for_topk(activations):
    """ReLU needs it -- the L1 penalty is otherwise gamed by rescaling.

    TopK does not: there is no penalty to game, and constraining it would only
    remove capacity.
    """
    for kind, constrained in (("relu", True), ("topk", False)):
        sae, _ = train_sae(kind, _cfg(kind), activations, steps=50, batch_size=128, seed=0)
        norms = sae.decoder_norms().detach()
        if constrained:
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        else:
            assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_the_l1_penalty_is_weighted_by_the_decoder_norm(activations):
    """Penalising the activations alone is trivially gamed by rescaling."""
    sae = make_sae("relu", _cfg("relu"), generator=torch.Generator().manual_seed(0))
    out = sae(activations[:64])
    with torch.no_grad():
        sae.W_d.mul_(0.5)
    rescaled = sae(activations[:64])
    # Halving the decoder doubles the activations needed, so an unweighted
    # penalty would be unchanged; the weighted one is not.
    assert float(rescaled.sparsity_term.detach()) != pytest.approx(
        float(out.sparsity_term.detach()), rel=1e-6
    )


def test_unknown_kind_lists_the_known_ones():
    with pytest.raises(ValueError, match="known:"):
        make_sae("sparsemax", _cfg("relu"))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def test_normalisation_scale_makes_the_mean_norm_root_d(activations):
    scale = normalisation_scale(activations)
    scaled = activations * scale
    assert float(scaled.norm(dim=-1).mean()) == pytest.approx(8**0.5, rel=1e-6)


def test_the_scale_is_reported_because_mse_is_meaningless_without_it(activations):
    _sae, report = train_sae("topk", _cfg("topk"), activations * 100, steps=50,
                             batch_size=128, seed=0)
    assert report.activation_scale != 1.0
    assert report.to_dict()["activation_scale"] == report.activation_scale


def test_the_sparsity_coefficient_is_ramped(activations):
    """At full strength from step one the dictionary dies before it learns anything."""
    _sae, report = train_sae(
        "relu", _cfg("relu", lambda_sparsity=1.0), activations, steps=100,
        batch_size=128, seed=0, lambda_warmup=50,
    )
    assert report.lambda_warmup == 50
    assert report.final_l0 > 0, "the dictionary collapsed; is the warmup applied?"


def test_training_is_reproducible(activations):
    a = train_sae("topk", _cfg("topk"), activations, steps=60, batch_size=128, seed=5)[1]
    b = train_sae("topk", _cfg("topk"), activations, steps=60, batch_size=128, seed=5)[1]
    assert a.final_mse == b.final_mse


def test_resampling_revives_dead_features():
    """A dead feature receives no gradient and cannot recover on its own."""
    g = torch.Generator()
    g.manual_seed(0)
    # A low-rank input, so most of a wide dictionary has nothing to do.
    base = torch.randn(4, 8, generator=g, dtype=torch.float64)
    weights = torch.rand(2048, 4, generator=g, dtype=torch.float64)
    activations = weights @ base
    _sae, report = train_sae(
        "relu", _cfg("relu", expansion=8, lambda_sparsity=5e-2), activations,
        steps=400, batch_size=128, seed=0, resample_every=100,
    )
    assert report.resampled > 0


def test_the_auxiliary_loss_runs_and_changes_the_outcome(activations):
    plain = train_sae("relu", _cfg("relu", lambda_sparsity=5e-2), activations,
                      steps=150, batch_size=128, seed=0)[1]
    auxed = train_sae("relu", _cfg("relu", lambda_sparsity=5e-2), activations,
                      steps=150, batch_size=128, seed=0, aux_k=4)[1]
    assert plain.final_mse != pytest.approx(auxed.final_mse, rel=1e-9)


def test_training_accepts_a_stream_and_reports_the_buffer_size(activations):
    def stream():
        for i in range(0, activations.shape[0], 128):
            yield activations[i : i + 128]

    _sae, report = train_sae(
        "topk", _cfg("topk"), stream(), steps=10, batch_size=64, seed=0,
        buffer_capacity=512,
    )
    assert report.buffer_capacity == 512


# ---------------------------------------------------------------------------
# recovery, against a known basis
# ---------------------------------------------------------------------------


def test_recovers_orthogonal_features_exactly():
    """With no superposition, recovery must be essentially perfect.

    The most informative test in this file: a dictionary that fails it is
    broken, and no reconstruction-loss curve would say so.
    """
    spec = SuperpositionSpec(n_features=8, n_hidden=8, sparsity=0.9)
    model, _ = train_superposition(spec, steps=1500, batch=512, seed=0)
    g = torch.Generator()
    g.manual_seed(1)
    activations = model.hidden(model.sample(8192, generator=g)).detach()
    norms = model.feature_norms()
    represented = norms > 0.1 * norms.max()
    sae, _ = train_sae(
        "topk", _cfg("topk", k=2, expansion=4), activations, steps=2000,
        batch_size=256, lr=3e-3, seed=0,
    )
    found = recovery(sae.directions, model.true_directions, represented=represented)
    assert found["matched_mean_cosine"] > 0.95, found
    assert found["fraction_true_above_0.9"] > 0.9, found


def test_recovery_reports_both_directions():
    """Only the true-to-learned direction is gameable by a bloated dictionary.

    In 32 dimensions, not 4: a random direction in four dimensions is already
    within about 0.7 cosine of its nearest axis, so a low-dimensional version
    of this test cannot distinguish a spurious feature from a real one.  That
    is a fact about high-dimensional geometry and the reason these metrics are
    only informative when the space is wide.
    """
    g = torch.Generator()
    g.manual_seed(0)
    true = torch.eye(32, dtype=torch.float64)
    # Every true feature plus two hundred spurious ones.
    learned = torch.cat([true, torch.randn(200, 32, generator=g, dtype=torch.float64)])
    found = recovery(learned, true)
    assert found["mmcs_true_to_learned"] == pytest.approx(1.0, abs=1e-9)
    assert found["mmcs_learned_to_true"] < 0.8
    assert found["fraction_learned_below_0.5"] > 0.5


def test_matched_cosine_is_not_fooled_by_duplicates():
    """One learned direction per true feature, assigned optimally."""
    true = torch.eye(4, dtype=torch.float64)
    # Four copies of one true feature and nothing else.
    learned = true[:1].repeat(4, 1)
    found = recovery(learned, true)
    assert found["mmcs_true_to_learned"] < 0.6
    assert found["matched_mean_cosine"] < 0.6


def test_recovery_ignores_the_sign_by_default():
    """A dictionary that learns a feature's negative has still found it.

    The sign is absorbed into the activation.  With ``signed=True`` the
    assignment can still avoid the negative diagonal by matching off-diagonal
    zeros, so the score collapses to zero rather than going negative -- which
    is the right behaviour and worth pinning down.
    """
    true = torch.eye(3, dtype=torch.float64)
    assert recovery(-true, true)["matched_mean_cosine"] == pytest.approx(1.0, abs=1e-9)
    signed = recovery(-true, true, signed=True)
    assert signed["matched_mean_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert signed["mmcs_true_to_learned"] == pytest.approx(0.0, abs=1e-9)


def test_recovery_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="dimension mismatch"):
        recovery(torch.zeros(4, 3), torch.zeros(4, 5))
    with pytest.raises(ValueError, match=r"\[n_features, d\]"):
        recovery(torch.zeros(4), torch.zeros(4, 5))


def test_frontier_groups_and_sorts_by_sparsity(activations):
    reports = [
        train_sae("topk", _cfg("topk", k=k), activations, steps=40, batch_size=128, seed=0)[1]
        for k in (6, 2, 4)
    ]
    points = frontier(reports)
    assert set(points) == {"topk"}
    assert [p["l0"] for p in points["topk"]] == sorted(p["l0"] for p in points["topk"])
