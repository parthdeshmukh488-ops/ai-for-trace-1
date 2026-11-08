"""Exact patching: the ground truth phase 4's approximations are measured against.

The two tests that matter most are ``test_patching_identity_is_noop`` -- which
catches a counterfactual being applied to the wrong run, the quietest possible
error here -- and ``test_batched_equals_unbatched``, which licenses the
optimisation that makes phases 4 and 5 affordable at all.
"""

from __future__ import annotations

import pytest
import torch

from circuitry import compile_program
from circuitry.attrib.metric import PromptPair, logit_diff, measure_baselines
from circuitry.attrib.patching import (
    GaussianAblation,
    MeanAblation,
    ResampleAblation,
    ZeroAblation,
    default_granularity,
    patch,
    patch_sites_together,
)
from circuitry.model.transformer import Transformer
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs


@pytest.fixture
def setup(tiny_config, gen):
    model = Transformer(tiny_config, generator=gen).eval()
    clean = torch.randint(0, tiny_config.d_vocab, (6, 5), generator=gen)
    corrupt = torch.randint(0, tiny_config.d_vocab, (6, 5), generator=gen)
    pair = PromptPair(clean, corrupt, logit_diff([0] * 6, [1] * 6), "random")
    return model, pair


@pytest.fixture
def compiled_setup():
    program, vocab = programs.build("hist")
    compiled = compile_program(program, vocab, 4, verify=False)
    report = compiled_pairs(
        compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    )
    return compiled, report.pair


# ---------------------------------------------------------------------------
# the identities
# ---------------------------------------------------------------------------


def test_patching_identity_is_noop(setup):
    """Patching a site with its own value from the same run must change nothing.

    Denoising into the clean run, the patched-in value *is* the current value,
    so the level must be exactly 1.  If a counterfactual were being applied to
    the wrong run, or the two caches swapped, this is where it shows.
    """
    model, pair = setup
    identity = PromptPair(pair.clean, pair.clean, pair.metric, "identity")
    result = patch(
        model, pair=identity, sites=["blocks.0.resid_post"], direction="denoise",
        baselines=measure_baselines(model, pair),
    )
    for level in result.levels.values():
        assert level == pytest.approx(1.0, abs=1e-9)


def test_patching_everything_recovers_clean(setup):
    """Denoising the whole residual stream at the last layer must reach level 1."""
    model, pair = setup
    level = patch_sites_together(
        model,
        pair=pair,
        sites=[f"blocks.{model.cfg.n_layers - 1}.resid_post"],
        direction="denoise",
    )
    assert level == pytest.approx(1.0, abs=1e-6)


def test_noising_everything_destroys_clean(setup):
    model, pair = setup
    level = patch_sites_together(
        model,
        pair=pair,
        sites=[f"blocks.{model.cfg.n_layers - 1}.resid_post"],
        direction="noise",
        counterfactual=ResampleAblation(),
    )
    assert level == pytest.approx(0.0, abs=1e-6)


def test_patching_nothing_leaves_the_base_run(setup):
    model, pair = setup
    denoise = patch_sites_together(
        model, pair=pair, sites=["blocks.0.resid_post"], direction="denoise",
        keep=lambda site, index: True,
    )
    noise = patch_sites_together(
        model, pair=pair, sites=["blocks.0.resid_post"], direction="noise",
        counterfactual=ResampleAblation(), keep=lambda site, index: True,
    )
    assert denoise == pytest.approx(0.0, abs=1e-6)
    assert noise == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def test_batched_equals_unbatched(setup):
    """chunk=1 is the naive one-index-per-pass implementation.

    If replication introduced any cross-talk between replicas the two would
    disagree, and every number in phases 4 and 5 would be wrong by an amount
    nothing else would reveal.
    """
    model, pair = setup
    kwargs = dict(pair=pair, sites="blocks.*.attn.z", direction="denoise")
    one = patch(model, chunk=1, **kwargs)
    many = patch(model, chunk=32, **kwargs)
    assert set(one.effects) == set(many.effects)
    for key, value in one.effects.items():
        assert value == pytest.approx(many.effects[key], abs=1e-12), key
    assert many.forward_passes < one.forward_passes


def test_forward_passes_are_counted_not_estimated(setup):
    model, pair = setup
    result = patch(
        model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
        granularity=default_granularity(pos="all"), chunk=1,
    )
    # Two baseline passes, two cache passes, then one per head.
    assert result.forward_passes == 4 + model.cfg.n_heads


# ---------------------------------------------------------------------------
# direction and counterfactual
# ---------------------------------------------------------------------------


def test_direction_is_required(setup):
    model, pair = setup
    with pytest.raises(TypeError):
        patch(model, pair=pair, sites=["blocks.0.attn.z"])


def test_unknown_direction_raises(setup):
    model, pair = setup
    with pytest.raises(ValueError, match="direction"):
        patch(model, pair=pair, sites=["blocks.0.attn.z"], direction="both")


def test_counterfactual_with_denoise_is_an_error(setup):
    """Silently ignoring it would let an experiment run one measurement four times."""
    model, pair = setup
    with pytest.raises(ValueError, match="no meaning when denoising"):
        patch(
            model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
            counterfactual=ZeroAblation(),
        )


def test_noise_without_a_counterfactual_is_an_error(setup):
    model, pair = setup
    with pytest.raises(ValueError, match="needs a counterfactual"):
        patch(model, pair=pair, sites=["blocks.0.attn.z"], direction="noise")


@pytest.mark.parametrize(
    "counterfactual",
    [
        ZeroAblation(),
        MeanAblation(over="batch"),
        MeanAblation(over="positions"),
        MeanAblation(over="both"),
        ResampleAblation(),
        GaussianAblation(sigma=1.0, generator=torch.Generator().manual_seed(0)),
    ],
)
def test_every_counterfactual_runs_and_names_itself(setup, counterfactual):
    model, pair = setup
    result = patch(
        model, pair=pair, sites=["blocks.0.attn.z"], direction="noise",
        counterfactual=counterfactual,
    )
    assert result.counterfactual == counterfactual.name
    assert len(result) == model.cfg.n_heads * pair.seq


def test_counterfactuals_differ_from_each_other(setup):
    """A guard against two of them being accidentally the same code path."""
    model, pair = setup
    results = {}
    for cf in (ZeroAblation(), MeanAblation(over="batch"), ResampleAblation()):
        result = patch(
            model, pair=pair, sites=["blocks.1.resid_post"], direction="noise",
            counterfactual=cf, granularity=default_granularity(pos="all"),
        )
        results[cf.name] = next(iter(result.effects.values()))
    assert len({round(v, 9) for v in results.values()}) == 3, results


def test_mean_over_positions_destroys_positional_structure():
    """The deliberately-wrong variant must actually behave differently."""
    batch_mean = MeanAblation(over="batch").value(
        "blocks.0.resid_post", torch.arange(24.0).reshape(2, 3, 4), torch.zeros(2, 3, 4)
    )
    pos_mean = MeanAblation(over="positions").value(
        "blocks.0.resid_post", torch.arange(24.0).reshape(2, 3, 4), torch.zeros(2, 3, 4)
    )
    # Averaging over positions makes every position identical; over the batch
    # does not.
    assert torch.allclose(pos_mean[:, 0], pos_mean[:, 1])
    assert not torch.allclose(batch_mean[:, 0], batch_mean[:, 1])


def test_mean_ablation_rejects_an_unknown_averaging_set():
    with pytest.raises(ValueError, match="batch, positions or both"):
        MeanAblation(over="everything")


# ---------------------------------------------------------------------------
# granularity
# ---------------------------------------------------------------------------


def test_granularity_controls_the_number_of_measurements(setup):
    model, pair = setup
    each = patch(
        model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
        granularity=default_granularity(),
    )
    pooled = patch(
        model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
        granularity=default_granularity(pos="all", head="all"),
    )
    assert len(each) == model.cfg.n_heads * pair.seq
    assert len(pooled) == 1


def test_pattern_key_axis_is_pooled_by_default(setup):
    """Iterating it is T times the measurements and is never wanted by accident."""
    model, pair = setup
    result = patch(
        model, pair=pair, sites=["blocks.0.attn.pattern"], direction="denoise",
    )
    assert len(result) == model.cfg.n_heads


def test_unknown_granularity_mode_raises(setup):
    model, pair = setup
    with pytest.raises(ValueError, match="'each' or 'all'"):
        patch(
            model, pair=pair, sites=["blocks.0.attn.z"], direction="denoise",
            granularity={"head": "some"},
        )


# ---------------------------------------------------------------------------
# on a model whose circuit is known
# ---------------------------------------------------------------------------


def test_the_head_the_compiler_used_is_the_one_that_matters(compiled_setup):
    """hist is computed by exactly one head, and patching it must recover most of it."""
    compiled, pair = compiled_setup
    result = patch(
        compiled.model, pair=pair, sites="blocks.*.attn.z", direction="denoise",
    )
    best = result.top(1)[0]
    assert best[0] == "blocks.0.attn.z"
    assert dict(best[1])["head"] == 0
    assert best[2] > 0.5


def test_result_serialises_with_both_levels_and_effects(setup):
    model, pair = setup
    payload = patch(
        model, pair=pair, sites=["blocks.0.attn.z"], direction="noise",
        counterfactual=ResampleAblation(),
    ).to_dict()
    assert payload["direction"] == "noise"
    assert payload["counterfactual"] == "resample"
    assert set(payload["effects"]) == set(payload["levels"])
    assert payload["baselines"]["n_items"] == 6
