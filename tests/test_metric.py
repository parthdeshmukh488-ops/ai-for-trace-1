"""Metrics, the normalisation convention, and the degeneracy filter."""

from __future__ import annotations

import pytest
import torch

from circuitry.attrib.metric import (
    Baselines,
    PromptPair,
    cross_entropy,
    effect_of,
    kl_from,
    logit_diff,
    logprob,
    measure_baselines,
    normalise,
)
from circuitry.model.transformer import Transformer


@pytest.fixture
def model(tiny_config, gen):
    return Transformer(tiny_config, generator=gen).eval()


def test_logit_diff_picks_the_named_tokens():
    logits = torch.zeros(2, 3, 5)
    logits[0, -1, 1] = 4.0
    logits[1, -1, 3] = 2.0
    metric = logit_diff([1, 3], [0, 0])
    assert metric(logits).tolist() == [4.0, 2.0]


def test_logit_diff_accepts_one_position_per_item():
    """Compiled-program pairs differ at different output positions."""
    logits = torch.zeros(2, 4, 5)
    logits[0, 1, 2] = 1.5
    logits[1, 3, 2] = 2.5
    metric = logit_diff([2, 2], [0, 0], position=torch.tensor([1, 3]))
    assert metric(logits).tolist() == [1.5, 2.5]


def test_logit_diff_resolves_negative_positions():
    logits = torch.zeros(1, 4, 3)
    logits[0, 3, 1] = 1.0
    assert logit_diff([1], [0], position=-1)(logits).tolist() == [1.0]
    assert logit_diff([1], [0], position=torch.tensor([-1]))(logits).tolist() == [1.0]


def test_logprob_is_a_log_softmax():
    logits = torch.tensor([[[0.0, 0.0]]])
    assert logprob([0])(logits).item() == pytest.approx(torch.log(torch.tensor(0.5)).item())


def test_kl_from_is_zero_against_itself():
    logits = torch.randn(3, 4, 6, generator=torch.Generator().manual_seed(0))
    assert kl_from(logits)(logits).abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_kl_from_is_positive_otherwise():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 4, 6, generator=g)
    b = torch.randn(3, 4, 6, generator=g)
    assert (kl_from(a)(b) > 0).all()


def test_cross_entropy_matches_torch():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 7, generator=g)
    targets = torch.randint(0, 7, (2, 5), generator=g)
    mine = cross_entropy(targets)(logits)
    reference = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7), targets[:, 1:].reshape(-1), reduction="none"
    ).reshape(2, 4).mean(dim=-1)
    assert torch.allclose(mine, reference, atol=1e-6)


def test_prompt_pair_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        PromptPair(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long),
                   logit_diff([0, 0], [1, 1]))


def test_normalised_metric_endpoints():
    """0 must be corrupt and 1 must be clean, not the other way round."""
    baselines = Baselines(
        clean=torch.tensor([4.0, 6.0]),
        corrupt=torch.tensor([0.0, 2.0]),
        keep=torch.tensor([True, True]),
        threshold=0.1,
    )
    assert normalise(baselines.corrupt, baselines) == pytest.approx(0.0)
    assert normalise(baselines.clean, baselines) == pytest.approx(1.0)
    halfway = (baselines.clean + baselines.corrupt) / 2
    assert normalise(halfway, baselines) == pytest.approx(0.5)


def test_effect_is_measured_from_the_direction_s_starting_level():
    """Denoising starts at 0 and noising starts at 1.

    Reporting the level as an effect would make a component that destroys
    nothing look maximally important, and summing such numbers over a
    component's edges gives an answer larger than the model.
    """
    assert effect_of(0.9, "denoise") == pytest.approx(0.9)
    assert effect_of(0.9, "noise") == pytest.approx(-0.1)
    assert effect_of(0.0, "noise") == pytest.approx(-1.0)


def test_effect_requires_a_known_direction():
    with pytest.raises(ValueError, match="direction"):
        effect_of(0.5, "both")


def test_degenerate_pairs_are_dropped_and_counted(model, tiny_config, gen):
    """A pair whose clean and corrupt scores agree has no signal to recover.

    Dividing by that gap is how a patching plot acquires effects of 40, so the
    items are dropped and the count is reported.
    """
    tokens = torch.randint(0, tiny_config.d_vocab, (4, 5), generator=gen)
    # Identical prompts: the gap is exactly zero for every item.
    pair = PromptPair(tokens, tokens.clone(), logit_diff([0] * 4, [1] * 4))
    baselines = measure_baselines(model, pair, threshold=1e-9)
    assert baselines.kept == 0
    assert baselines.dropped == 4
    assert baselines.report()["n_dropped"] == 4


def test_some_items_kept_and_some_dropped(model, tiny_config, gen):
    tokens = torch.randint(0, tiny_config.d_vocab, (4, 5), generator=gen)
    other = torch.randint(0, tiny_config.d_vocab, (4, 5), generator=gen)
    mixed = tokens.clone()
    mixed[:2] = other[:2]  # two differing pairs, two identical
    pair = PromptPair(tokens, mixed, logit_diff([0] * 4, [1] * 4))
    baselines = measure_baselines(model, pair, threshold=1e-9)
    assert baselines.kept == 2
    assert baselines.dropped == 2


def test_normalise_is_nan_when_everything_was_dropped():
    baselines = Baselines(
        clean=torch.tensor([1.0]), corrupt=torch.tensor([1.0]),
        keep=torch.tensor([False]), threshold=0.1,
    )
    assert normalise(torch.tensor([1.0]), baselines) != normalise(torch.tensor([1.0]), baselines)


def test_default_threshold_scales_with_the_task(model, tiny_config, gen):
    """A fixed absolute threshold would mean different things on different metrics."""
    tokens = torch.randint(0, tiny_config.d_vocab, (8, 5), generator=gen)
    other = torch.randint(0, tiny_config.d_vocab, (8, 5), generator=gen)
    pair = PromptPair(tokens, other, logit_diff([0] * 8, [1] * 8))
    scaled = PromptPair(tokens, other, lambda lg: 1000 * logit_diff([0] * 8, [1] * 8)(lg))
    a = measure_baselines(model, pair)
    b = measure_baselines(model, scaled)
    assert b.threshold == pytest.approx(1000 * a.threshold, rel=1e-6)
    assert a.kept == b.kept
