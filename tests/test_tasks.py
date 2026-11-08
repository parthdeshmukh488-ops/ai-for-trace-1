"""Prompt-pair generation, and the discard rate it is obliged to report."""

from __future__ import annotations

import pytest
import torch

from circuitry import compile_program
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs


def _compiled(name, max_len=4):
    program, vocab = programs.build(name)
    return compile_program(program, vocab, max_len, verify=False)


def test_pairs_have_differing_outputs_at_the_recorded_position():
    compiled = _compiled("sort")
    report = compiled_pairs(
        compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    )
    for item in range(report.n_found):
        clean = compiled.decode(compiled.model(report.pair.clean[item : item + 1]))
        corrupt = compiled.decode(compiled.model(report.pair.corrupt[item : item + 1]))
        at = report.positions[item] - 1  # model position 0 is the BOS symbol
        assert clean[at] != corrupt[at]


def test_discards_are_counted_and_reported():
    """A reader who is not told the rate cannot tell an easy task from a hard one."""
    compiled = _compiled("sort")
    report = compiled_pairs(
        compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    )
    payload = report.to_dict()
    assert payload["n_discarded"] >= 0
    assert payload["discard_rate"] == pytest.approx(
        report.n_discarded / (report.n_found + report.n_discarded)
    )
    assert payload["program"] == "sort"


def test_the_resulting_pairs_are_non_degenerate():
    from circuitry.attrib.metric import measure_baselines

    compiled = _compiled("hist")
    report = compiled_pairs(
        compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
    )
    baselines = measure_baselines(compiled.model, report.pair)
    assert baselines.dropped == 0


def test_generation_is_reproducible():
    compiled = _compiled("hist")
    a = compiled_pairs(compiled, n_pairs=6, length=4,
                       generator=torch.Generator().manual_seed(3))
    b = compiled_pairs(compiled, n_pairs=6, length=4,
                       generator=torch.Generator().manual_seed(3))
    assert torch.equal(a.pair.clean, b.pair.clean)
    assert a.positions == b.positions


def test_a_numerical_root_is_refused_with_the_alternative():
    compiled = _compiled("pair_balance")
    with pytest.raises(ValueError, match="numerical output"):
        compiled_pairs(compiled, n_pairs=4, length=4)


def test_an_impossible_length_is_rejected():
    compiled = _compiled("hist", max_len=4)
    with pytest.raises(ValueError, match="length must be between"):
        compiled_pairs(compiled, n_pairs=4, length=9)


def test_a_constant_program_raises_rather_than_returning_nothing():
    """Better a clear error than a silently empty batch that measures nothing."""
    from circuitry.symbolic.lang import Map, tokens

    constant = Map(lambda _t: "z", tokens, declared_support=("z",), label="constant")
    compiled = compile_program(constant, ("a", "b"), 3, verify=False)
    with pytest.raises(ValueError, match="no pair"):
        compiled_pairs(compiled, n_pairs=4, length=3, max_attempts=50)
