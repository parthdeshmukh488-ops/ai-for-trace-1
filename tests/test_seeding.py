"""Explicit randomness, checked for the property that actually matters.

Reproducibility is easy to claim and easy to lose.  The tests that matter here
are the ones about *independence*: a generator's stream must not depend on what
anybody else drew, and a derived seed must not depend on how many items were
processed before it.  Those are the properties that make a sweep's individual
runs reproducible in isolation, which is what you need when one configuration
out of ninety behaved oddly.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from circuitry.util.seeding import derive_seed, generator, seed_everything


def test_same_seed_gives_the_same_stream():
    a = torch.randn(8, generator=generator(0))
    b = torch.randn(8, generator=generator(0))
    assert torch.equal(a, b)


def test_different_seeds_give_different_streams():
    a = torch.randn(8, generator=generator(0))
    b = torch.randn(8, generator=generator(1))
    assert not torch.equal(a, b)


def test_a_generators_stream_is_independent_of_other_draws():
    """The property that makes library code safe to call in any order.

    Drawing from an unrelated generator -- or from the global one, which is
    what a stray diagnostic would do -- must not shift this generator's stream.
    """
    gen = generator(7)
    first = torch.randn(4, generator=gen)

    gen = generator(7)
    torch.randn(100)  # global draws
    torch.randn(100, generator=generator(999))  # an unrelated generator
    random.random()
    second = torch.randn(4, generator=gen)

    assert torch.equal(first, second)


def test_generator_rejects_a_non_integer_seed():
    with pytest.raises(TypeError, match="seed must be an int"):
        generator("zero")  # type: ignore[arg-type]


def test_derive_seed_is_deterministic():
    assert derive_seed(0, "node", 3) == derive_seed(0, "node", 3)


def test_derive_seed_separates_parts_from_their_concatenation():
    """``("a", "bc")`` and ``("ab", "c")`` must not collide.

    They would under naive string concatenation, which is the obvious
    implementation and is wrong: two different tree nodes would then share a
    sampling stream, and the duplication would look like a model behaviour.
    """
    assert derive_seed(0, "a", "bc") != derive_seed(0, "ab", "c")


def test_derive_seed_varies_with_every_argument():
    base = derive_seed(0, "x", 1)
    assert base != derive_seed(1, "x", 1)
    assert base != derive_seed(0, "y", 1)
    assert base != derive_seed(0, "x", 2)


def test_derive_seed_is_in_range_for_torch():
    """torch rejects seeds outside the signed 64-bit range."""
    for parts in (("a",), ("b", 2), ("c", None, 3.5)):
        value = derive_seed(12345, *parts)
        assert 0 <= value <= 2**63 - 1
        generator(value)  # must not raise


def test_derive_seed_is_stable_across_processes():
    """A fixed expected value, so a change of hash implementation is caught.

    :func:`hash` would pass every other test in this file and still differ
    between runs unless PYTHONHASHSEED is pinned, which is exactly the failure
    this asserts against.

    The constant is recorded from the implementation rather than derived -- it
    is an anchor, not a prediction -- so changing the hash function is supposed
    to fail here and make you decide whether the change was intended.
    """
    assert derive_seed(0, "circuitry") == 711156766224931010


def test_seed_everything_seeds_all_three_global_generators():
    seed_everything(123, deterministic_algorithms=False)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123, deterministic_algorithms=False)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_seed_everything_can_enable_deterministic_algorithms():
    try:
        seed_everything(5, deterministic_algorithms=True)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)
