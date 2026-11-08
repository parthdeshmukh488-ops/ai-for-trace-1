"""The public surface says what is built and what is not, and stays honest.

The point of this test is not the stubs themselves; it is that ``_PENDING``
cannot drift.  Half-way through a long build the most likely documentation
error is a name that was implemented three phases ago and is still advertised
as unbuilt, or a name promised in the specification that nobody noticed is
missing.  Both fail here.
"""

from __future__ import annotations

import pytest

import circuitry

# The surface promised by docs/SPEC.md section 0.
PROMISED = {
    "ACT_FNS",
    "Circuit",
    "ModelConfig",
    "Transformer",
    "attribute",
    "capture",
    "compile_program",
    "discover",
    "intervene",
    "load_gpt2",
    "patch",
    "path_patch",
    "sites",
    "steer",
    "train_sae",
}


def test_every_promised_name_is_present():
    missing = PROMISED - set(circuitry.__all__)
    assert not missing, f"names promised by the specification but absent: {sorted(missing)}"


def test_all_is_exactly_the_promised_surface():
    extra = set(circuitry.__all__) - PROMISED
    assert not extra, (
        f"names exported but not in the specification: {sorted(extra)}; "
        "either add them to docs/SPEC.md section 0 or stop exporting them"
    )


def test_every_name_in_all_is_reachable():
    for name in circuitry.__all__:
        assert hasattr(circuitry, name), f"{name} is in __all__ but not an attribute"


@pytest.mark.parametrize("name", sorted(circuitry._PENDING))
def test_unimplemented_stubs_name_their_phase(name):
    """An early call must say which phase it is waiting for."""
    stub = getattr(circuitry, name)
    assert getattr(stub, "_pending_phase", None) == circuitry._PENDING[name]
    with pytest.raises(NotImplementedError) as excinfo:
        stub()
    message = str(excinfo.value)
    assert name in message
    assert f"phase {circuitry._PENDING[name]}" in message
    assert "BUILD_PLAN" in message


def test_pending_names_are_not_secretly_implemented():
    """A name that works must not still be advertised as pending."""
    for name in sorted(circuitry._PENDING):
        obj = getattr(circuitry, name)
        assert getattr(obj, "_pending_phase", None) is not None, (
            f"circuitry.{name} appears to be implemented but is still listed in "
            "_PENDING; remove it from that dict"
        )
