"""Mechanistic interpretability, measured against circuits that are known.

The public surface is deliberately flat: real work should be possible from
``circuitry.<thing>`` without reaching into submodules, because the submodule
layout is an implementation detail.

Names the build plan has not reached yet are present but raise, naming the
phase that implements them.  An early call then says what is missing rather
than failing with an import error three frames deep.  ``_PENDING`` is kept
honest by ``tests/test_public_surface.py``: a name listed as pending that is
in fact implemented fails that test, and so does a pending name whose stub
does not name its phase.
"""

from __future__ import annotations

from .model import ACT_FNS, ModelConfig, sites
from .model.hooks import capture, intervene
from .model.loading import load_gpt2
from .model.transformer import Transformer
from .symbolic import compile_program

# Name -> the phase of docs/BUILD_PLAN.md that implements it.  Entries are
# deleted from here as each phase lands, and the real object is imported above.
_PENDING: dict[str, int] = {
    "patch": 3,
    "path_patch": 3,
    "Circuit": 3,
    "attribute": 4,
    "discover": 5,
    "train_sae": 6,
    "steer": 9,
}

# Written out rather than built from _PENDING: a literal is what a reader and a
# static checker can both see, and tests/test_public_surface.py is what keeps
# it in step with the specification and with _PENDING.
__all__ = [
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
]


def _unbuilt(name: str, phase: int):
    def stub(*_args, **_kwargs):
        raise NotImplementedError(
            f"circuitry.{name} lands in phase {phase}; see docs/BUILD_PLAN.md"
        )

    stub.__name__ = name
    stub.__qualname__ = name
    stub.__doc__ = f"Not implemented yet: lands in phase {phase}. See docs/BUILD_PLAN.md."
    stub._pending_phase = phase
    return stub


for _name, _phase in _PENDING.items():
    globals().setdefault(_name, _unbuilt(_name, _phase))

del _name, _phase
