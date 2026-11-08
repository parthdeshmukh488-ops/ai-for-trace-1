"""Explicit randomness.

Determinism in this repository is a tested property rather than an
aspiration, and the thing that makes it hard is not the seeding but the
*sharing*: a library function that draws from the global generator couples
every caller to every other caller's call order.  Add a diagnostic print that
happens to sample, and a dictionary trained afterwards is a different
dictionary.  That failure is invisible, survives a rerun with the same seed,
and is exactly the kind of thing a sparse-autoencoder comparison cannot
tolerate.

So: this module is the only place in the package allowed to touch global RNG
state, and it does so only in :func:`seed_everything`, which exists for
experiment scripts and tests rather than for library code.  Everything in
``circuitry`` that samples takes a ``torch.Generator`` explicitly.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["derive_seed", "generator", "seed_everything"]


def generator(seed: int, *, device: str | torch.device = "cpu") -> torch.Generator:
    """A generator seeded to ``seed``, independent of global state."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def derive_seed(seed: int, *parts: object) -> int:
    """A reproducible child seed from a parent seed and some labels.

    Used wherever a per-item generator is wanted -- one per tree node, one per
    dictionary training run in a sweep -- so that an item's randomness does not
    depend on how many items were processed before it.  That independence is
    what makes a sweep's individual runs reproducible in isolation, which is
    what you want at three in the morning when one configuration out of ninety
    behaved oddly.

    Hashing is done with a stable algorithm rather than :func:`hash`, whose
    string salt changes between processes unless ``PYTHONHASHSEED`` is set.
    """
    import hashlib

    payload = "|".join(repr(p) for p in (seed, *parts)).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def seed_everything(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed the global generators.  For scripts and tests, not for library code.

    With ``deterministic_algorithms``, torch is asked to refuse non-deterministic
    kernels rather than silently using them.  On CPU this costs nothing.  On
    CUDA some operations have no deterministic implementation and will raise;
    that is the intended behaviour -- a measurement that cannot be reproduced
    should fail loudly -- but it means the GPU tier passes
    ``deterministic_algorithms=False`` and reports ``allclose``-level
    reproducibility rather than bitwise.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on the test tier
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        # cuBLAS needs this set before the first allocation to make its GEMMs
        # reproducible; setting it here is late but harmless on CPU and is the
        # documented requirement on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
