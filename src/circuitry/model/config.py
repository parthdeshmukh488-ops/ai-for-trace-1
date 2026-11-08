"""The model's shape, in the one form the rest of the package agrees on.

A config object rather than a pile of keyword arguments, and frozen rather
than mutable, because nearly everything downstream caches something derived
from it: the site table, the basis allocation of a compiled model, the FLOP
model, the shape assertions in every test.  A config that can be edited after
a cache has been built from it is a way of producing results whose stated
shape is not the shape they were computed at.

``d_head`` is allowed to be independent of ``d_model / n_heads``.  Real
checkpoints always have ``n_heads * d_head == d_model``, but compiled models
(see :mod:`circuitry.symbolic`) do not: their width is the sum of their
variables' one-hot blocks, which is whatever the program needs, and forcing it
to be divisible by the head count would mean padding the residual stream with
dimensions that mean nothing -- in a regime whose whole value is that every
dimension means something.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["ACT_FNS", "ModelConfig"]

# The activation functions the engine implements.  ``gelu_tanh`` is the
# approximation the GPT-2 checkpoints were trained with and differs from the
# exact error-function form by up to ~1e-3 in activation space -- above the
# forward-pass cross-check tolerance, so choosing the wrong one here shows up
# as a cross-check failure rather than as a silent inaccuracy.  ``relu`` and
# ``identity`` exist for compiled and analytic models: ``identity`` makes the
# MLP linear, which is what the first-order attribution validation test needs.
ACT_FNS = ("gelu_tanh", "gelu_exact", "relu", "identity", "solu")

_POSITIONAL = ("learned", "none", "sinusoidal")
_DTYPES = ("float32", "float64", "bfloat16", "float16")


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    d_head: int | None = None
    act_fn: str = "gelu_tanh"
    ln_eps: float = 1e-5
    use_ln: bool = True
    positional: str = "learned"
    tie_unembed: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("n_layers", "n_heads", "d_model", "d_mlp", "d_vocab", "n_ctx"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.d_head is None:
            if self.d_model % self.n_heads:
                raise ValueError(
                    f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}; "
                    "pass d_head explicitly"
                )
            object.__setattr__(self, "d_head", self.d_model // self.n_heads)
        if not isinstance(self.d_head, int) or self.d_head < 1:
            raise ValueError(f"d_head must be a positive int, got {self.d_head!r}")
        if self.act_fn not in ACT_FNS:
            raise ValueError(f"act_fn must be one of {ACT_FNS}, got {self.act_fn!r}")
        if self.positional not in _POSITIONAL:
            raise ValueError(f"positional must be one of {_POSITIONAL}, got {self.positional!r}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
        if self.ln_eps <= 0:
            raise ValueError(f"ln_eps must be positive, got {self.ln_eps}")

    @property
    def d_attn(self) -> int:
        """The width the heads jointly occupy; equals ``d_model`` for real checkpoints."""
        return self.n_heads * self.d_head

    def with_(self, **changes) -> ModelConfig:
        """A copy with fields replaced, revalidated.

        Used by the weight-folding tests, which need the same config with
        ``use_ln`` flipped, and by the attribution validation test, which needs
        it with ``act_fn="identity"``.
        """
        if "d_head" not in changes and "n_heads" in changes:
            # Recompute rather than carrying over a d_head derived from the old
            # head count, which would silently change d_attn.
            changes = {**changes, "d_head": None}
        return replace(self, **changes)
