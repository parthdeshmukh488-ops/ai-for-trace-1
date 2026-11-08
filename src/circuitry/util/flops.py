"""What a forward pass costs, derived rather than quoted.

Several measurements in this repository compare methods on cost -- exact
patching against its gradient approximations, greedy discovery against edge
ranking -- and a comparison on cost needs a cost model that is right to within
a factor that is itself known.  Published "6N per token" rules of thumb are
not enough: they fold the attention term into a constant, drop the unembedding
entirely, and are stated for training rather than for a single forward pass.
So the arithmetic is written out here.

The convention throughout is that a multiply-accumulate is **two** FLOPs, and
that a matrix-vector product of shape ``(d_out, d_in)`` against one token
therefore costs ``2 * d_in * d_out``.  Elementwise work -- LayerNorm, the
activation function, softmax -- is ``O(d)`` or ``O(T)`` per token against the
``O(d^2)`` of the projections, and is excluded; :func:`forward_flops` names
that exclusion in its return value so a caller cannot mistake the number for
a complete one.

Per layer, for one token at context length ``T``:

* **QKVO projections.** Four matrices of shape ``(d, d)`` when
  ``n_heads * d_head == d_model``, so ``4 * 2 * d^2``.  When ``d_head`` is set
  so that ``n_heads * d_head != d_model`` the Q/K/V matrices are
  ``(d, n_heads * d_head)`` and O is ``(n_heads * d_head, d)``, which is the
  general form used below.
* **Attention.** The query dots every key in the context
  (``T * n_heads * d_head`` MACs) and the pattern then weights every value
  (the same again), giving ``2 * 2 * T * n_heads * d_head``.  This is the term
  that makes cost quadratic in sequence length, and the one a per-token
  constant hides.
* **MLP.** Two matrices between ``d`` and ``d_mlp``, so ``2 * 2 * d * d_mlp``.

Once per token, outside the layer stack:

* **Embedding.** A row gather. Zero FLOPs -- it is memory traffic, not
  arithmetic, and counting it as a matmul against a one-hot (which is how it
  is sometimes written) overstates the cost by ``2 * d * V`` per token.
* **Unembedding.** A real matrix-vector product against ``(d, V)``, so
  ``2 * d * V``.  For GPT-2 small at ``d = 768`` and ``V = 50257`` this is
  77 MFLOPs per token against 85 MFLOPs for one layer's projections and MLP --
  a third of the twelve-layer total.  Dropping it, which is common, is not a
  rounding error.

Backward is taken as twice forward: one pass for the gradient with respect to
the inputs and one for the gradient with respect to the weights, each costing
about what the forward matmul did.  This is the standard approximation and it
is an approximation; :func:`backward_flops` says so rather than implying the
factor is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FlopCount",
    "attention_flops",
    "backward_flops",
    "decode_flops",
    "forward_flops",
    "mlp_flops",
    "projection_flops",
    "unembed_flops",
]


@dataclass(frozen=True)
class FlopCount:
    """A forward-pass cost, broken down so the dominant term is visible.

    ``excludes`` names what was deliberately left out, so that a number from
    this module can never be mistaken for a complete accounting.
    """

    projections: int
    attention: int
    mlp: int
    unembed: int
    excludes: str = "elementwise work: layernorm, activation, softmax, residual adds"

    @property
    def total(self) -> int:
        return self.projections + self.attention + self.mlp + self.unembed

    def __int__(self) -> int:
        return self.total


def _kv_width(cfg) -> int:
    """The width the heads actually occupy, which need not equal ``d_model``."""
    return cfg.n_heads * cfg.d_head


def projection_flops(cfg, n_tokens: int) -> int:
    """Q, K, V and O for ``n_tokens`` tokens across every layer."""
    dkv = _kv_width(cfg)
    # Q, K, V are (d_model -> dkv); O is (dkv -> d_model).  Four products, each
    # 2 * d_in * d_out per token.
    per_token = 4 * 2 * cfg.d_model * dkv
    return cfg.n_layers * n_tokens * per_token


def attention_flops(cfg, context_lengths) -> int:
    """Scores and the value-weighted sum, given each token's context length.

    ``context_lengths`` is the number of keys each query attends to, which is
    the token's absolute position plus one under a causal mask.  Passing the
    lengths rather than a sequence length is what lets prefill and incremental
    decode share this function instead of each deriving the sum themselves.
    """
    dkv = _kv_width(cfg)
    total_keys = int(sum(context_lengths))
    # Two passes over the keys -- q.k, then pattern.v -- each 2 FLOPs per
    # element of the head dimension.
    return cfg.n_layers * 2 * 2 * total_keys * dkv


def mlp_flops(cfg, n_tokens: int) -> int:
    """Both MLP matrices for ``n_tokens`` tokens across every layer."""
    per_token = 2 * 2 * cfg.d_model * cfg.d_mlp
    return cfg.n_layers * n_tokens * per_token


def unembed_flops(cfg, n_tokens: int) -> int:
    """The projection to vocabulary logits.  Not negligible; see the module docstring."""
    return n_tokens * 2 * cfg.d_model * cfg.d_vocab


def forward_flops(cfg, seq_len: int, *, batch: int = 1, unembed_all: bool = True) -> FlopCount:
    """One forward pass over ``seq_len`` tokens, with a causal mask.

    ``unembed_all=False`` counts the unembedding for the final position only,
    which is what generation does and what makes the unembedding term small;
    the default is the training and analysis case, where every position's
    logits are produced.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    n = seq_len * batch
    # Under a causal mask the token at position i attends to i + 1 keys, so the
    # total over a prefill is the triangular number, not seq_len^2.
    keys = [(i + 1) for i in range(seq_len)] * batch
    unembed_tokens = n if unembed_all else batch
    return FlopCount(
        projections=projection_flops(cfg, n),
        attention=attention_flops(cfg, keys),
        mlp=mlp_flops(cfg, n),
        unembed=unembed_flops(cfg, unembed_tokens),
    )


def decode_flops(cfg, prefix_len: int, n_new: int, *, batch: int = 1) -> FlopCount:
    """Generating ``n_new`` tokens after a cached prefix of ``prefix_len``.

    The prefill is *not* included: this is the incremental cost only, so that a
    caller composing a prefill and a decode adds two numbers rather than
    subtracting one from another.
    """
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
    if n_new < 1:
        raise ValueError(f"n_new must be at least 1, got {n_new}")
    n = n_new * batch
    keys = [prefix_len + i + 1 for i in range(n_new)] * batch
    return FlopCount(
        projections=projection_flops(cfg, n),
        attention=attention_flops(cfg, keys),
        mlp=mlp_flops(cfg, n),
        unembed=unembed_flops(cfg, n),
    )


def backward_flops(cfg, seq_len: int, *, batch: int = 1, unembed_all: bool = True) -> FlopCount:
    """Twice the forward cost, which is an approximation and not an identity.

    The two passes -- gradient with respect to inputs, gradient with respect to
    weights -- each cost about one forward matmul, so the factor is 2 for the
    matmul-dominated terms.  It is not exact: recomputation under activation
    checkpointing raises it, and a frozen matrix needs only the input gradient
    and so halves its own contribution.  Callers comparing a gradient method
    against a forward-only one should quote the factor they assumed.
    """
    fwd = forward_flops(cfg, seq_len, batch=batch, unembed_all=unembed_all)
    return FlopCount(
        projections=2 * fwd.projections,
        attention=2 * fwd.attention,
        mlp=2 * fwd.mlp,
        unembed=2 * fwd.unembed,
        excludes=fwd.excludes + "; the factor of 2 is the standard approximation",
    )
