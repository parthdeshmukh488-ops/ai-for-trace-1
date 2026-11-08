"""A transformer written so that its insides are addressable.

Wrapping a standard implementation gets you the activations and fights you on
the two things this project actually needs: *replacing* an intermediate
mid-forward without recomputing what came before it, and decomposing the
residual stream into per-component contributions.  Writing the forward pass
makes the decomposition the natural thing and the hook points structural
rather than bolted on.

The cost is that correctness becomes this file's problem, which is what
``benchmarks/cross_check_reference.py`` is for: an independently written
implementation is the strongest available oracle, and if the two agree on
every activation site to machine precision, both are very unlikely to be
wrong in the same way.

Three decisions worth stating, because each one is a place where a plausible
alternative would quietly spoil a measurement.

* **The causal mask is re-applied after the ``scores`` hook**, when there is
  one -- see ``ModelConfig.causal``.  Masked entries
  are not a place where information may be injected: patching one must be a
  no-op, not a behaviour change, or an experiment sweeping "every entry of the
  attention pattern" would spend half its measurements letting the model see
  the future.  The engine therefore masks, offers the tensor to hooks, and
  masks again.
* **Per-head output is opt-in.**  ``attn.result`` is ``[B,T,H,d]`` -- ``H``
  times the size of ``attn.out``, 151 MB per layer for GPT-2 small at batch 8
  and 512 positions.  It is materialised only when a hook is listening, which
  the registry reports.  This is a documented performance cliff rather than
  something to discover later.
* **The LayerNorm scale is its own site.**  LayerNorm is not linear, and the
  non-linearity is entirely in that one scalar per position.  Exposing it means
  it can be frozen, which is what makes direct logit attribution exact rather
  than approximate, and it means the error the approximation introduces can be
  *measured* instead of assumed small.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig
from .hooks import HookRegistry

__all__ = ["MLP", "Attention", "Block", "LayerNorm", "Transformer", "act_fn"]


def act_fn(name: str, x: torch.Tensor) -> torch.Tensor:
    """The activation functions, with the GPT-2 approximation named as such.

    ``gelu_tanh`` is what the GPT-2 checkpoints were trained with.  It differs
    from the exact error-function form by up to about 1e-3 in activation space,
    which is above the cross-check tolerance in the forward-pass comparison --
    so importing weights under the wrong one fails loudly rather than
    degrading quietly.
    """
    if name == "gelu_tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
    if name == "gelu_exact":
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    if name == "relu":
        return torch.relu(x)
    if name == "identity":
        return x
    if name == "solu":
        # Softmax Linear Unit: divides by its own softmax rather than a fixed
        # curve, so it is scale-sensitive in a way the others are not.
        return x * torch.softmax(x, dim=-1)
    raise ValueError(f"unknown activation function {name!r}")


class LayerNorm(nn.Module):
    """Centre, scale, then an affine map -- with the scale exposed as a site.

    When ``cfg.use_ln`` is false this is the identity, but the sites still
    exist and still report a scale of one.  Keeping them means every piece of
    analysis code works unchanged on a model without LayerNorm, which is the
    configuration the compiled models and the linearity validation test use.
    """

    def __init__(self, cfg: ModelConfig, prefix: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix
        self.weight = nn.Parameter(torch.ones(cfg.d_model))
        self.bias = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: torch.Tensor, hooks: HookRegistry) -> torch.Tensor:
        if not self.cfg.use_ln:
            one = torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
            hooks(f"{self.prefix}.scale", one)
            return hooks(f"{self.prefix}.normalized", x)
        centred = x - x.mean(dim=-1, keepdim=True)
        # The biased variance, which is what every transformer LayerNorm uses.
        var = centred.pow(2).mean(dim=-1, keepdim=True)
        scale = hooks(f"{self.prefix}.scale", (var + self.cfg.ln_eps).rsqrt())
        out = centred * scale * self.weight + self.bias
        return hooks(f"{self.prefix}.normalized", out)


class Attention(nn.Module):
    """Multi-head attention with per-head weights kept separate.

    ``W_Q`` is ``[H, d, dh]`` rather than a fused ``[d, H*dh]`` so that a head
    is an index rather than a slice.  Fusing is faster and makes every
    per-head operation an exercise in reshape arithmetic; given that nearly
    everything in this package is a per-head operation, the trade goes the
    other way here.
    """

    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        h, d, dh = cfg.n_heads, cfg.d_model, cfg.d_head
        self.W_Q = nn.Parameter(torch.zeros(h, d, dh))
        self.W_K = nn.Parameter(torch.zeros(h, d, dh))
        self.W_V = nn.Parameter(torch.zeros(h, d, dh))
        self.W_O = nn.Parameter(torch.zeros(h, dh, d))
        self.b_Q = nn.Parameter(torch.zeros(h, dh))
        self.b_K = nn.Parameter(torch.zeros(h, dh))
        self.b_V = nn.Parameter(torch.zeros(h, dh))
        self.b_O = nn.Parameter(torch.zeros(d))

    def forward(self, normed: torch.Tensor, hooks: HookRegistry, mask: torch.Tensor):
        p = f"blocks.{self.layer}.attn"
        q = hooks(f"{p}.q", torch.einsum("btd,hdk->bthk", normed, self.W_Q) + self.b_Q)
        k = hooks(f"{p}.k", torch.einsum("btd,hdk->bthk", normed, self.W_K) + self.b_K)
        v = hooks(f"{p}.v", torch.einsum("btd,hdk->bthk", normed, self.W_V) + self.b_V)

        raw = torch.einsum("bqhk,bshk->bhqs", q, k) / math.sqrt(self.cfg.d_head)
        seq = normed.shape[1]
        if self.cfg.causal:
            causal = mask[:seq, :seq]
            neg_inf = torch.finfo(raw.dtype).min
            raw = raw.masked_fill(causal, neg_inf)
            scores = hooks(f"{p}.scores", raw)
            # Masked again: a masked entry is not a channel, so patching one is
            # a no-op by contract.  See the module docstring.
            scores = scores.masked_fill(causal, neg_inf)
        else:
            scores = hooks(f"{p}.scores", raw)
        pattern = hooks(f"{p}.pattern", torch.softmax(scores, dim=-1))

        z = hooks(f"{p}.z", torch.einsum("bhqs,bshk->bqhk", pattern, v))
        if hooks.wanted(f"{p}.result"):
            result = hooks(f"{p}.result", torch.einsum("bqhk,hkd->bqhd", z, self.W_O))
            out = result.sum(dim=2) + self.b_O
        else:
            out = torch.einsum("bqhk,hkd->bqd", z, self.W_O) + self.b_O
        return hooks(f"{p}.out", out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        self.W_in = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.zeros(cfg.d_mlp, cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, normed: torch.Tensor, hooks: HookRegistry) -> torch.Tensor:
        p = f"blocks.{self.layer}.mlp"
        pre = hooks(f"{p}.pre", normed @ self.W_in + self.b_in)
        post = hooks(f"{p}.post", act_fn(self.cfg.act_fn, pre))
        return hooks(f"{p}.out", post @ self.W_out + self.b_out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        self.ln1 = LayerNorm(cfg, f"blocks.{layer}.ln1")
        self.attn = Attention(cfg, layer)
        self.ln2 = LayerNorm(cfg, f"blocks.{layer}.ln2")
        self.mlp = MLP(cfg, layer)

    def forward(self, resid: torch.Tensor, hooks: HookRegistry, mask) -> torch.Tensor:
        resid = hooks(f"blocks.{self.layer}.resid_pre", resid)
        attn_out = self.attn(self.ln1(resid, hooks), hooks, mask)
        mid = hooks(f"blocks.{self.layer}.resid_mid", resid + attn_out)
        mlp_out = self.mlp(self.ln2(mid, hooks), hooks)
        return hooks(f"blocks.{self.layer}.resid_post", mid + mlp_out)


class Transformer(nn.Module):
    """A GPT-2-shaped transformer with every intermediate addressable.

    ``model.hooks`` is the registry; use :func:`circuitry.intervene` and
    :func:`circuitry.capture` rather than touching it, so that the
    reversibility contract holds.
    """

    def __init__(self, cfg: ModelConfig, *, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.hooks = HookRegistry()
        self.W_E = nn.Parameter(torch.zeros(cfg.d_vocab, cfg.d_model))
        self.W_pos = nn.Parameter(torch.zeros(cfg.n_ctx, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg, "ln_final")
        self.W_U = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_vocab))
        self.b_U = nn.Parameter(torch.zeros(cfg.d_vocab))
        # Registered rather than recomputed: the comparison itself is cheap but
        # allocating a [T,T] bool every forward pass is not, and a buffer moves
        # with .to() while a cached tensor would not.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.to(getattr(torch, cfg.dtype))
        if generator is not None:
            self.randomise(generator)

    # -- construction -----------------------------------------------------

    def randomise(self, generator: torch.Generator, *, embed_std: float = 0.02) -> None:
        """Fill with random weights scaled so activations neither vanish nor blow up.

        ``1/sqrt(fan_in)`` for the projections rather than GPT-2's flat 0.02:
        at the widths the test models use, 0.02 produces logits indistinguishable
        from zero, and a test whose metric has no signal passes for the wrong
        reason.  Embeddings keep 0.02 because they have no fan-in to scale by.
        """
        cfg = self.cfg

        def fill(param: nn.Parameter, std: float) -> None:
            with torch.no_grad():
                param.copy_(
                    torch.randn(param.shape, generator=generator, dtype=param.dtype) * std
                )

        fill(self.W_E, embed_std)
        fill(self.W_pos, embed_std)
        for block in self.blocks:
            for w in (block.attn.W_Q, block.attn.W_K, block.attn.W_V):
                fill(w, cfg.d_model**-0.5)
            fill(block.attn.W_O, cfg.d_head**-0.5)
            fill(block.mlp.W_in, cfg.d_model**-0.5)
            fill(block.mlp.W_out, cfg.d_mlp**-0.5)
        fill(self.W_U, cfg.d_model**-0.5)
        if cfg.tie_unembed:
            with torch.no_grad():
                self.W_U.copy_(self.W_E.T)

    # -- properties -------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        return self.W_E.dtype

    @property
    def device(self) -> torch.device:
        return self.W_E.device

    # -- forward ----------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [batch, pos] or [pos], got {tuple(tokens.shape)}")
        batch, seq = tokens.shape
        if seq > self.cfg.n_ctx:
            raise ValueError(f"sequence length {seq} exceeds n_ctx={self.cfg.n_ctx}")
        if tokens.max() >= self.cfg.d_vocab or tokens.min() < 0:
            raise ValueError(
                f"token ids must lie in [0, {self.cfg.d_vocab}), got "
                f"[{int(tokens.min())}, {int(tokens.max())}]"
            )
        hooks = self.hooks
        embed = hooks("embed", self.W_E[tokens])
        if self.cfg.positional == "none":
            pos = torch.zeros_like(embed)
        elif self.cfg.positional == "learned":
            pos = self.W_pos[:seq].expand(batch, seq, self.cfg.d_model)
        else:
            pos = _sinusoidal(seq, self.cfg.d_model, embed.dtype, embed.device).expand(
                batch, seq, self.cfg.d_model
            )
        pos = hooks("pos_embed", pos)
        resid = embed + pos
        for block in self.blocks:
            resid = block(resid, hooks, self.causal_mask)
        final = self.ln_final(resid, hooks)
        return hooks("logits", final @ self.W_U + self.b_U)

    def residual_decomposition(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every term written into the residual stream, plus their sum.

        The identity ``resid_post[last] == embed + pos_embed + sum(attn.out) +
        sum(mlp.out)`` is exact by construction and touches nearly every code
        path, which is why the test built on it catches more engine bugs than
        the rest of the suite put together.
        """
        from .hooks import capture

        want = ["embed", "pos_embed", f"blocks.{self.cfg.n_layers - 1}.resid_post"]
        want += [f"blocks.{i}.attn.out" for i in range(self.cfg.n_layers)]
        want += [f"blocks.{i}.mlp.out" for i in range(self.cfg.n_layers)]
        with capture(self, want) as store:
            self(tokens)
        terms = {name: store[name] for name in want}
        total = terms["embed"] + terms["pos_embed"]
        for i in range(self.cfg.n_layers):
            total = total + terms[f"blocks.{i}.attn.out"] + terms[f"blocks.{i}.mlp.out"]
        terms["sum"] = total
        return terms


def _sinusoidal(seq: int, d: int, dtype, device) -> torch.Tensor:
    pos = torch.arange(seq, dtype=torch.float64, device=device)[:, None]
    i = torch.arange(d, dtype=torch.float64, device=device)[None, :]
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    out = torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))
    return out.to(dtype)[None]
