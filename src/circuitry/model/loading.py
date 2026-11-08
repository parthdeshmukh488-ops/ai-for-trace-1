"""Importing external weights, and the three folds that make the stream readable.

Two separable jobs live here, and keeping them separable is the point.

**The import** is a pure renaming-and-reshaping of someone else's tensors. The
mapping was derived from the checkpoint's actual shapes rather than from
memory, and is written out as a table below so a future reader can check it
without running anything.

**The folds** are three exactly behaviour-preserving rewrites that remove
degrees of freedom the residual stream does not actually have. Each has a test
asserting the logits are unchanged, because "exactly behaviour-preserving" is
a claim and not a hope:

* *Fold LayerNorm.* ``LN(x) = (x - mean) * s * gamma + beta`` feeding a linear
  ``W, b`` is the same as ``(x - mean) * s`` feeding ``diag(gamma) W`` and
  ``b + beta W``.  Afterwards every weight that reads the stream reads it
  directly, and direct logit attribution is exact up to the single scalar
  ``s``.
* *Centre writing weights.*  LayerNorm subtracts the mean over the model
  dimension, so the mean of anything written into the stream is unobservable
  downstream.  Removing it changes nothing and deletes a meaningless component
  from every decomposition.  Worth being precise about why it is exact: if
  ``x' = x - c*1`` then ``mean(x') = mean(x) - c`` and the centred vector is
  unchanged, so the scale is unchanged too.
* *Centre the unembedding.*  Adding a constant to every logit leaves the
  softmax alone, so ``W_U`` may be centred over the vocabulary.  This one does
  change the raw logits, so its test asserts the log-softmax and the argmax
  rather than the logits.

The first two require LayerNorm to be present; applying them to a model
without it silently changes behaviour, so they refuse.

The checkpoint mapping, by shape:

| external                          | shape        | here                      |
|-----------------------------------|--------------|---------------------------|
| ``wte.weight``                    | ``[V, d]``   | ``W_E``                   |
| ``wpe.weight``                    | ``[n_ctx,d]``| ``W_pos``                 |
| ``h.L.ln_1.{weight,bias}``        | ``[d]``      | ``blocks.L.ln1.*``        |
| ``h.L.attn.c_attn.weight``        | ``[d, 3d]``  | ``W_Q|W_K|W_V``, split then ``[d,H,dh] -> [H,d,dh]`` |
| ``h.L.attn.c_attn.bias``          | ``[3d]``     | ``b_Q|b_K|b_V`` as ``[H,dh]`` |
| ``h.L.attn.c_proj.weight``        | ``[d, d]``   | ``W_O`` as ``[H,dh,d]``   |
| ``h.L.attn.c_proj.bias``          | ``[d]``      | ``b_O``                   |
| ``h.L.mlp.c_fc.{weight,bias}``    | ``[d,dm]``   | ``W_in``, ``b_in``        |
| ``h.L.mlp.c_proj.{weight,bias}``  | ``[dm,d]``   | ``W_out``, ``b_out``      |
| ``ln_f.{weight,bias}``            | ``[d]``      | ``ln_final.*``            |
| ``lm_head.weight``                | ``[V, d]``   | ``W_U`` transposed        |

The checkpoint stores the attention and MLP projections in ``[in, out]``
order, which happens to match the convention here, so none of them is
transposed.  That is a coincidence worth stating: the obvious defensive
transpose is wrong, and a transposed ``W_in`` on a square-ish layer produces a
model that runs and is simply a different function.
"""

from __future__ import annotations

import torch

from .config import ModelConfig
from .transformer import Transformer

__all__ = [
    "center_unembed",
    "center_writing_weights",
    "fold_layernorm",
    "from_reference_state_dict",
    "load_gpt2",
    "reference_config",
    "to_reference",
]


class FoldingNotApplicable(RuntimeError):
    """A fold that is only exact when LayerNorm is present, on a model without it."""


# ---------------------------------------------------------------------------
# the folds
# ---------------------------------------------------------------------------


@torch.no_grad()
def fold_layernorm(model: Transformer) -> Transformer:
    """Absorb every LayerNorm's gain and bias into the weights that follow it."""
    if not model.cfg.use_ln:
        raise FoldingNotApplicable("fold_layernorm needs LayerNorm; this model has none")

    def fold_into(gain, bias, weights_biases):
        for weight, bias_param, contract in weights_biases:
            # contract says which axis of `weight` is the model dimension.
            if contract == "hdk":  # [H, d, dh]
                if bias_param is not None:
                    bias_param += torch.einsum("d,hdk->hk", bias, weight)
                weight *= gain[None, :, None]
            elif contract == "dm":  # [d, d_mlp] or [d, V]
                if bias_param is not None:
                    bias_param += bias @ weight
                weight *= gain[:, None]
            else:  # pragma: no cover - guards a future layout
                raise AssertionError(contract)

    for block in model.blocks:
        fold_into(
            block.ln1.weight.data,
            block.ln1.bias.data,
            [
                (block.attn.W_Q.data, block.attn.b_Q.data, "hdk"),
                (block.attn.W_K.data, block.attn.b_K.data, "hdk"),
                (block.attn.W_V.data, block.attn.b_V.data, "hdk"),
            ],
        )
        block.ln1.weight.fill_(1.0)
        block.ln1.bias.zero_()
        fold_into(
            block.ln2.weight.data,
            block.ln2.bias.data,
            [(block.mlp.W_in.data, block.mlp.b_in.data, "dm")],
        )
        block.ln2.weight.fill_(1.0)
        block.ln2.bias.zero_()

    fold_into(
        model.ln_final.weight.data,
        model.ln_final.bias.data,
        [(model.W_U.data, model.b_U.data, "dm")],
    )
    model.ln_final.weight.fill_(1.0)
    model.ln_final.bias.zero_()
    return model


@torch.no_grad()
def center_writing_weights(model: Transformer) -> Transformer:
    """Remove the mean-over-``d_model`` component from everything written to the stream."""
    if not model.cfg.use_ln:
        raise FoldingNotApplicable(
            "center_writing_weights is only behaviour-preserving when a LayerNorm "
            "downstream removes the mean component; this model has none"
        )
    model.W_E.data -= model.W_E.data.mean(dim=-1, keepdim=True)
    model.W_pos.data -= model.W_pos.data.mean(dim=-1, keepdim=True)
    for block in model.blocks:
        block.attn.W_O.data -= block.attn.W_O.data.mean(dim=-1, keepdim=True)
        block.attn.b_O.data -= block.attn.b_O.data.mean()
        block.mlp.W_out.data -= block.mlp.W_out.data.mean(dim=-1, keepdim=True)
        block.mlp.b_out.data -= block.mlp.b_out.data.mean()
    return model


@torch.no_grad()
def center_unembed(model: Transformer) -> Transformer:
    """Centre ``W_U`` over the vocabulary, so logit attributions are comparable."""
    model.W_U.data -= model.W_U.data.mean(dim=-1, keepdim=True)
    model.b_U.data -= model.b_U.data.mean()
    return model


# ---------------------------------------------------------------------------
# import and export
# ---------------------------------------------------------------------------


def reference_config(hf_config) -> ModelConfig:
    """A :class:`ModelConfig` matching an external GPT-2 configuration."""
    act = {"gelu_new": "gelu_tanh", "gelu": "gelu_exact", "relu": "relu"}.get(
        hf_config.activation_function, hf_config.activation_function
    )
    return ModelConfig(
        n_layers=hf_config.n_layer,
        n_heads=hf_config.n_head,
        d_model=hf_config.n_embd,
        d_mlp=hf_config.n_inner or 4 * hf_config.n_embd,
        d_vocab=hf_config.vocab_size,
        n_ctx=hf_config.n_positions,
        act_fn=act,
        ln_eps=hf_config.layer_norm_epsilon,
        tie_unembed=True,
        dtype="float32",
    )


@torch.no_grad()
def from_reference_state_dict(cfg: ModelConfig, state: dict) -> Transformer:
    """Build a :class:`Transformer` from an external GPT-2 state dict."""
    model = Transformer(cfg)
    d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head

    def get(key: str) -> torch.Tensor:
        for prefix in ("", "transformer."):
            if prefix + key in state:
                return state[prefix + key]
        raise KeyError(f"checkpoint has no {key!r}; keys look like {list(state)[:4]}")

    model.W_E.copy_(get("wte.weight"))
    model.W_pos.copy_(get("wpe.weight")[: cfg.n_ctx])

    for layer, block in enumerate(model.blocks):
        block.ln1.weight.copy_(get(f"h.{layer}.ln_1.weight"))
        block.ln1.bias.copy_(get(f"h.{layer}.ln_1.bias"))
        block.ln2.weight.copy_(get(f"h.{layer}.ln_2.weight"))
        block.ln2.bias.copy_(get(f"h.{layer}.ln_2.bias"))

        qkv_w = get(f"h.{layer}.attn.c_attn.weight")  # [d, 3d]
        qkv_b = get(f"h.{layer}.attn.c_attn.bias")  # [3d]
        for i, (weight, bias) in enumerate(
            ((block.attn.W_Q, block.attn.b_Q), (block.attn.W_K, block.attn.b_K),
             (block.attn.W_V, block.attn.b_V))
        ):
            chunk_w = qkv_w[:, i * d : (i + 1) * d]  # [d, d] = [d, H*dh]
            weight.copy_(chunk_w.reshape(d, h, dh).permute(1, 0, 2))
            bias.copy_(qkv_b[i * d : (i + 1) * d].reshape(h, dh))

        block.attn.W_O.copy_(get(f"h.{layer}.attn.c_proj.weight").reshape(h, dh, d))
        block.attn.b_O.copy_(get(f"h.{layer}.attn.c_proj.bias"))
        block.mlp.W_in.copy_(get(f"h.{layer}.mlp.c_fc.weight"))
        block.mlp.b_in.copy_(get(f"h.{layer}.mlp.c_fc.bias"))
        block.mlp.W_out.copy_(get(f"h.{layer}.mlp.c_proj.weight"))
        block.mlp.b_out.copy_(get(f"h.{layer}.mlp.c_proj.bias"))

    model.ln_final.weight.copy_(get("ln_f.weight"))
    model.ln_final.bias.copy_(get("ln_f.bias"))
    head = state.get("lm_head.weight")
    model.W_U.copy_((head if head is not None else get("wte.weight")).T)
    model.b_U.zero_()
    return model


def load_gpt2(
    name: str = "gpt2",
    *,
    fold_ln: bool = True,
    center_writing: bool = True,
    center_unembedding: bool = True,
    dtype: str = "float32",
) -> Transformer:
    """Import an external GPT-2 checkpoint, folded for analysis by default.

    All three folds default on because all three are exact and each one
    removes a degree of freedom that would otherwise show up as structure in a
    decomposition that is not really there.  Pass them off to compare against
    the unmodified checkpoint, which is what the cross-check does.
    """
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(name)
    cfg = reference_config(hf.config).with_(dtype=dtype)
    model = from_reference_state_dict(cfg, hf.state_dict())
    model.to(getattr(torch, dtype))
    if fold_ln:
        fold_layernorm(model)
    if center_writing:
        center_writing_weights(model)
    if center_unembedding:
        center_unembed(model)
    return model.eval()


@torch.no_grad()
def to_reference(model: Transformer):
    """Export to an independently written implementation, for the cross-check.

    Exporting a *random* model and comparing is a stronger test than importing
    a real one, and a separable one: it checks the forward pass without the
    weight mapping in the way, needs no network, and can therefore run in CI.
    Importing a real checkpoint then checks the mapping, which is a different
    claim.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = model.cfg
    if cfg.act_fn != "gelu_tanh":
        raise ValueError(f"the reference implementation uses gelu_new; this model uses {cfg.act_fn}")
    if not cfg.use_ln or cfg.positional != "learned":
        raise ValueError("the reference implementation assumes LayerNorm and learned positions")
    if cfg.n_heads * cfg.d_head != cfg.d_model:
        raise ValueError(
            f"the reference implementation requires n_heads*d_head == d_model; got "
            f"{cfg.n_heads}*{cfg.d_head} != {cfg.d_model}"
        )
    hf = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=cfg.d_vocab,
            n_positions=cfg.n_ctx,
            n_embd=cfg.d_model,
            n_layer=cfg.n_layers,
            n_head=cfg.n_heads,
            n_inner=cfg.d_mlp,
            activation_function="gelu_new",
            layer_norm_epsilon=cfg.ln_eps,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            # The fused attention kernels do not return their patterns, and
            # comparing patterns is half the point of the cross-check.
            attn_implementation="eager",
        )
    )
    d, h, dh = cfg.d_model, cfg.n_heads, cfg.d_head
    state = {}
    state["transformer.wte.weight"] = model.W_E.data.clone()
    state["transformer.wpe.weight"] = model.W_pos.data.clone()
    for layer, block in enumerate(model.blocks):
        pre = f"transformer.h.{layer}"
        state[f"{pre}.ln_1.weight"] = block.ln1.weight.data.clone()
        state[f"{pre}.ln_1.bias"] = block.ln1.bias.data.clone()
        state[f"{pre}.ln_2.weight"] = block.ln2.weight.data.clone()
        state[f"{pre}.ln_2.bias"] = block.ln2.bias.data.clone()
        qkv_w = torch.cat(
            [
                w.data.permute(1, 0, 2).reshape(d, d)
                for w in (block.attn.W_Q, block.attn.W_K, block.attn.W_V)
            ],
            dim=1,
        )
        qkv_b = torch.cat(
            [b.data.reshape(d) for b in (block.attn.b_Q, block.attn.b_K, block.attn.b_V)]
        )
        state[f"{pre}.attn.c_attn.weight"] = qkv_w
        state[f"{pre}.attn.c_attn.bias"] = qkv_b
        state[f"{pre}.attn.c_proj.weight"] = block.attn.W_O.data.reshape(h * dh, d).clone()
        state[f"{pre}.attn.c_proj.bias"] = block.attn.b_O.data.clone()
        state[f"{pre}.mlp.c_fc.weight"] = block.mlp.W_in.data.clone()
        state[f"{pre}.mlp.c_fc.bias"] = block.mlp.b_in.data.clone()
        state[f"{pre}.mlp.c_proj.weight"] = block.mlp.W_out.data.clone()
        state[f"{pre}.mlp.c_proj.bias"] = block.mlp.b_out.data.clone()
    state["transformer.ln_f.weight"] = model.ln_final.weight.data.clone()
    state["transformer.ln_f.bias"] = model.ln_final.bias.data.clone()
    state["lm_head.weight"] = model.W_U.data.T.clone()
    missing, unexpected = hf.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if "masked_bias" not in k and ".attn.bias" not in k]
    missing = [k for k in missing if "masked_bias" not in k and ".attn.bias" not in k]
    if missing or unexpected:
        raise AssertionError(f"export mismatch: missing={missing} unexpected={unexpected}")
    if model.b_U.data.abs().max() > 0:
        raise ValueError("the reference implementation has no unembedding bias; b_U must be zero")
    return hf.to(model.dtype).eval()
