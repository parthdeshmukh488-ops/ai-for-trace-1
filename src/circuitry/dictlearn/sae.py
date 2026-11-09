"""Sparse autoencoders, and the details that decide whether they work.

All four variants share one shape:

    f    = enc(W_e (x - b_d) + b_e)
    xhat = W_d f + b_d

The pre-encoder subtraction of the decoder bias is not cosmetic.  It makes the
model's claim literally ``x ~ b_d + sum_i f_i W_d[:, i]`` -- a baseline plus a
sparse combination of directions -- and without it the bias is absorbed into
the features, so a "feature" can be a constant offset.

Where the four differ is how sparsity is obtained, and each has one detail
that silently breaks it if missed:

* **ReLU with an L1 penalty.**  The penalty must be weighted by the decoder
  column's norm, ``lambda * sum_i f_i ||W_d[:, i]||``.  Penalising ``f`` alone
  is trivially gamed: scale a feature's activation down and its decoder column
  up, and the reconstruction is unchanged while the penalty falls.  The norm
  factor removes that degeneracy exactly, and the columns are also
  renormalised after every step as a second line of defence.
* **TopK.**  Keep the largest ``k`` pre-activations and zero the rest.  ``L0``
  is then exactly ``k`` and there is no coefficient to tune, which is the
  point.  The norm constraint is *not* applied here -- there is no penalty to
  game, and constraining it would only remove capacity.
* **BatchTopK.**  Keep the largest ``k * batch`` activations across the whole
  batch, so ``L0`` varies per example and the easy examples give their budget
  to the hard ones.  The catch: at inference there is no batch, so a single
  global threshold has to stand in for it, estimated as a running average of
  the ``k*batch``-th value during training.  **Train and inference behaviour
  therefore differ**, and the threshold is reported with every run because a
  dictionary evaluated in training mode and used in inference mode is two
  different dictionaries.
* **JumpReLU.**  ``f = z * H(z - theta)`` with a learned threshold per feature.
  Both the Heaviside and the ``L0`` count have zero derivative almost
  everywhere, so a straight-through estimator substitutes a rectangle kernel
  of bandwidth ``eps``.  That bandwidth trades gradient bias against variance
  and is a real hyperparameter, not a detail: too small and the thresholds
  never move, too large and they move for the wrong reasons.

Dead features are handled by two remedies, implemented so they can be
compared rather than chosen by citation: resampling (reinitialise a dead
feature onto an under-reconstructed input) and an auxiliary loss (reconstruct
the residual using the top dead features).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "BatchTopKSae",
    "JumpReluSae",
    "ReluSae",
    "SaeConfig",
    "SaeOutput",
    "TopKSae",
    "make_sae",
]


@dataclass(frozen=True)
class SaeConfig:
    d_in: int
    expansion: int = 4
    k: int | None = None  # TopK and BatchTopK
    lambda_sparsity: float = 1e-3  # ReLU and JumpReLU
    jump_bandwidth: float = 1e-3
    dtype: str = "float32"

    @property
    def n_features(self) -> int:
        return self.d_in * self.expansion


@dataclass
class SaeOutput:
    features: torch.Tensor
    reconstruction: torch.Tensor
    mse: torch.Tensor
    sparsity_term: torch.Tensor
    l0: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return self.mse + self.sparsity_term


class _BaseSae(nn.Module):
    """Shared parameters and the geometry claim they encode."""

    needs_norm_constraint = False

    def __init__(self, cfg: SaeConfig, *, generator=None) -> None:
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        f, d = cfg.n_features, cfg.d_in
        # Decoder columns initialised to unit norm, encoder to its transpose:
        # a sensible starting point that makes the first step's features
        # meaningful rather than noise.
        decoder = torch.randn(d, f, generator=generator, dtype=dtype)
        decoder = decoder / decoder.norm(dim=0, keepdim=True)
        self.W_d = nn.Parameter(decoder)
        self.W_e = nn.Parameter(decoder.T.clone())
        self.b_e = nn.Parameter(torch.zeros(f, dtype=dtype))
        self.b_d = nn.Parameter(torch.zeros(d, dtype=dtype))
        self.register_buffer(
            "firing_count", torch.zeros(f, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "steps_seen", torch.zeros((), dtype=torch.long), persistent=False
        )

    # -- geometry --------------------------------------------------------

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_d) @ self.W_e.T + self.b_e

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_d.T + self.b_d

    @property
    def directions(self) -> torch.Tensor:
        """``[n_features, d_in]``: the learned feature directions, one per row."""
        return self.W_d.detach().T

    def decoder_norms(self) -> torch.Tensor:
        return self.W_d.norm(dim=0)

    @torch.no_grad()
    def renormalise(self) -> None:
        if self.needs_norm_constraint:
            self.W_d.div_(self.W_d.norm(dim=0, keepdim=True).clamp(min=1e-8))

    # -- bookkeeping -----------------------------------------------------

    @torch.no_grad()
    def observe(self, features: torch.Tensor) -> None:
        self.firing_count += (features > 0).sum(dim=0)
        self.steps_seen += features.shape[0]

    def dead_fraction(self) -> float:
        return float((self.firing_count == 0).sum()) / self.cfg.n_features

    @torch.no_grad()
    def reset_counts(self) -> None:
        self.firing_count.zero_()
        self.steps_seen.zero_()

    # -- the forward pass ------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def sparsity(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=features.dtype, device=features.device)

    def forward(self, x: torch.Tensor) -> SaeOutput:
        features = self.encode(x)
        reconstruction = self.decode(features)
        mse = (x - reconstruction).pow(2).sum(dim=-1).mean()
        return SaeOutput(
            features=features,
            reconstruction=reconstruction,
            mse=mse,
            sparsity_term=self.sparsity(features),
            l0=(features > 0).sum(dim=-1).to(mse.dtype).mean(),
        )


class ReluSae(_BaseSae):
    """ReLU with an L1 penalty weighted by the decoder's column norms."""

    needs_norm_constraint = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.pre_activation(x))

    def sparsity(self, features: torch.Tensor) -> torch.Tensor:
        # The norm factor is what makes the penalty ungameable; see the module
        # docstring.
        weighted = features * self.decoder_norms()
        return self.cfg.lambda_sparsity * weighted.sum(dim=-1).mean()


class TopKSae(_BaseSae):
    """Keep the largest k pre-activations.  L0 is exactly k by construction."""

    def __init__(self, cfg: SaeConfig, *, generator=None) -> None:
        if cfg.k is None:
            raise ValueError("TopKSae needs k")
        if not 1 <= cfg.k <= cfg.n_features:
            raise ValueError(f"k must be in [1, {cfg.n_features}], got {cfg.k}")
        super().__init__(cfg, generator=generator)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_activation(x)
        values, indices = z.topk(self.cfg.k, dim=-1)
        out = torch.zeros_like(z)
        # ReLU after the selection, so a selected-but-negative pre-activation
        # contributes nothing rather than a negative feature.
        return out.scatter(-1, indices, torch.relu(values))


class BatchTopKSae(_BaseSae):
    """Keep the largest ``k * batch`` activations across the whole batch.

    Per-example ``L0`` varies, which is the point, and inference needs a
    standalone threshold.  ``threshold`` tracks a running average of the
    cut value seen in training and is what ``eval()`` uses.
    """

    def __init__(self, cfg: SaeConfig, *, generator=None, momentum: float = 0.99) -> None:
        if cfg.k is None:
            raise ValueError("BatchTopKSae needs k")
        super().__init__(cfg, generator=generator)
        self.momentum = momentum
        self.register_buffer(
            "threshold", torch.zeros((), dtype=getattr(torch, cfg.dtype)), persistent=False
        )
        self.register_buffer(
            "threshold_set", torch.zeros((), dtype=torch.bool), persistent=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.pre_activation(x))
        if not self.training:
            if not bool(self.threshold_set):
                raise RuntimeError(
                    "BatchTopKSae has no inference threshold yet: it is estimated during "
                    "training, so the model must be trained (or given one) before eval()"
                )
            return z * (z > self.threshold)
        budget = min(self.cfg.k * z.shape[0], z.numel())
        flat = z.flatten()
        values, indices = flat.topk(budget)
        cut = values[-1].detach()
        with torch.no_grad():
            if bool(self.threshold_set):
                self.threshold.mul_(self.momentum).add_((1 - self.momentum) * cut)
            else:
                self.threshold.copy_(cut)
                self.threshold_set.fill_(True)
        out = torch.zeros_like(flat)
        out = out.scatter(0, indices, values)
        return out.reshape(z.shape)


class JumpReluSae(_BaseSae):
    """``z * H(z - theta)`` with a learned threshold and a straight-through estimator."""

    def __init__(self, cfg: SaeConfig, *, generator=None) -> None:
        super().__init__(cfg, generator=generator)
        dtype = getattr(torch, cfg.dtype)
        self.log_theta = nn.Parameter(
            torch.full((cfg.n_features,), -2.0, dtype=dtype)
        )

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.pre_activation(x))
        return _JumpReLU.apply(z, self.theta, self.cfg.jump_bandwidth)

    def sparsity(self, features: torch.Tensor) -> torch.Tensor:
        # L0 itself, counted through the same straight-through estimator, so
        # the thresholds receive a gradient from the sparsity term at all.
        gate = _StepCount.apply(features, self.theta, self.cfg.jump_bandwidth)
        return self.cfg.lambda_sparsity * gate.sum(dim=-1).mean()


class _JumpReLU(torch.autograd.Function):
    """``z`` where ``z > theta``, zero elsewhere, with a rectangle-kernel gradient."""

    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return z * (z > theta)

    @staticmethod
    def backward(ctx, grad):
        z, theta = ctx.saved_tensors
        eps = ctx.bandwidth
        gate = (z > theta).to(grad.dtype)
        # d/dz: the gate itself, which is the honest derivative away from the
        # threshold.  d/dtheta: a rectangle of width eps around it, which is
        # the estimator -- the true derivative is a delta function.
        near = ((z - theta).abs() < eps / 2).to(grad.dtype) / eps
        return grad * gate, -(grad * z * near).sum(dim=0), None


class _StepCount(torch.autograd.Function):
    """``H(z - theta)`` summed, with the same rectangle-kernel estimator."""

    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad):
        z, theta = ctx.saved_tensors
        eps = ctx.bandwidth
        near = ((z - theta).abs() < eps / 2).to(grad.dtype) / eps
        return grad * near, -(grad * near).sum(dim=0), None


_KINDS = {
    "relu": ReluSae,
    "topk": TopKSae,
    "batchtopk": BatchTopKSae,
    "jumprelu": JumpReluSae,
}


def make_sae(kind: str, cfg: SaeConfig, *, generator=None) -> _BaseSae:
    try:
        builder = _KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown dictionary kind {kind!r}; known: {sorted(_KINDS)}") from None
    return builder(cfg, generator=generator)
