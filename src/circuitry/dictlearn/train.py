"""Training a dictionary, and the three things that decide whether it learns.

**The sparsity coefficient needs a warmup.**  Starting at full strength kills
the dictionary in the first few steps: every feature is penalised before any
of them has learned to reconstruct anything, so they all go to zero and stay
there.  Ramping from zero over the first few hundred steps avoids it, and the
ramp length is reported.

**The activations need normalising, and the scale needs recording.**  Residual
streams at different layers differ in norm by an order of magnitude, so a
reconstruction MSE compared across layers without normalisation is comparing
two different units.  Activations are scaled so that ``E||x|| = sqrt(d)``, the
factor is stored on the report, and anything that uses the dictionary for
intervention has to undo it.

**Dead features need a remedy, and there are two.**  A feature that never
fires contributes nothing and cannot recover on its own, because it receives
no gradient.  Resampling reinitialises it onto an input the dictionary
currently reconstructs badly; an auxiliary loss asks the top dead features to
reconstruct the residual.  Both are implemented so that which one is better is
a measurement rather than a citation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from .sae import SaeConfig, _BaseSae, make_sae

__all__ = ["TrainReport", "normalisation_scale", "train_sae"]


@dataclass
class TrainReport:
    kind: str
    config: SaeConfig
    steps: int
    batch_size: int
    final_mse: float
    final_l0: float
    final_explained_variance: float
    dead_fraction: float
    activation_scale: float
    buffer_capacity: int | None
    lambda_warmup: int
    resampled: int
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "expansion": self.config.expansion,
            "k": self.config.k,
            "lambda_sparsity": self.config.lambda_sparsity,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "final_mse": self.final_mse,
            "final_l0": self.final_l0,
            "final_explained_variance": self.final_explained_variance,
            "dead_fraction": self.dead_fraction,
            "activation_scale": self.activation_scale,
            "buffer_capacity": self.buffer_capacity,
            "lambda_warmup": self.lambda_warmup,
            "resampled": self.resampled,
        }


def normalisation_scale(sample: torch.Tensor) -> float:
    """The factor that makes ``E||x|| = sqrt(d)``.

    Reported with every run because a reconstruction loss is meaningless
    without it: the same dictionary on the same activations at a different
    scale has a different MSE.
    """
    d = sample.shape[-1]
    mean_norm = float(sample.reshape(-1, d).norm(dim=-1).mean())
    if mean_norm == 0:
        return 1.0
    return (d**0.5) / mean_norm


def train_sae(
    kind: str,
    cfg: SaeConfig,
    activations: torch.Tensor | Iterable[torch.Tensor],
    *,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_warmup: int = 200,
    seed: int = 0,
    buffer_capacity: int | None = None,
    resample_every: int | None = None,
    aux_k: int | None = None,
    aux_weight: float = 1.0 / 32,
    log_every: int = 0,
    normalise: bool = True,
) -> tuple[_BaseSae, TrainReport]:
    """Train one dictionary.  ``activations`` is a tensor or a stream of them."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    sae = make_sae(kind, cfg, generator=generator)
    dtype = getattr(torch, cfg.dtype)

    if isinstance(activations, torch.Tensor):
        pool = activations.reshape(-1, activations.shape[-1]).to(dtype)
        scale = normalisation_scale(pool) if normalise else 1.0
        pool = pool * scale

        def draw() -> torch.Tensor:
            idx = torch.randint(0, pool.shape[0], (batch_size,), generator=generator)
            return pool[idx]

        capacity = None
    else:
        from ..util.store import ShuffleBuffer

        capacity = buffer_capacity or 64 * batch_size
        buffer = ShuffleBuffer(
            activations, capacity=capacity, batch_size=batch_size, generator=generator
        )
        first = next(buffer)
        scale = normalisation_scale(first) if normalise else 1.0
        held = [first]

        def draw() -> torch.Tensor:
            if held:
                return held.pop() * scale
            return next(buffer) * scale

    optimiser = torch.optim.AdamW(sae.parameters(), lr=lr)
    history: list = []
    resampled = 0
    target_lambda = cfg.lambda_sparsity

    for step in range(steps):
        x = draw().to(dtype)
        # Ramp the penalty from zero; see the module docstring.
        ramp = min(1.0, (step + 1) / lambda_warmup) if lambda_warmup else 1.0
        object.__setattr__(sae.cfg, "lambda_sparsity", target_lambda * ramp)
        out = sae(x)
        loss = out.loss
        if aux_k:
            loss = loss + aux_weight * _auxiliary_loss(sae, x, out, aux_k)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        sae.renormalise()
        sae.observe(out.features.detach())

        if resample_every and (step + 1) % resample_every == 0 and step + 1 < steps:
            resampled += _resample_dead(sae, x, out, generator)
            sae.reset_counts()
        if log_every and step % log_every == 0:
            history.append(
                {
                    "step": step,
                    "mse": float(out.mse.detach()),
                    "l0": float(out.l0.detach()),
                    "dead_fraction": sae.dead_fraction(),
                }
            )

    object.__setattr__(sae.cfg, "lambda_sparsity", target_lambda)
    with torch.no_grad():
        x = draw().to(dtype)
        out = sae(x)
        variance = (x - x.mean(dim=0)).pow(2).sum(dim=-1).mean()
        explained = 1.0 - float(out.mse / variance.clamp(min=1e-12))
    return sae, TrainReport(
        kind=kind,
        config=cfg,
        steps=steps,
        batch_size=batch_size,
        final_mse=float(out.mse),
        final_l0=float(out.l0),
        final_explained_variance=explained,
        dead_fraction=sae.dead_fraction(),
        activation_scale=scale,
        buffer_capacity=capacity,
        lambda_warmup=lambda_warmup,
        resampled=resampled,
        history=history,
    )


def _auxiliary_loss(sae, x: torch.Tensor, out, aux_k: int) -> torch.Tensor:
    """Ask the top dead features to reconstruct what the live ones missed.

    The dead features get a gradient this way without being forced into the
    main reconstruction, which is what distinguishes it from simply lowering
    the sparsity penalty.
    """
    dead = sae.firing_count == 0
    if not bool(dead.any()):
        return torch.zeros((), dtype=x.dtype, device=x.device)
    residual = (x - out.reconstruction).detach()
    z = sae.pre_activation(x)
    masked = torch.where(dead, z, torch.full_like(z, float("-inf")))
    take = min(aux_k, int(dead.sum()))
    values, indices = masked.topk(take, dim=-1)
    sparse = torch.zeros_like(z).scatter(-1, indices, torch.relu(values))
    guess = sparse @ sae.W_d.T
    return (residual - guess).pow(2).sum(dim=-1).mean()


@torch.no_grad()
def _resample_dead(sae, x: torch.Tensor, out, generator) -> int:
    """Reinitialise dead features onto badly-reconstructed inputs."""
    dead = (sae.firing_count == 0).nonzero(as_tuple=True)[0]
    if dead.numel() == 0:
        return 0
    error = (x - out.reconstruction).pow(2).sum(dim=-1)
    if float(error.sum()) <= 0:
        return 0
    picks = torch.multinomial(error / error.sum(), min(dead.numel(), x.shape[0]),
                              replacement=True, generator=generator)
    chosen = x[picks] - sae.b_d
    chosen = chosen / chosen.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n = min(dead.numel(), chosen.shape[0])
    sae.W_d[:, dead[:n]] = chosen[:n].T
    # The encoder gets a smaller magnitude so the revived feature does not
    # immediately dominate; the usual choice is a fraction of the live mean.
    live = ~(sae.firing_count == 0)
    scale = float(sae.W_e[live].norm(dim=-1).mean()) if bool(live.any()) else 1.0
    sae.W_e[dead[:n]] = chosen[:n] * 0.2 * scale
    sae.b_e[dead[:n]] = 0.0
    return int(n)
