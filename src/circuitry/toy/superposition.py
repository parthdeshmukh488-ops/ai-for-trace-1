"""A model whose features are known, because they were planted.

The point of this file is that it is the only place in interpretability where
"did the dictionary find the right features" is a measurement rather than a
judgement.  A sparse autoencoder is supposed to undo superposition -- to
recover a network's true, sparsely-active features from the dense directions
they are packed into -- and that claim is normally assessed by reconstruction
loss and L0, neither of which is the thing being claimed.

Here the features are put in by hand.  ``n`` features, each with an importance
and a sparsity, are projected into ``m < n`` dimensions by a learned ``W``, and
feature ``i``'s direction in that hidden space is exactly ``W[:, i]``.  Train a
dictionary on the hidden activations and the recovery question has an answer:
for each true direction, how well does the best learned direction match it.

The construction, following the standard one:

    x in R^n,  x_i = 0 with probability S, else ~ U[0, 1]
    h  = W x                                   W in R^{m x n}
    x' = ReLU(W^T h + b)
    loss = sum_i I_i E[(x_i - x'_i)^2]

Tying the decoder to ``W^T`` is what forces superposition: the model has to
both store and retrieve ``n`` features through ``m`` dimensions, so it packs
them at angles rather than orthogonally, and how it packs them depends on the
sparsity.  Which is the reason recovery has to be reported *as a function of
sparsity* rather than as one number -- superposition is a function of sparsity,
so a dictionary's success at undoing it must be too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["SuperpositionModel", "SuperpositionSpec", "train_superposition"]


@dataclass(frozen=True)
class SuperpositionSpec:
    """How many features, how sparse, how much room, and how much each matters."""

    n_features: int
    n_hidden: int
    sparsity: float  # probability a feature is *inactive*
    importance: str = "uniform"  # "uniform" | "decay"
    decay: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        if self.n_features < 1 or self.n_hidden < 1:
            raise ValueError("n_features and n_hidden must be positive")
        if self.importance not in ("uniform", "decay"):
            raise ValueError(f"importance must be uniform or decay, got {self.importance!r}")

    @property
    def crowding(self) -> float:
        """Features per hidden dimension.  Above 1 there must be superposition."""
        return self.n_features / self.n_hidden

    def importances(self, dtype=torch.float64) -> torch.Tensor:
        if self.importance == "uniform":
            return torch.ones(self.n_features, dtype=dtype)
        return self.decay ** torch.arange(self.n_features, dtype=dtype)


class SuperpositionModel(nn.Module):
    def __init__(self, spec: SuperpositionSpec, *, generator=None, dtype=torch.float64) -> None:
        super().__init__()
        self.spec = spec
        self.W = nn.Parameter(
            torch.randn(
                spec.n_hidden, spec.n_features, generator=generator, dtype=dtype
            )
            / spec.n_hidden**0.5
        )
        self.b = nn.Parameter(torch.zeros(spec.n_features, dtype=dtype))

    def sample(self, batch: int, *, generator=None) -> torch.Tensor:
        """Sparse feature vectors: mostly zero, uniform where active."""
        spec = self.spec
        dtype = self.W.dtype
        values = torch.rand(batch, spec.n_features, generator=generator, dtype=dtype)
        active = (
            torch.rand(batch, spec.n_features, generator=generator, dtype=dtype)
            >= spec.sparsity
        )
        return values * active

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.hidden(x) @ self.W + self.b)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        importance = self.spec.importances(self.W.dtype).to(self.W.device)
        return (importance * (x - self(x)).pow(2)).sum(dim=-1).mean()

    @property
    def true_directions(self) -> torch.Tensor:
        """``[n_features, n_hidden]``: the ground truth, one row per feature."""
        return self.W.detach().T

    def feature_norms(self) -> torch.Tensor:
        """How much of each feature the model actually represents.

        A feature the model gave up on has a near-zero column, and no
        dictionary can recover a direction that is not there.  Recovery
        metrics must therefore be reported over the features the model
        *represents*, which this makes checkable.
        """
        return self.W.detach().norm(dim=0)


def train_superposition(
    spec: SuperpositionSpec,
    *,
    steps: int = 4000,
    batch: int = 1024,
    lr: float = 1e-2,
    seed: int = 0,
    dtype=torch.float64,
    log_every: int = 0,
) -> tuple[SuperpositionModel, dict]:
    """Train the planted model.  Returns it and a short report."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    model = SuperpositionModel(spec, generator=generator, dtype=dtype)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        x = model.sample(batch, generator=generator)
        loss = model.loss(x)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if log_every and step % log_every == 0:
            history.append({"step": step, "loss": float(loss)})
    with torch.no_grad():
        final = float(model.loss(model.sample(4096, generator=generator)))
        norms = model.feature_norms()
    return model, {
        "final_loss": final,
        "steps": steps,
        "crowding": spec.crowding,
        "sparsity": spec.sparsity,
        # How many features the model bothered to represent at all.
        "n_represented": int((norms > 0.1 * norms.max()).sum()),
        "n_features": spec.n_features,
        "history": history,
    }
