"""What a patching experiment measures, and the convention for reporting it.

Two decisions here carry more weight than the code does.

**Normalisation.**  A raw metric difference is uninterpretable across prompts:
a logit difference of 0.4 means one thing on a pair whose clean-to-corrupt gap
is 0.5 and another on a pair whose gap is 8.  So every patching result is
reported as

    m_norm = (M(patched) - M(corrupt)) / (M(clean) - M(corrupt))

with ``0`` meaning the patch achieved nothing and ``1`` meaning it fully
restored the clean behaviour.  The denominator is per item, and an item whose
denominator is near zero has no signal at all -- clean and corrupt score the
same, so there is nothing for a patch to recover.  Dividing by it is how a
patching plot acquires effects of 40 and a conclusion about a circuit that is
really a conclusion about one bad prompt pair.  :class:`Baselines` therefore
drops those items and *reports how many it dropped*.

**Direction.**  Denoising and noising answer different questions and are not
two ways of doing the same thing:

* *Denoising* runs the corrupt prompt and patches the clean activation in.  A
  large effect means the patched set is **sufficient** to produce the
  behaviour.
* *Noising* runs the clean prompt and patches a counterfactual in.  A large
  effect means the patched set is **necessary**.

A circuit can be sufficient without being necessary and the reverse, so the
API makes ``direction`` a required argument with no default, and every result
carries it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

__all__ = [
    "UNPATCHED_LEVEL",
    "Baselines",
    "Direction",
    "Metric",
    "PromptPair",
    "cross_entropy",
    "effect_of",
    "kl_from",
    "logit_diff",
    "logprob",
    "normalise",
    "normalise_per_item",
]

Direction = Literal["denoise", "noise"]


class Metric(Protocol):
    """Maps logits ``[B, T, V]`` to one scalar per batch item, shape ``[B]``."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...


def logit_diff(correct, incorrect, *, position=-1) -> Metric:
    """The workhorse: how much more the model prefers one token to another.

    ``correct`` and ``incorrect`` are per-item token ids, so a batch may ask a
    different question of every prompt -- which is what the indirect-object
    task needs, since the names change from prompt to prompt.

    ``position`` may be a single index or one index per item.  Per item is not
    a convenience: a batch of compiled-program prompt pairs will differ at
    different output positions, and forcing one shared position would either
    throw most of the batch away or measure a position where the two prompts
    agree, which is the same as measuring nothing.
    """
    correct = torch.as_tensor(correct)
    incorrect = torch.as_tensor(incorrect)
    position = torch.as_tensor(position)

    def metric(logits: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(logits.shape[0], device=logits.device)
        where = position.to(logits.device)
        if where.ndim == 0:
            where = where.expand(logits.shape[0])
        # Negative indices are allowed and are resolved against this batch.
        where = torch.where(where < 0, where + logits.shape[1], where)
        at = logits[rows, where]
        return at[rows, correct.to(at.device)] - at[rows, incorrect.to(at.device)]

    return metric


def logprob(correct, *, position: int = -1) -> Metric:
    correct = torch.as_tensor(correct)

    def metric(logits: torch.Tensor) -> torch.Tensor:
        at = logits[:, position, :].log_softmax(dim=-1)
        rows = torch.arange(at.shape[0], device=at.device)
        return at[rows, correct.to(at.device)]

    return metric


def kl_from(reference: torch.Tensor, *, position: int = -1) -> Metric:
    """``KL(reference || current)`` at one position, per item.

    The asymmetry matters and is the usual choice: it asks how much
    probability mass the reference puts where the patched model does not, so a
    patch that collapses the distribution is penalised heavily.  The reverse
    direction would barely notice.
    """
    ref = reference[:, position, :].log_softmax(dim=-1)

    def metric(logits: torch.Tensor) -> torch.Tensor:
        cur = logits[:, position, :].log_softmax(dim=-1)
        return (ref.exp() * (ref - cur)).sum(dim=-1)

    return metric


def cross_entropy(targets: torch.Tensor) -> Metric:
    """Mean next-token loss per item, over every position that has a target."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        logprobs = logits[:, :-1].log_softmax(dim=-1)
        wanted = targets[:, 1:].to(logits.device)
        picked = logprobs.gather(-1, wanted[..., None])[..., 0]
        return -picked.mean(dim=-1)

    return metric


@dataclass(frozen=True)
class PromptPair:
    """A clean prompt, a counterfactual prompt, and the question asked of both."""

    clean: torch.Tensor
    corrupt: torch.Tensor
    metric: Metric
    label: str = ""

    def __post_init__(self) -> None:
        if self.clean.shape != self.corrupt.shape:
            raise ValueError(
                f"clean and corrupt must have the same shape; got "
                f"{tuple(self.clean.shape)} and {tuple(self.corrupt.shape)}. Patching "
                "requires the two runs to be aligned position by position."
            )

    @property
    def batch(self) -> int:
        return self.clean.shape[0]

    @property
    def seq(self) -> int:
        return self.clean.shape[1]


@dataclass(frozen=True)
class Baselines:
    """The clean and corrupt scores, and which items carry any signal at all."""

    clean: torch.Tensor  # [B]
    corrupt: torch.Tensor  # [B]
    keep: torch.Tensor  # bool [B]
    threshold: float

    @property
    def dropped(self) -> int:
        return int((~self.keep).sum())

    @property
    def kept(self) -> int:
        return int(self.keep.sum())

    @property
    def gap(self) -> torch.Tensor:
        return self.clean - self.corrupt

    def report(self) -> dict:
        return {
            "n_items": int(self.keep.numel()),
            "n_kept": self.kept,
            "n_dropped": self.dropped,
            "degeneracy_threshold": self.threshold,
            "mean_clean": float(self.clean[self.keep].mean()) if self.kept else float("nan"),
            "mean_corrupt": float(self.corrupt[self.keep].mean()) if self.kept else float("nan"),
            "mean_gap": float(self.gap[self.keep].mean()) if self.kept else float("nan"),
        }


def measure_baselines(
    model, pair: PromptPair, *, threshold: float | None = None
) -> Baselines:
    """Score both prompts and decide which items are usable.

    ``threshold`` defaults to a tenth of the median absolute gap, which scales
    with the task instead of assuming a unit.  An item whose gap is below it
    is dropped: the normalised metric would divide by nearly nothing, and a
    single such pair can dominate a whole patching sweep.
    """
    with torch.inference_mode():
        clean = pair.metric(model(pair.clean)).detach().clone()
        corrupt = pair.metric(model(pair.corrupt)).detach().clone()
    gap = (clean - corrupt).abs()
    if threshold is None:
        threshold = 0.1 * float(gap.median())
    keep = gap > threshold
    return Baselines(clean=clean, corrupt=corrupt, keep=keep, threshold=float(threshold))


def normalise_per_item(values: torch.Tensor, baselines: Baselines) -> torch.Tensor:
    """The normalised level for each usable item, before averaging.

    Needed because the sampling error of a patching measurement is a property
    of the *prompt sample*, and estimating it means resampling items.  Averaging
    first throws that away.
    """
    keep = baselines.keep
    num = values[keep] - baselines.corrupt[keep]
    den = baselines.clean[keep] - baselines.corrupt[keep]
    return num / den


def normalise(values: torch.Tensor, baselines: Baselines) -> float:
    """``(patched - corrupt) / (clean - corrupt)``, averaged over usable items.

    This is the *level*, not the effect: 0 is corrupt behaviour and 1 is clean,
    whichever direction the patch was run in.  Use :func:`effect_of` to turn it
    into a signed change.
    """
    if not baselines.kept:
        return float("nan")
    keep = baselines.keep
    num = values[keep] - baselines.corrupt[keep]
    den = baselines.clean[keep] - baselines.corrupt[keep]
    return float((num / den).mean())


#: The normalised level of the *unpatched* run, per direction.  Denoising starts
#: from the corrupt prompt, so nothing-patched reads 0; noising starts from the
#: clean prompt, so nothing-patched reads 1.
UNPATCHED_LEVEL: dict[str, float] = {"denoise": 0.0, "noise": 1.0}


def effect_of(level: float, direction: Direction) -> float:
    """The signed change a patch produced, from its direction's starting level.

    Worth a named function, because conflating the level with the effect is a
    mistake that survives every sanity check.  When denoising, the unpatched
    run reads 0 and the level *is* the effect -- how much of the clean
    behaviour was restored.  When noising, the unpatched run reads 1, so a
    patch that destroys nothing reads 0.95 and a patch that destroys everything
    reads 0.  Reporting 0.95 as an "effect" makes a component that does nothing
    look like the most important thing in the model, and summing such numbers
    over a component's outgoing edges gives an answer larger than the model.

    So: denoising effects are positive and mean "restored this much"; noising
    effects are negative and mean "destroyed this much".  Magnitudes are
    comparable across directions, which is what phase 4 needs.
    """
    if direction not in UNPATCHED_LEVEL:
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")
    return level - UNPATCHED_LEVEL[direction]
