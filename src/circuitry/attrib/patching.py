"""Exact activation patching: the definition every approximation is measured against.

Patching re-runs the forward pass with one activation replaced and asks what
the metric did.  There is nothing clever about it, and that is the point --
this module is the ground truth for phase 4, so it is written to be obviously
right rather than fast, with one exception.

**The exception is batching, and it is not optional.**  Patching one index at
a time costs one forward pass per index, and the experiments in phases 4 and 5
enumerate every head at every position of every program.  But many indices can
be measured in a single pass: replicate the prompt batch ``K`` times and patch
a different index in each replica.  The model never mixes information across
the batch dimension, so the ``K`` measurements are independent and exact.
This turns hours into minutes and is the single largest performance decision
in the repository, so there is a test asserting that ``chunk=1`` -- which is
the naive one-index-per-pass implementation -- and ``chunk=32`` agree exactly.

**Direction and the counterfactual are not the same choice.**  Denoising runs
the *corrupt* prompt and patches the *clean* activation in; there is no choice
of counterfactual to make, because the clean value is what "denoise" means.
Noising runs the clean prompt and patches something else in, and *that* is
where zero, mean, resample and Gaussian differ.  The API reflects this: a
counterfactual with ``direction="denoise"`` is an error rather than being
silently ignored, because silently ignoring it would let an experiment believe
it had compared four counterfactuals when it had run the same measurement four
times.

**Granularity.**  A site has an index space (position, or position and head,
or head and two sequence axes), and patching "layer 8" means nothing without
saying which positions.  The default iterates position and head and pools over
query, key, neuron and vocabulary -- iterating an attention pattern's key axis
is ``T`` times more measurements and is occasionally what you want, never what
you want by accident.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Protocol

import torch

from ..model import sites as _sites
from ..model.hooks import capture, intervene
from .metric import (
    Baselines,
    Direction,
    Metric,
    PromptPair,
    effect_of,
    measure_baselines,
    normalise,
)

__all__ = [
    "Counterfactual",
    "GaussianAblation",
    "MeanAblation",
    "PatchResult",
    "ResampleAblation",
    "ZeroAblation",
    "default_granularity",
    "patch",
    "patch_sites_together",
]

#: Axes iterated one value at a time, against axes pooled into a single patch.
#: Position and head are the units circuits are usually described in; the
#: others are large and are pooled unless asked for explicitly.
DEFAULT_GRANULARITY = {
    "pos": "each",
    "head": "each",
    "query": "all",
    "key": "all",
    "neuron": "all",
    "vocab": "all",
}


class Counterfactual(Protocol):
    """What to patch *in* when noising.  Not used when denoising."""

    name: str

    def value(self, site: str, clean: torch.Tensor, corrupt: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ZeroAblation:
    """Replace with zeros.

    The cheapest counterfactual and the most questionable: a zeroed activation
    puts the network in a state it occupies on no input at all, so the
    behaviour it destroys may be nothing more than the model objecting to being
    off-distribution.  Implemented because it is in wide use and phase 5
    measures what it costs.
    """

    name: str = "zero"

    def value(self, site, clean, corrupt):
        return torch.zeros_like(clean)


@dataclass(frozen=True)
class MeanAblation:
    """Replace with a mean -- and the averaging set is part of the method.

    ``over="batch"`` averages across the prompt distribution and keeps
    positional structure.  ``over="positions"`` averages along the sequence,
    which destroys positional information while looking like it preserved the
    distribution; it is the common bug, implemented deliberately so phase 5
    can measure what it costs rather than asserting it is wrong.  ``over="both"``
    does both.
    """

    over: str = "batch"
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.over not in ("batch", "positions", "both"):
            raise ValueError(f"over must be batch, positions or both; got {self.over!r}")
        object.__setattr__(self, "name", f"mean[{self.over}]")

    def value(self, site, clean, corrupt):
        out = clean
        if self.over in ("batch", "both"):
            out = out.mean(dim=0, keepdim=True).expand_as(clean)
        if self.over in ("positions", "both") and "pos" in _sites.index_space(site):
            axis = _sites.axis_dim(site, "pos")
            out = out.mean(dim=axis, keepdim=True).expand_as(clean)
        return out.contiguous()


@dataclass(frozen=True)
class ResampleAblation:
    """Replace with the value the activation takes on a matched counterfactual prompt.

    The only counterfactual that keeps the model on its own distribution, and
    therefore the one against which the others are judged in phase 5.
    """

    name: str = "resample"

    def value(self, site, clean, corrupt):
        return corrupt


@dataclass(frozen=True)
class GaussianAblation:
    """Replace with noise of the activation's own scale.

    A middle case between zeroing and resampling: on-scale but off-manifold.
    Needs an explicit generator, like everything stochastic here.
    """

    sigma: float = 1.0
    generator: torch.Generator | None = None
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"gaussian[{self.sigma:g}]")

    def value(self, site, clean, corrupt):
        noise = torch.randn(
            clean.shape, generator=self.generator, dtype=clean.dtype, device=clean.device
        )
        return clean.mean() + self.sigma * clean.std() * noise


def default_granularity(**overrides) -> dict[str, str]:
    return {**DEFAULT_GRANULARITY, **overrides}


@dataclass(frozen=True)
class PatchResult:
    """Signed effects, keyed by site and index, with the cost it took.

    ``effects`` holds the *change* a patch produced from its direction's
    unpatched baseline -- positive when denoising ("restored this much"),
    negative when noising ("destroyed this much").  ``levels`` holds the raw
    normalised metric, on the 0-is-corrupt, 1-is-clean scale.  Both are kept
    because the level is what a reader wants on a plot and the effect is what
    can be compared across directions or summed over edges; see
    :func:`circuitry.attrib.metric.effect_of`.
    """

    effects: dict[tuple[str, tuple[tuple[str, int], ...]], float]
    levels: dict[tuple[str, tuple[tuple[str, int], ...]], float]
    direction: str
    counterfactual: str
    forward_passes: int
    baselines: dict
    granularity: dict[str, str]

    def __len__(self) -> int:
        return len(self.effects)

    def level(self, site: str, **index) -> float:
        return self.levels[(site, tuple(sorted(index.items())))]

    def by_site(self) -> dict[str, dict[tuple, float]]:
        out: dict[str, dict[tuple, float]] = {}
        for (site, index), value in self.effects.items():
            out.setdefault(site, {})[index] = value
        return out

    def top(self, n: int = 10) -> list[tuple[str, tuple, float]]:
        """The largest effects by magnitude, which is what a reader wants first."""
        items = [(site, index, value) for (site, index), value in self.effects.items()]
        return sorted(items, key=lambda t: -abs(t[2]))[:n]

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "counterfactual": self.counterfactual,
            "forward_passes": self.forward_passes,
            "baselines": self.baselines,
            "granularity": self.granularity,
            "effects": {
                f"{site}|{'.'.join(f'{k}={v}' for k, v in index)}": value
                for (site, index), value in sorted(self.effects.items())
            },
            "levels": {
                f"{site}|{'.'.join(f'{k}={v}' for k, v in index)}": value
                for (site, index), value in sorted(self.levels.items())
            },
        }


def _index_list(site: str, cfg, batch: int, seq: int, granularity) -> list[dict[str, int]]:
    """Every index to patch separately at this site, given the granularity."""
    space = _sites.index_space(site)
    shape = _sites.shape(site, cfg, batch, seq)
    axes: list[tuple[str, range]] = []
    for axis in space:
        mode = granularity.get(axis, "all")
        if mode == "each":
            axes.append((axis, range(shape[_sites.axis_dim(site, axis)])))
        elif mode != "all":
            raise ValueError(f"granularity for {axis!r} must be 'each' or 'all', got {mode!r}")
    if not axes:
        return [{}]
    names = [a for a, _ in axes]
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*(r for _, r in axes))]


def _patch_hook(site: str, source: torch.Tensor, indices, batch: int):
    """Write each index's source value into its own slice of a replicated batch."""
    dims = {axis: _sites.axis_dim(site, axis) for index in indices for axis in index}

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        for k, index in enumerate(indices):
            target = [slice(None)] * out.ndim
            src = [slice(None)] * source.ndim
            target[0] = slice(k * batch, (k + 1) * batch)
            for axis, value in index.items():
                target[dims[axis]] = value
                src[dims[axis]] = value
            out[tuple(target)] = source[tuple(src)]
        return out

    return hook


def patch(
    model,
    *,
    pair: PromptPair,
    sites,
    direction: Direction,
    counterfactual: Counterfactual | None = None,
    granularity: dict[str, str] | None = None,
    chunk: int = 32,
    baselines: Baselines | None = None,
    metric: Metric | None = None,
    include_expensive: bool = False,
) -> PatchResult:
    """Measure the exact effect of replacing each indexed activation.

    ``direction`` has no default.  See the module docstring: denoising finds
    sufficient sets and noising finds necessary ones, and conflating them is
    the most common way to misread a patching plot.
    """
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")
    if direction == "denoise" and counterfactual is not None:
        raise ValueError(
            "a counterfactual has no meaning when denoising: the patched-in value is the "
            "clean activation by definition. Pass direction='noise' to choose one."
        )
    if direction == "noise" and counterfactual is None:
        raise ValueError(
            "noising needs a counterfactual to patch in; ResampleAblation() is the one "
            "that keeps the model on its own distribution"
        )
    metric = metric or pair.metric
    pair = PromptPair(pair.clean, pair.corrupt, metric, pair.label)
    granularity = granularity or default_granularity()
    cfg = model.cfg
    names = (
        _sites.resolve(sites, cfg, include_expensive=include_expensive)
        if isinstance(sites, str)
        else _sites.resolve(list(sites), cfg, include_expensive=include_expensive)
    )

    if baselines is None:
        baselines = measure_baselines(model, pair)
    passes = 2

    with capture(model, list(names)) as clean_store, torch.inference_mode():
        model(pair.clean)
    with capture(model, list(names)) as corrupt_store, torch.inference_mode():
        model(pair.corrupt)
    passes += 2

    base_tokens = pair.corrupt if direction == "denoise" else pair.clean
    batch, seq = pair.batch, pair.seq

    effects: dict[tuple[str, tuple], float] = {}
    levels: dict[tuple[str, tuple], float] = {}
    for site in names:
        if direction == "denoise":
            source = clean_store[site]
        else:
            source = counterfactual.value(site, clean_store[site], corrupt_store[site])
        indices = _index_list(site, cfg, batch, seq, granularity)
        for start in range(0, len(indices), chunk):
            group = indices[start : start + chunk]
            replicated = base_tokens.repeat(len(group), 1)
            hook = _patch_hook(site, source, group, batch)
            with intervene(model, {site: hook}), torch.inference_mode():
                logits = model(replicated)
            passes += 1
            for k, index in enumerate(group):
                values = metric(logits[k * batch : (k + 1) * batch])
                key = (site, tuple(sorted(index.items())))
                level = normalise(values, baselines)
                levels[key] = level
                effects[key] = effect_of(level, direction)

    return PatchResult(
        effects=effects,
        levels=levels,
        direction=direction,
        counterfactual=counterfactual.name if counterfactual else "clean",
        forward_passes=passes,
        baselines=baselines.report(),
        granularity=dict(granularity),
    )


def patch_sites_together(
    model,
    *,
    pair: PromptPair,
    sites,
    direction: Direction,
    counterfactual: Counterfactual | None = None,
    baselines: Baselines | None = None,
    keep=None,
) -> float:
    """One measurement with *all* the named sites patched at once.

    What faithfulness needs: a circuit is evaluated by patching everything
    outside it simultaneously, not one component at a time.  ``keep`` takes
    ``(site, index)`` and returns ``True`` for the indices to leave *alone*,
    so a caller passes "the circuit" and everything else is ablated.
    """
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")
    if direction == "denoise" and counterfactual is not None:
        raise ValueError("a counterfactual has no meaning when denoising")
    if direction == "noise" and counterfactual is None:
        raise ValueError("noising needs a counterfactual")

    cfg = model.cfg
    names = _sites.resolve(sites if isinstance(sites, str) else list(sites), cfg)
    if baselines is None:
        baselines = measure_baselines(model, pair)
    with capture(model, list(names)) as clean_store, torch.inference_mode():
        model(pair.clean)
    with capture(model, list(names)) as corrupt_store, torch.inference_mode():
        model(pair.corrupt)
    base_tokens = pair.corrupt if direction == "denoise" else pair.clean

    hooks = {}
    for site in names:
        source = (
            clean_store[site]
            if direction == "denoise"
            else counterfactual.value(site, clean_store[site], corrupt_store[site])
        )
        chosen = [
            index
            for index in _all_indices(site, cfg, pair.batch, pair.seq)
            if keep is None or not keep(site, index)
        ]
        if chosen:
            hooks[site] = _multi_index_hook(site, source, chosen)

    if not hooks:
        # Nothing to patch means the base run itself, which is 0 when denoising
        # and 1 when noising by the normalisation convention.
        with torch.inference_mode():
            logits = model(base_tokens)
        return normalise(pair.metric(logits), baselines)

    with intervene(model, hooks), torch.inference_mode():
        logits = model(base_tokens)
    return normalise(pair.metric(logits), baselines)


def _all_indices(site: str, cfg, batch: int, seq: int) -> list[dict[str, int]]:
    """Every concrete index of a site's whole index space."""
    space = _sites.index_space(site)
    shape = _sites.shape(site, cfg, batch, seq)
    ranges = [range(shape[_sites.axis_dim(site, axis)]) for axis in space]
    return [
        dict(zip(space, combo, strict=True)) for combo in itertools.product(*ranges)
    ]


def _multi_index_hook(site: str, source: torch.Tensor, indices):
    """Patch many indices of one site in the same, unreplicated, forward pass."""
    dims = {axis: _sites.axis_dim(site, axis) for index in indices for axis in index}

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        for index in indices:
            selector = [slice(None)] * out.ndim
            for axis, value in index.items():
                selector[dims[axis]] = value
            out[tuple(selector)] = source[tuple(selector)]
        return out

    return hook
