"""Reading and writing activations mid-forward, reversibly.

The engine calls into a registry at every site rather than using framework
module hooks, for three reasons.

* **Reversibility is structural.**  A registry is a dictionary the context
  manager empties on exit, so there is no handle to leak.  The contract --
  after any intervention block, the model produces *bitwise* the same output
  as before it -- is then a property of the design rather than of remembering
  to deregister, and ``tests/test_hooks.py`` checks it with ``torch.equal``
  rather than ``allclose``.  Every number in this repository is downstream of
  that holding.
* **Sites are not modules.**  ``attn.scores`` is an intermediate inside one
  module and ``resid_mid`` is the sum of two of them; neither has a module
  whose output it is, so a module hook cannot name them.
* **The engine can skip work nobody asked for.**  ``attn.result`` costs
  ``n_heads`` times what ``attn.out`` does, so the forward pass materialises
  it only when :meth:`HookRegistry.wanted` says somebody is listening.

A hook takes the tensor and returns a replacement, or returns ``None`` to
observe without changing anything.  Several hooks on one site compose in
registration order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager

import torch

from ..util.store import ActivationStore
from . import sites as _sites

__all__ = [
    "HookRegistry",
    "add",
    "capture",
    "freeze",
    "intervene",
    "project_out",
    "replace",
    "replace_at",
    "scale",
]

Hook = Callable[[torch.Tensor], torch.Tensor | None]


class HookRegistry:
    """Site name -> the hooks listening on it, applied in registration order."""

    __slots__ = ("_hooks",)

    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = {}

    def wanted(self, name: str) -> bool:
        """Whether anything is listening, so the engine can skip expensive sites."""
        return name in self._hooks

    def any_wanted(self, names: Sequence[str]) -> bool:
        return any(name in self._hooks for name in names)

    def add(self, name: str, fn: Hook) -> None:
        self._hooks.setdefault(name, []).append(fn)

    def clear(self) -> None:
        self._hooks.clear()

    def snapshot(self) -> dict[str, list[Hook]]:
        return {name: list(fns) for name, fns in self._hooks.items()}

    def restore(self, snapshot: Mapping[str, list[Hook]]) -> None:
        self._hooks = {name: list(fns) for name, fns in snapshot.items()}

    def __call__(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        fns = self._hooks.get(name)
        if not fns:
            return tensor
        for fn in fns:
            out = fn(tensor)
            if out is not None:
                tensor = out
        return tensor

    def __repr__(self) -> str:
        return f"HookRegistry({len(self._hooks)} sites)"


# ---------------------------------------------------------------------------
# contexts
# ---------------------------------------------------------------------------


@contextmanager
def intervene(model, hooks: Mapping[str, Hook | Sequence[Hook]]) -> Iterator[None]:
    """Install hooks for the duration of the block, then restore exactly.

    Nests: an inner block's hooks are added to the outer block's and removed
    on exit, which is what lets a patching experiment run inside a capture
    without the capture seeing a half-installed state.
    """
    _sites.validate(list(hooks), model.cfg)
    before = model.hooks.snapshot()
    try:
        for name, fn in hooks.items():
            for one in fn if isinstance(fn, (list, tuple)) else [fn]:
                model.hooks.add(name, one)
        yield
    finally:
        model.hooks.restore(before)


@contextmanager
def capture(
    model,
    site_names: str | Sequence[str],
    *,
    budget_bytes: int | None = None,
    expect: tuple[int, int] | None = None,
    detach: bool = True,
    include_expensive: bool = False,
) -> Iterator[ActivationStore]:
    """Record activations at ``site_names`` into a store.

    ``site_names`` may be concrete names or glob patterns; patterns are
    expanded by :func:`circuitry.model.sites.resolve`, which raises rather than
    matching nothing.

    ``expect=(batch, seq)`` turns the budget into a *pre-flight* check: the
    requirement for the whole capture is computed from the site shapes and
    rejected before a byte is allocated.  Without it the budget is still
    enforced, but incrementally, so the site that breaks it is named and the
    ones before it have already been allocated.  Pass ``expect`` when the cost
    is in question; for GPT-2 small at 512 positions it usually is.

    ``detach=True`` by default.  A capture used as a patching source must not
    hold the autograd graph alive, and keeping it is both a memory leak and a
    way to accidentally backpropagate through a cached value.  Gradient work
    passes ``detach=False`` deliberately.
    """
    names = (
        _sites.resolve(site_names, model.cfg, include_expensive=include_expensive)
        if isinstance(site_names, str) or not _all_concrete(site_names)
        else _sites.validate(site_names, model.cfg)
    )
    store = ActivationStore(budget_bytes=budget_bytes)
    if expect is not None:
        batch, seq = expect
        dtype = model.dtype
        store.check_plan({name: (_sites.shape(name, model.cfg, batch, seq), dtype) for name in names})

    def make(name: str) -> Hook:
        def record(tensor: torch.Tensor) -> None:
            store.store(name, tensor.detach() if detach else tensor)
            return None

        return record

    with intervene(model, {name: make(name) for name in names}):
        yield store


def _all_concrete(names: Sequence[str]) -> bool:
    return all(not any(ch in name for ch in "*?[") and "-" not in name for name in names)


# ---------------------------------------------------------------------------
# combinators
# ---------------------------------------------------------------------------


def replace(value: torch.Tensor) -> Hook:
    """Substitute the whole tensor."""

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape != value.shape:
            raise ValueError(
                f"replacement has shape {tuple(value.shape)} but the site holds "
                f"{tuple(tensor.shape)}"
            )
        return value.to(dtype=tensor.dtype, device=tensor.device)

    return hook


def replace_at(site: str, index: Mapping[str, int | slice], value) -> Hook:
    """Substitute at a slice of the site's index space.

    ``index`` is keyed by axis name -- ``{"pos": 5, "head": 3}`` -- so the
    caller does not have to know which tensor dimension a head is, and the
    same call works for ``attn.z`` and ``attn.pattern`` even though their
    layouts differ.  See :func:`circuitry.model.sites.axis_dim`.
    """
    dims = {axis: _sites.axis_dim(site, axis) for axis in index}

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        selector: list[object] = [slice(None)] * tensor.ndim
        for axis, key in index.items():
            selector[dims[axis]] = key
        out = tensor.clone()
        src = value
        if isinstance(src, torch.Tensor):
            src = src.to(dtype=tensor.dtype, device=tensor.device)
        out[tuple(selector)] = src
        return out

    return hook


def add(vector: torch.Tensor, *, strength: float = 1.0, positions=None) -> Hook:
    """Add a direction into a residual-stream site: the steering primitive."""

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        delta = (strength * vector).to(dtype=tensor.dtype, device=tensor.device)
        out = tensor.clone()
        if positions is None:
            out = out + delta
        else:
            out[:, positions] = out[:, positions] + delta
        return out

    return hook


def scale(factor: float) -> Hook:
    """Multiply a site by a constant; ``factor=0`` is zero ablation."""

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * factor

    return hook


def project_out(directions: torch.Tensor) -> Hook:
    """Remove the component along one or more directions.

    ``directions`` is ``[d]`` or ``[k, d]``.  It is orthonormalised first:
    projecting out two near-parallel directions by subtracting each in turn
    removes roughly twice the shared component, which is a quiet way to damage
    a residual stream far more than intended.
    """
    basis = directions if directions.ndim == 2 else directions[None, :]
    q, _ = torch.linalg.qr(basis.T.to(torch.float64))
    q = q.T

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        proj = q.to(dtype=tensor.dtype, device=tensor.device)
        coeffs = tensor @ proj.T
        return tensor - coeffs @ proj

    return hook


def freeze(store: ActivationStore, site: str) -> Hook:
    """Pin a site to a cached value.  The primitive path patching is built from."""
    cached = store[site]

    def hook(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape != cached.shape:
            raise ValueError(
                f"frozen value for {site!r} has shape {tuple(cached.shape)} but the site "
                f"holds {tuple(tensor.shape)}; the cache was taken at a different "
                "batch or sequence length"
            )
        return cached.to(dtype=tensor.dtype, device=tensor.device)

    return hook
