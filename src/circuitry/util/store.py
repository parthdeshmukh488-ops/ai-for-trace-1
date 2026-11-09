"""Activation storage with a budget that is checked before it is exceeded.

The memory cliffs in this package are sharp and badly signposted.  Per-head
attention results are ``[batch, pos, head, d_model]``: for GPT-2 small at
batch 8 and 512 positions that is 151 MB *per layer*, so capturing them across
twelve layers is 1.8 GB.  Attention patterns are ``[batch, head, pos, pos]``
and quadratic in sequence length: 402 MB per layer at 1024 positions.  Neither
is unreasonable to want; both are unreasonable to discover by having the
process killed.

So a capture states its budget, and the store computes the requirement from
the site shapes *before* allocating anything and raises with the breakdown.
The alternative -- allocate and hope -- produces a bare OOM with no indication
of which site was responsible, which is a bad trade for the two lines of
arithmetic it saves.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch

__all__ = ["ActivationBudgetExceeded", "ActivationStore", "bytes_for"]

_DTYPE_BYTES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.bool: 1,
}


class ActivationBudgetExceeded(RuntimeError):
    """Raised before allocating, naming the sites and the overage."""

    def __init__(self, required: int, budget: int, breakdown: Mapping[str, int]) -> None:
        worst = sorted(breakdown.items(), key=lambda kv: -kv[1])[:5]
        detail = ", ".join(f"{name}={_human(size)}" for name, size in worst)
        super().__init__(
            f"capture needs {_human(required)} but the budget is {_human(budget)} "
            f"(over by {_human(required - budget)}); largest sites: {detail}"
        )
        self.required = required
        self.budget = budget
        self.breakdown = dict(breakdown)


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    raise AssertionError("unreachable")


def bytes_for(shape: Sequence[int], dtype: torch.dtype) -> int:
    """Storage for one dense tensor of this shape and dtype."""
    try:
        width = _DTYPE_BYTES[dtype]
    except KeyError as exc:  # pragma: no cover - guards a future dtype
        raise ValueError(f"no byte width known for dtype {dtype}") from exc
    total = width
    for dim in shape:
        if dim < 0:
            raise ValueError(f"shape {tuple(shape)} has a negative dimension")
        total *= dim
    return total


class ActivationStore:
    """A name-keyed collection of activations with an optional byte budget.

    Not a plain dict, for two reasons.  It enforces the budget, and it refuses
    to overwrite a site silently: a second write to the same name during one
    forward pass means either a loop body that was entered twice or a site name
    that is not as unique as it looked, and both are bugs worth hearing about
    rather than a last-write-wins policy.
    """

    def __init__(self, *, budget_bytes: int | None = None) -> None:
        if budget_bytes is not None and budget_bytes < 0:
            raise ValueError(f"budget_bytes must be non-negative, got {budget_bytes}")
        self._budget = budget_bytes
        self._data: dict[str, torch.Tensor] = {}
        self._used = 0

    # -- planning ---------------------------------------------------------

    def check_plan(self, plan: Mapping[str, tuple[Sequence[int], torch.dtype]]) -> int:
        """Validate a whole capture before any of it is allocated.

        ``plan`` maps site name to ``(shape, dtype)``.  Returns the total
        requirement in bytes, or raises :class:`ActivationBudgetExceeded`.
        """
        breakdown = {name: bytes_for(shape, dtype) for name, (shape, dtype) in plan.items()}
        required = sum(breakdown.values())
        if self._budget is not None and self._used + required > self._budget:
            raise ActivationBudgetExceeded(self._used + required, self._budget, breakdown)
        return required

    # -- mapping-ish interface -------------------------------------------

    def store(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._data:
            raise KeyError(
                f"site {name!r} was already captured in this store; a second write means "
                "either a duplicated hook or a site name that is not unique"
            )
        size = bytes_for(tensor.shape, tensor.dtype)
        if self._budget is not None and self._used + size > self._budget:
            raise ActivationBudgetExceeded(self._used + size, self._budget, {name: size})
        self._data[name] = tensor
        self._used += size

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self._data[name]
        except KeyError:
            known = ", ".join(sorted(self._data)) or "nothing"
            raise KeyError(f"site {name!r} was not captured; this store holds: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def budget_bytes(self) -> int | None:
        return self._budget

    def detach(self) -> ActivationStore:
        """A copy holding detached tensors, for caches that outlive a graph."""
        out = ActivationStore(budget_bytes=self._budget)
        for name, tensor in self._data.items():
            out.store(name, tensor.detach())
        return out

    def __repr__(self) -> str:
        return (
            f"ActivationStore({len(self._data)} sites, {_human(self._used)}"
            + (f" of {_human(self._budget)}" if self._budget is not None else "")
            + ")"
        )


class ShuffleBuffer:
    """A streaming, shuffling activation source for dictionary training.

    Training a sparse autoencoder on a language model's residuals needs more
    activations than fit in memory, so they arrive as a stream of forward
    passes.  The trouble is that a stream of forward passes is *ordered*:
    consecutive rows come from consecutive tokens of the same document, and
    they are strongly correlated.  A dictionary trained on that learns document
    structure instead of features, and nothing about its reconstruction loss
    says so.

    So rows are pooled into a buffer and drawn from it at random.  The buffer's
    *size* is therefore a confound rather than a detail -- a buffer of one
    batch does no shuffling at all -- and it is reported with every run.
    ``tests/test_store.py`` measures the decorrelation it achieves rather than
    assuming it.

    The refill policy is to top up whenever the buffer drops below half
    capacity, which keeps the mixing depth roughly constant instead of letting
    it decay as the buffer drains.
    """

    def __init__(
        self,
        source,
        *,
        capacity: int,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if capacity < batch_size:
            raise ValueError(
                f"capacity {capacity} is below the batch size {batch_size}; the buffer "
                "would do no shuffling at all"
            )
        self._source = iter(source)
        self.capacity = capacity
        self.batch_size = batch_size
        self._generator = generator or torch.Generator()
        self._buffer: torch.Tensor | None = None
        self.exhausted = False
        self.rows_seen = 0

    def _refill(self) -> None:
        chunks = [] if self._buffer is None else [self._buffer]
        have = 0 if self._buffer is None else self._buffer.shape[0]
        while have < self.capacity and not self.exhausted:
            try:
                chunk = next(self._source)
            except StopIteration:
                self.exhausted = True
                break
            chunk = chunk.reshape(-1, chunk.shape[-1]).detach()
            chunks.append(chunk)
            have += chunk.shape[0]
            self.rows_seen += chunk.shape[0]
        self._buffer = torch.cat(chunks, dim=0) if chunks else None

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self._buffer is None or self._buffer.shape[0] < self.capacity // 2:
            self._refill()
        if self._buffer is None or self._buffer.shape[0] < self.batch_size:
            raise StopIteration
        order = torch.randperm(self._buffer.shape[0], generator=self._generator)
        take, keep = order[: self.batch_size], order[self.batch_size :]
        batch = self._buffer[take]
        self._buffer = self._buffer[keep]
        return batch
