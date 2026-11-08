"""Allocating the residual stream, one disjoint block per variable.

This is the file that makes the compiled regime worth having.  In a trained
model, "which dimensions carry the subject token" is a research question with
no clean answer.  Here it is a table, written down before any weights exist,
and that table *is* the ground truth every discovery method is scored against.

The layout is a direct sum:

* one dimension for the beginning-of-sequence indicator, first, because
  several constructions read it and keeping it at a fixed offset makes the
  weight-building code easier to check by eye;
* ``|vocab|`` dimensions for ``tokens``, one-hot;
* ``max_len`` dimensions for ``indices``, one-hot;
* for each computed variable, its own block -- ``|support|`` dimensions if it
  is categorical, one if it is numerical.

Nothing is ever overwritten.  A variable gets a fresh block and keeps it, so
no compiled component has a side effect, and "what does this head do" has a
one-line answer.  The cost is width: a program with twenty intermediate
variables over a ten-symbol vocabulary is a few hundred dimensions wide.  That
is the right trade for a laboratory, and it is why ``d_model`` here is
whatever the program needs rather than a round number -- forcing it to a
multiple of the head count would pad the stream with dimensions that mean
nothing, in the one regime whose entire value is that every dimension means
something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lang import CompileError, Encoding, Selector, SOp, nodes_in_order

__all__ = ["BasisAllocator", "Layout", "VariableSpace"]


@dataclass(frozen=True)
class VariableSpace:
    """Which dimensions a variable occupies, and what each one means."""

    name: str
    encoding: str
    values: tuple | None  # categorical only; None for numerical
    start: int
    width: int

    @property
    def stop(self) -> int:
        return self.start + self.width

    def dim_of(self, value: object) -> int:
        """The absolute residual dimension holding ``value``'s one-hot entry."""
        if self.values is None:
            raise TypeError(f"{self.name!r} is numerical and has no per-value dimension")
        try:
            return self.start + self.values.index(value)
        except ValueError:
            raise KeyError(
                f"{value!r} is not in the support of {self.name!r}; its support is "
                f"{self.values!r}"
            ) from None

    def __repr__(self) -> str:
        kind = f"{len(self.values)} values" if self.values is not None else "numerical"
        return f"VariableSpace({self.name!r}, [{self.start}:{self.stop}], {kind})"


@dataclass
class Layout:
    """The whole residual-stream allocation for one compiled program."""

    bos_dim: int
    spaces: dict[int, VariableSpace] = field(default_factory=dict)  # keyed by id(node)
    order: list[int] = field(default_factory=list)
    width: int = 0

    def of(self, node: SOp) -> VariableSpace:
        try:
            return self.spaces[id(node)]
        except KeyError:
            raise KeyError(
                f"{node!r} has no allocated block; it is not part of the compiled program"
            ) from None

    def all_spaces(self) -> tuple[VariableSpace, ...]:
        return tuple(self.spaces[key] for key in self.order)

    def describe(self) -> str:
        lines = [f"d_model = {self.width}", f"  [{self.bos_dim}]  bos"]
        lines += [f"  [{s.start}:{s.stop}]  {s.name}" for s in self.all_spaces()]
        return "\n".join(lines)


class BasisAllocator:
    """Hands out disjoint blocks, and asserts disjointness as it goes.

    The invariant is checked here rather than only in a test: an overlapping
    allocation would make two variables alias, and the resulting model would
    compute something coherent but different from its source program -- the
    single worst failure mode available to this project, because the ground
    truth would be wrong rather than the model.
    """

    def __init__(self) -> None:
        self._next = 0
        self._claimed: list[tuple[int, int, str]] = []
        self.layout = Layout(bos_dim=0)
        self.layout.bos_dim = self._claim(1, "bos")

    def _claim(self, width: int, name: str) -> int:
        if width < 1:
            raise ValueError(f"block {name!r} would have width {width}")
        start = self._next
        for lo, hi, other in self._claimed:
            if start < hi and lo < start + width:
                raise AssertionError(
                    f"block {name!r} at [{start}:{start + width}] overlaps {other!r} "
                    f"at [{lo}:{hi}] -- the allocator is broken"
                )
        self._claimed.append((start, start + width, name))
        self._next += width
        self.layout.width = self._next
        return start

    def add(self, node: SOp, support: tuple | None, *, name: str | None = None) -> VariableSpace:
        label = name or node.label
        if id(node) in self.layout.spaces:
            return self.layout.spaces[id(node)]
        if node.encoding == Encoding.CATEGORICAL:
            if support is None:
                raise CompileError(
                    f"{label!r} is categorical but its support could not be bounded, so it "
                    "cannot be stored one-hot. Declare it with declared_support, or make it "
                    "numerical."
                )
            width, values = len(support), tuple(support)
        else:
            width, values = 1, None
        start = self._claim(width, label)
        space = VariableSpace(label, node.encoding, values, start, width)
        self.layout.spaces[id(node)] = space
        self.layout.order.append(id(node))
        return space

    def add_scratch(self, name: str, width: int) -> VariableSpace:
        """A block not corresponding to a program variable.

        Used by ``SelectorWidth``, which needs somewhere to put the raw
        attention weight on the beginning-of-sequence position before an MLP
        turns it into a count.  Scratch blocks appear in the layout -- they are
        real dimensions a discovery method can find -- but they are marked as
        machinery rather than as program variables.
        """
        start = self._claim(width, name)
        space = VariableSpace(name, Encoding.NUMERICAL if width == 1 else Encoding.CATEGORICAL,
                              None, start, width)
        key = -len(self.layout.order) - 1  # negative keys cannot collide with id()
        self.layout.spaces[key] = space
        self.layout.order.append(key)
        return space


def allocate(root: SOp, supports: dict[int, tuple | None]) -> tuple[BasisAllocator, list]:
    """Allocate blocks for every node of a program, in traversal order."""
    alloc = BasisAllocator()
    order = [n for n in nodes_in_order(root) if not isinstance(n, Selector)]
    for node in order:
        alloc.add(node, supports.get(id(node)))
    return alloc, order
