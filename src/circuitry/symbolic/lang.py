"""A small sequence-processing language, and an interpreter for it.

This module is the foundation of the project's one real advantage over
ordinary interpretability work: a model whose algorithm is known because
somebody wrote it down first.  A program here is compiled to exact transformer
weights by :mod:`circuitry.symbolic.compiler`, and the compiler emits the
computational graph it built, which is then what every discovery method is
scored against.

Two consequences shape the design.

* **The interpreter is the oracle for the compiler, so it must be obviously
  correct.**  It is written in plain Python over plain lists, with no
  numerics, no vectorisation and no cleverness.  It is slow, and that is the
  right trade: the compiler's exhaustive verification runs it on every input
  of a small vocabulary, where "obviously correct" is worth more than fast.
  If this file is wrong, every ground truth in the repository is wrong in the
  same direction and nothing downstream will reveal it.
* **Supports are part of the language, not an afterthought.**  Compiling a
  categorical variable means allocating one residual dimension per value it
  can take, so the compiler needs the *finite set* of values each expression
  can produce, over the whole input space.  :func:`support` computes it by
  forward propagation, and an expression whose support cannot be bounded is
  rejected at compile time rather than producing a model that is wrong on
  inputs nobody tried.

Semantics, stated precisely because two of them are easy to get backwards:

* A selector is indexed ``[query, key]``, and the predicate is called as
  ``pred(key_value, query_value)`` -- key first.  This is the convention the
  language this one is modelled on uses, and reversing it silently turns
  ``LT`` into ``GT``.
* ``Aggregate`` over a *categorical* operand requires that each query select
  at most one key.  A mean of one-hot vectors is a distribution, not a value,
  so a query selecting two keys has no categorical answer; the interpreter
  raises :class:`IllDefinedAggregate` rather than inventing one.  The compiler
  checks the same condition over the entire input space before emitting
  weights.

The input sequence does **not** include a beginning-of-sequence symbol.
Compiled models need one -- it is where attention goes when a selector matches
nothing, and it is how selector width is read out -- but that is a fact about
the compilation target, not about the language, so the compiler prepends it
and the interpreter never sees it.  Keeping it out of here is what lets
``indices`` mean "position in the input" in both regimes.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

__all__ = [
    "Aggregate",
    "Encoding",
    "IllDefinedAggregate",
    "Map",
    "Predicate",
    "SOp",
    "Select",
    "Selector",
    "SelectorWidth",
    "SequenceMap",
    "UnboundedValueSet",
    "evaluate",
    "evaluate_selector",
    "indices",
    "nodes_in_order",
    "support",
    "tokens",
]


class IllDefinedAggregate(ValueError):
    """A categorical aggregate whose query selected more than one key."""


class UnboundedValueSet(ValueError):
    """A categorical expression whose set of possible values cannot be bounded."""


# ---------------------------------------------------------------------------
# encodings and predicates
# ---------------------------------------------------------------------------


class Encoding:
    """How a variable occupies the residual stream.

    ``CATEGORICAL`` means a finite value set stored one-hot, which is what
    makes exact lookup-table MLPs and hard attention possible.  ``NUMERICAL``
    means a single dimension holding a magnitude, which is what ``Aggregate``
    produces when it is averaging rather than selecting, and what
    ``SelectorWidth`` reads.
    """

    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    BOTH = (CATEGORICAL, NUMERICAL)


def _as_encoding(value: str) -> str:
    if value not in Encoding.BOTH:
        raise ValueError(f"encoding must be one of {Encoding.BOTH}, got {value!r}")
    return value


class Predicate:
    """The comparisons a selector may use, as named, hashable objects.

    Named rather than arbitrary callables because the compiler has to turn a
    predicate into a matrix of attention logits, and it does that by
    *evaluating* the predicate over the product of the key and query value
    sets.  An arbitrary callable would work for that too, but naming them
    keeps the compiled model's ground-truth graph readable and makes a program
    printable, which matters when a disagreement has to be investigated.
    """

    __slots__ = ("fn", "name")
    _registry: ClassVar[dict[str, Predicate]] = {}

    def __init__(self, name: str, fn: Callable[[object, object], bool]) -> None:
        self.name = name
        self.fn = fn

    def __call__(self, key_value: object, query_value: object) -> bool:
        return bool(self.fn(key_value, query_value))

    def __repr__(self) -> str:
        return f"Predicate.{self.name}"

    def __hash__(self) -> int:
        return hash(("Predicate", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Predicate) and other.name == self.name


def _register(name: str, fn) -> Predicate:
    pred = Predicate(name, fn)
    Predicate._registry[name] = pred
    setattr(Predicate, name, pred)
    return pred


# Comparisons read "key <op> query": EQ is symmetric, but LT means
# key < query, which is the convention the aggregate semantics below assume.
_register("EQ", lambda k, q: k == q)
_register("NEQ", lambda k, q: k != q)
_register("LT", lambda k, q: k < q)
_register("LEQ", lambda k, q: k <= q)
_register("GT", lambda k, q: k > q)
_register("GEQ", lambda k, q: k >= q)
_register("TRUE", lambda k, q: True)
_register("FALSE", lambda k, q: False)


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SOp:
    """A sequence operation: a function from an input sequence to a sequence.

    Compared by identity, not by value.  Two separately-constructed ``Map``
    nodes computing the same thing are two variables and get two residual
    blocks; reuse the same Python object to share one.  Making sharing
    explicit is worth more here than automatic common-subexpression
    elimination, because the ground-truth graph should be the graph the author
    wrote, not a graph an optimiser derived from it.
    """

    label: str = field(default="", kw_only=True)
    encoding: str = field(default=Encoding.CATEGORICAL, kw_only=True)

    def __post_init__(self) -> None:
        self.encoding = _as_encoding(self.encoding)
        if not self.label:
            self.label = self._default_label()

    def _default_label(self) -> str:
        return f"{type(self).__name__.lower()}_{id(self) % 10_000:04d}"

    @property
    def operands(self) -> tuple[SOp | Selector, ...]:
        return ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label!r})"


@dataclass(eq=False)
class _Tokens(SOp):
    def _default_label(self) -> str:
        return "tokens"


@dataclass(eq=False)
class _Indices(SOp):
    def _default_label(self) -> str:
        return "indices"


#: The input symbols.
tokens = _Tokens()
#: The position of each input symbol, counting from zero and excluding BOS.
indices = _Indices()


@dataclass(eq=False)
class Map(SOp):
    """``f`` applied to each element of ``inner``.

    ``declared_support`` is the escape hatch for the one thing forward
    propagation cannot do: turning a *numerical* value into a categorical one.
    A numerical variable's support is not enumerable in general -- the mean of
    a prefix of plus and minus ones takes a different set of values at every
    length -- but a discretisation of it, ``lambda m: "neg" if m < 0 else
    "pos"``, obviously has a small one.  The author declares that set and the
    compiler *verifies* the declaration exhaustively over the input space
    before emitting any weights, so a wrong declaration is a compile error
    rather than a model that is wrong on inputs nobody tried.

    Without this, no counting-and-thresholding program compiles, which rules
    out bracket matching and anything else that asks a numerical question.
    """

    fn: Callable[[object], object]
    inner: SOp
    declared_support: tuple | None = field(default=None, kw_only=True)

    @property
    def operands(self) -> tuple[SOp, ...]:
        return (self.inner,)


@dataclass(eq=False)
class SequenceMap(SOp):
    """``f`` applied elementwise to two sequences.

    The compiler turns this into a lookup table over the *product* of the two
    operands' value sets, so it is the construct that makes a compiled model
    wide.  A ``SequenceMap`` over two 10-value variables needs 100 neurons;
    over two 50-value variables, 2500.  The compiler enforces a configured
    bound and says which ``SequenceMap`` exceeded it.
    """

    fn: Callable[[object, object], object]
    lhs: SOp
    rhs: SOp
    declared_support: tuple | None = field(default=None, kw_only=True)

    @property
    def operands(self) -> tuple[SOp, ...]:
        return (self.lhs, self.rhs)


@dataclass(eq=False)
class Selector:
    """A boolean ``[query, key]`` matrix."""

    label: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{type(self).__name__.lower()}_{id(self) % 10_000:04d}"

    @property
    def operands(self) -> tuple[SOp | Selector, ...]:
        return ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label!r})"


@dataclass(eq=False)
class Select(Selector):
    """``M[query, key] = predicate(keys[key], queries[query])``.

    Note the argument order: the sequence supplying *keys* comes first, and
    the predicate is called key-value first.  ``Select(indices, indices, LT)``
    therefore selects, for each query position, the strictly earlier
    positions.
    """

    keys: SOp
    queries: SOp
    predicate: Predicate

    @property
    def operands(self) -> tuple[SOp, ...]:
        return (self.keys, self.queries)


@dataclass(eq=False)
class Aggregate(SOp):
    """The mean of ``inner`` over the keys each query selects.

    For a ``CATEGORICAL`` result the selection must be at most one key per
    query -- see the module docstring -- and ``default`` is the value when
    nothing is selected.  For a ``NUMERICAL`` result the mean is taken as
    written, and ``default`` applies to an empty selection as before.
    """

    selector: Selector
    inner: SOp
    default: object = None

    @property
    def operands(self) -> tuple[SOp | Selector, ...]:
        return (self.selector, self.inner)


@dataclass(eq=False)
class SelectorWidth(SOp):
    """How many keys each query selects.

    Always ``NUMERICAL`` as computed, but almost always immediately wrapped in
    a ``Map`` back to a categorical value, because that is the form the rest
    of a program can index with.  The compiler implements it as attention to
    the beginning-of-sequence symbol followed by a lookup table, so it costs a
    head and an MLP.
    """

    selector: Selector

    def __post_init__(self) -> None:
        self.encoding = Encoding.NUMERICAL
        super().__post_init__()

    @property
    def operands(self) -> tuple[Selector, ...]:
        return (self.selector,)


# ---------------------------------------------------------------------------
# traversal
# ---------------------------------------------------------------------------


def nodes_in_order(root: SOp | Selector) -> list[SOp | Selector]:
    """Every node reachable from ``root``, operands before the nodes using them.

    A deterministic topological order, which the compiler relies on: it
    allocates residual blocks and emits layers in this order, so a change here
    would change every compiled model's basis.  Depth-first post-order with an
    identity-keyed visited set gives it.
    """
    seen: dict[int, None] = {}
    out: list[SOp | Selector] = []

    def visit(node: SOp | Selector, stack: tuple[int, ...]) -> None:
        if id(node) in seen:
            return
        if id(node) in stack:
            raise ValueError(f"program contains a cycle through {node!r}")
        for operand in node.operands:
            visit(operand, (*stack, id(node)))
        seen[id(node)] = None
        out.append(node)

    visit(root, ())
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate(root: SOp, sequence: Sequence[Hashable]) -> list:
    """Run a program on one input sequence.

    Plain, slow and direct.  This is the reference the compiler is checked
    against, so it does not share code with the compiler and does not try to
    be efficient.
    """
    cache: dict[int, list] = {}
    sel_cache: dict[int, list[list[bool]]] = {}
    seq = list(sequence)
    for node in nodes_in_order(root):
        if isinstance(node, Selector):
            sel_cache[id(node)] = _eval_selector(node, cache, seq)
        else:
            cache[id(node)] = _eval_sop(node, cache, sel_cache, seq)
    return cache[id(root)]


def evaluate_selector(selector: Selector, sequence: Sequence[Hashable]) -> list[list[bool]]:
    """The ``[query][key]`` boolean matrix a selector produces on this input."""
    cache: dict[int, list] = {}
    sel_cache: dict[int, list[list[bool]]] = {}
    seq = list(sequence)
    for node in nodes_in_order(selector):
        if isinstance(node, Selector):
            sel_cache[id(node)] = _eval_selector(node, cache, seq)
        else:
            cache[id(node)] = _eval_sop(node, cache, sel_cache, seq)
    return sel_cache[id(selector)]


def _eval_selector(node: Selector, cache, seq) -> list[list[bool]]:
    if isinstance(node, Select):
        keys = cache[id(node.keys)]
        queries = cache[id(node.queries)]
        return [[node.predicate(k, q) for k in keys] for q in queries]
    raise TypeError(f"cannot evaluate selector of type {type(node).__name__}")


def _eval_sop(node: SOp, cache, sel_cache, seq) -> list:
    if isinstance(node, _Tokens):
        return list(seq)
    if isinstance(node, _Indices):
        return list(range(len(seq)))
    if isinstance(node, Map):
        return [node.fn(v) for v in cache[id(node.inner)]]
    if isinstance(node, SequenceMap):
        lhs, rhs = cache[id(node.lhs)], cache[id(node.rhs)]
        return [node.fn(a, b) for a, b in zip(lhs, rhs, strict=True)]
    if isinstance(node, SelectorWidth):
        matrix = sel_cache[id(node.selector)]
        return [sum(1 for flag in row if flag) for row in matrix]
    if isinstance(node, Aggregate):
        return _eval_aggregate(node, cache, sel_cache)
    raise TypeError(f"cannot evaluate SOp of type {type(node).__name__}")


def _eval_aggregate(node: Aggregate, cache, sel_cache) -> list:
    matrix = sel_cache[id(node.selector)]
    values = cache[id(node.inner)]
    out = []
    for query, row in enumerate(matrix):
        chosen = [values[key] for key, flag in enumerate(row) if flag]
        if not chosen:
            out.append(node.default)
        elif node.encoding == Encoding.CATEGORICAL:
            if len(chosen) > 1:
                raise IllDefinedAggregate(
                    f"categorical Aggregate {node.label!r} selected {len(chosen)} keys "
                    f"at query position {query}: a mean of one-hot values is a "
                    f"distribution, not a value. Selected values: {chosen!r}"
                )
            out.append(chosen[0])
        else:
            out.append(sum(chosen) / len(chosen))
    return out


# ---------------------------------------------------------------------------
# supports
# ---------------------------------------------------------------------------


def support(
    root: SOp,
    vocab: Iterable[Hashable],
    max_len: int,
    *,
    product_limit: int = 4096,
) -> dict[int, tuple | None]:
    """The finite set of values each node can take, or ``None`` if unbounded.

    Returned keyed by ``id(node)``.  The compiler needs this because a
    categorical variable is stored one-hot, so it must know how many values to
    reserve dimensions for; a node whose support is ``None`` cannot be
    compiled categorically and the compiler says so.

    A note on a distinction that is easy to collapse, and that an earlier
    version of this module got wrong.  A node's *encoding* is how it sits in
    the residual stream -- one-hot across many dimensions, or a magnitude in
    one.  Its *support* is the set of values it can take.  Those are
    independent: ``SelectorWidth`` is numerical, because a count is naturally a
    magnitude and reading it out of attention gives a magnitude, but its
    support is exactly ``0..max_len`` and is therefore perfectly well known.
    Keying the support rules on the encoding -- which is what the first version
    did -- makes every ``Map`` over a width unbounded, which rules out counting
    programs: sorting, histograms, most frequent token, essentially everything
    interesting.  The rules below key on node *type* instead.

    ``product_limit`` bounds the lookup table a ``SequenceMap`` may expand to.
    Exceeding it raises rather than emitting an MLP with tens of thousands of
    neurons, because such a model is both slow and useless as a laboratory.
    """
    vocab = tuple(dict.fromkeys(vocab))
    if max_len < 1:
        raise ValueError(f"max_len must be at least 1, got {max_len}")
    out: dict[int, tuple | None] = {}
    for node in nodes_in_order(root):
        if isinstance(node, Selector):
            continue
        out[id(node)] = _node_support(node, out, vocab, max_len, product_limit)
    return out


def _sorted_values(values: Iterable) -> tuple:
    unique = tuple(dict.fromkeys(values))
    try:
        return tuple(sorted(unique, key=lambda v: (v is None, v)))
    except TypeError:
        # Mixed, mutually incomparable types: keep insertion order so the basis
        # is at least deterministic, and say nothing about ordering.
        return unique


def _node_support(node, out, vocab, max_len, product_limit) -> tuple | None:
    declared = getattr(node, "declared_support", None)
    if declared is not None:
        # A claim, not a derivation.  The compiler checks it exhaustively; see
        # Map.declared_support for why the escape hatch has to exist.
        return _sorted_values(declared)
    if isinstance(node, _Tokens):
        return _sorted_values(vocab)
    if isinstance(node, _Indices):
        return tuple(range(max_len))
    if isinstance(node, SelectorWidth):
        # Numerical, but bounded by construction: a query can select no key or
        # every key, and nothing in between is excluded.
        return tuple(range(max_len + 1))
    if isinstance(node, Map):
        inner = out[id(node.inner)]
        if inner is None:
            return None
        return _sorted_values(node.fn(v) for v in inner)
    if isinstance(node, SequenceMap):
        lhs, rhs = out[id(node.lhs)], out[id(node.rhs)]
        if lhs is None or rhs is None:
            return None
        size = len(lhs) * len(rhs)
        if size > product_limit:
            raise UnboundedValueSet(
                f"SequenceMap {node.label!r} would need a lookup table over "
                f"{len(lhs)}x{len(rhs)}={size} input pairs, above the limit of "
                f"{product_limit}. Reduce the operands' supports or raise "
                "product_limit deliberately."
            )
        return _sorted_values(node.fn(a, b) for a, b in itertools.product(lhs, rhs))
    if isinstance(node, Aggregate):
        inner = out[id(node.inner)]
        if node.encoding == Encoding.NUMERICAL:
            # A mean of magnitudes: the support is the set of means of every
            # subset, which is not worth enumerating and is not needed, since a
            # numerical variable occupies one dimension however many values it
            # takes.
            return None
        if inner is None:
            return None
        return _sorted_values((*inner, node.default))
    raise TypeError(f"no support rule for {type(node).__name__}")
