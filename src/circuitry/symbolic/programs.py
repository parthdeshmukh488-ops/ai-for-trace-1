"""The program suite: what the compiled regime is measured on.

A suite rather than one example, and reported per program rather than pooled,
because the programs differ in exactly the way that matters.  ``shift`` has a
three-node graph and any method will recover it; ``dyck1`` has sixteen edges
across six layers and nine unused heads to be wrong about.  A single headline
number would be carried by the easy ones.

Each entry states the depth of its dependency chain and what structure it is
there to exercise.  Two are load-bearing:

* ``two_hop`` is the smallest program with a genuine two-step composition --
  ``tokens[tokens[i]]`` -- which is the structure first-order attribution
  methods are expected to get wrong, so phase 4 reports it separately.
* ``dyck1`` is the deepest, and is built entirely out of categorical
  variables.  The obvious formulation uses a running numerical balance and a
  threshold, which needs a magnitude decoded back into a category; replacing
  the balance with a difference of two counts avoids that and compiles to a
  cleaner graph.  Worth knowing that the counting formulation exists.

A note on what is *not* here.  Dyck-2 -- bracket matching with two bracket
types -- was specified and is not implemented, because it is not expressible
in this language.  Checking a closer's type against its matching opener means
finding the *most recent* unmatched opener at the current depth, and "most
recent among the positions satisfying a condition that depends on the query"
is a three-index quantity: a selector produces one boolean per
``(query, key)`` pair, and ranking the selected keys within each query's own
selection is not something any composition of selectors and aggregates
expresses.  Dyck-1 needs no such ranking because a count suffices.  The gap is
a property of the language rather than of this file, and ``unique`` takes the
slot as a second four-deep composed program.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .lang import (
    Aggregate,
    Encoding,
    Map,
    Predicate,
    Select,
    SelectorWidth,
    SequenceMap,
    SOp,
    indices,
    tokens,
)

__all__ = ["PROGRAMS", "ProgramSpec", "build", "names"]

#: Stands in for "no index", in the key-encoding idiom used throughout below.
#: A selector compares one key value with one query value through a single
#: predicate, so a conjunction such as "position j is earlier than i *and*
#: holds an opening bracket" cannot be written directly.  The idiom is to fold
#: the second condition into the key: give a position its own index when the
#: condition holds and an unreachable sentinel when it does not, then compare
#: with ``LEQ``.  ``NEVER`` is that sentinel, and it has to exceed any real
#: index, which is why it is checked against ``max_len`` at build time.
NEVER = 9999


@dataclass(frozen=True)
class ProgramSpec:
    """One program, with the vocabulary it is meant to run over."""

    name: str
    depth: int
    vocab: tuple
    exercises: str
    make: Callable[[], SOp]

    def build(self) -> SOp:
        return self.make()


def _shift() -> SOp:
    previous = Map(lambda i: i - 1, indices, label="previous")
    return Aggregate(
        Select(indices, previous, Predicate.EQ), tokens, default="_", label="shift"
    )


def _hist() -> SOp:
    return SelectorWidth(Select(tokens, tokens, Predicate.EQ), label="hist")


def _two_hop() -> SOp:
    return Aggregate(
        Select(indices, tokens, Predicate.EQ), tokens, default=-1, label="two_hop"
    )


def _reverse() -> SOp:
    length = SelectorWidth(Select(tokens, tokens, Predicate.TRUE), label="length")
    target = SequenceMap(lambda n, i: n - 1 - i, length, indices, label="target")
    return Aggregate(
        Select(indices, target, Predicate.EQ), tokens, default="_", label="reverse"
    )


def _sort() -> SOp:
    # Rank by (token, index) rather than by token alone, so that equal tokens
    # get distinct ranks.  Ranking by token alone makes the final aggregate
    # ill defined on any input with a repeat -- two keys for one query -- and
    # the compiler rejects it, correctly.
    keyed = SequenceMap(lambda t, i: (t, i), tokens, indices, label="sort_key")
    rank = SelectorWidth(Select(keyed, keyed, Predicate.LT), label="rank")
    return Aggregate(
        Select(rank, indices, Predicate.EQ), tokens, default="_", label="sort"
    )


def _most_freq() -> SOp:
    count = SelectorWidth(Select(tokens, tokens, Predicate.EQ), label="count")
    # Negated count so that ascending rank is descending frequency, with the
    # index breaking ties the same way _sort does.
    keyed = SequenceMap(lambda c, i: (-c, i), count, indices, label="freq_key")
    rank = SelectorWidth(Select(keyed, keyed, Predicate.LT), label="freq_rank")
    return Aggregate(
        Select(rank, indices, Predicate.EQ), tokens, default="_", label="most_freq"
    )


def _unique() -> SOp:
    """How many distinct tokens the whole sequence contains.

    A position is the first occurrence of its token when no earlier position
    holds the same token, which is a count of zero; the answer is then the
    number of first occurrences.  Two chained counts, so four deep.
    """
    keyed = SequenceMap(lambda t, i: (t, i), tokens, indices, label="tok_idx")
    earlier_same = SelectorWidth(
        Select(keyed, keyed, Predicate.LT), label="earlier_or_same"
    )
    earlier_identical = SelectorWidth(
        Select(
            SequenceMap(lambda t, i: (t, i), tokens, indices, label="tok_idx_2"),
            keyed,
            _same_token_earlier(),
        ),
        label="earlier_identical",
    )
    is_first = Map(lambda n: n == 0, earlier_identical, label="is_first")
    first_key = SequenceMap(
        lambda flag, i: i if flag else NEVER, is_first, indices, label="first_key"
    )
    del earlier_same
    return SelectorWidth(
        Select(first_key, Map(lambda _i: NEVER - 1, indices, label="all_positions"),
               Predicate.LEQ),
        label="unique",
    )


def _same_token_earlier() -> Predicate:
    """``key`` holds the same token as ``query`` and comes strictly earlier."""
    return Predicate("SAME_TOKEN_EARLIER", lambda k, q: k[0] == q[0] and k[1] < q[1])


def _dyck1() -> SOp:
    """Validity of every prefix of a one-bracket-type string.

    ``T`` where the prefix is balanced, ``P`` where it is a valid but
    incomplete prefix, ``F`` once it has gone wrong and from then on.

    Built out of counts rather than a running numerical balance.  The
    numerical formulation needs the balance discretised back into a category
    before anything can branch on it, which costs a declared support and a
    step decoder; two counts and a subtraction stay categorical throughout.
    """
    open_key = SequenceMap(
        lambda t, i: i if t == "(" else NEVER, tokens, indices, label="open_key"
    )
    close_key = SequenceMap(
        lambda t, i: i if t == ")" else NEVER, tokens, indices, label="close_key"
    )
    n_open = SelectorWidth(Select(open_key, indices, Predicate.LEQ), label="n_open")
    n_close = SelectorWidth(Select(close_key, indices, Predicate.LEQ), label="n_close")
    balance = SequenceMap(lambda a, b: a - b, n_open, n_close, label="balance")
    negative_key = SequenceMap(
        lambda b, i: i if b < 0 else NEVER, balance, indices, label="negative_key"
    )
    ever_negative = SelectorWidth(
        Select(negative_key, indices, Predicate.LEQ), label="ever_negative"
    )
    return SequenceMap(
        lambda b, bad: "F" if bad > 0 else ("T" if b == 0 else "P"),
        balance,
        ever_negative,
        label="dyck1",
    )


def _pair_balance() -> SOp:
    """The running fraction of opening brackets: the numerical path.

    The only program in the suite whose root is a magnitude rather than a
    category, which makes it the one that exercises numerical ``Aggregate``,
    the numerical read-out, and the default-through-BOS on a magnitude.
    """
    step = Map(
        lambda t: 1.0 if t == "(" else -1.0,
        tokens,
        encoding=Encoding.NUMERICAL,
        label="step",
    )
    return Aggregate(
        Select(indices, indices, Predicate.LEQ),
        step,
        default=0.0,
        encoding=Encoding.NUMERICAL,
        label="pair_balance",
    )


PROGRAMS: dict[str, ProgramSpec] = {
    spec.name: spec
    for spec in (
        ProgramSpec("shift", 2, ("a", "b"), "index arithmetic and one aggregate", _shift),
        ProgramSpec("hist", 2, ("a", "b"), "a count, decoded from the BOS weight", _hist),
        ProgramSpec("two_hop", 2, (0, 1, 2), "two chained lookups: the composition case", _two_hop),
        ProgramSpec("reverse", 3, ("a", "b"), "a count feeding index arithmetic", _reverse),
        ProgramSpec("sort", 3, ("a", "b"), "rank by a pair, then gather", _sort),
        ProgramSpec("most_freq", 4, ("a", "b"), "count, then rank by the count", _most_freq),
        ProgramSpec("unique", 4, ("a", "b"), "two chained counts over a custom predicate", _unique),
        ProgramSpec("dyck1", 6, ("(", ")"), "two counts, a difference, and a running or", _dyck1),
        ProgramSpec("pair_balance", 2, ("(", ")"), "the numerical aggregate and read-out",
                    _pair_balance),
    )
}


def names() -> tuple[str, ...]:
    """Suite order: shallowest first, so a failure appears at the simplest program."""
    return tuple(sorted(PROGRAMS, key=lambda n: (PROGRAMS[n].depth, n)))


def build(name: str) -> tuple[SOp, tuple]:
    """``(program, vocab)`` for one entry of the suite."""
    try:
        spec = PROGRAMS[name]
    except KeyError:
        raise KeyError(f"no program named {name!r}; the suite is {names()}") from None
    return spec.build(), spec.vocab
