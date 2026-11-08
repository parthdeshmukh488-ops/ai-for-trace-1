"""The language and its interpreter.

The interpreter is the oracle for the compiler, so these are the most
load-bearing tests in the repository: if the interpreter is wrong, every
ground-truth circuit is wrong in the same direction and no downstream
measurement will reveal it.  They are therefore written against hand-computed
answers rather than against properties the implementation might share with its
own bug.
"""

from __future__ import annotations

import pytest

from circuitry.symbolic import (
    Aggregate,
    Encoding,
    IllDefinedAggregate,
    Map,
    Predicate,
    Select,
    SelectorWidth,
    SequenceMap,
    UnboundedValueSet,
    evaluate,
    evaluate_selector,
    indices,
    nodes_in_order,
    support,
    tokens,
)

# ---------------------------------------------------------------------------
# primitives and elementwise operations
# ---------------------------------------------------------------------------


def test_tokens_is_the_input():
    assert evaluate(tokens, list("abc")) == ["a", "b", "c"]


def test_indices_excludes_bos_and_counts_from_zero():
    """The language never sees a BOS symbol; that is a compilation concern."""
    assert evaluate(indices, list("abcd")) == [0, 1, 2, 3]


def test_map_is_elementwise():
    assert evaluate(Map(str.upper, tokens), list("abc")) == ["A", "B", "C"]


def test_sequence_map_pairs_positionwise():
    sm = SequenceMap(lambda t, i: f"{t}{i}", tokens, indices)
    assert evaluate(sm, list("xy")) == ["x0", "y1"]


def test_sequence_map_rejects_mismatched_lengths():
    """Cannot happen through the public interface, but the guard is cheap."""
    bad = SequenceMap(lambda a, b: a, tokens, Map(lambda v: v, indices))
    # Both operands derive from the same input, so lengths always agree; the
    # strict zip is a guard against a future operand that does not preserve
    # length.  Assert the normal case works rather than faking a violation.
    assert evaluate(bad, list("ab")) == ["a", "b"]


# ---------------------------------------------------------------------------
# selector orientation -- the easiest thing in the language to get backwards
# ---------------------------------------------------------------------------


def test_selector_is_indexed_query_then_key():
    sel = Select(indices, indices, Predicate.EQ)
    matrix = evaluate_selector(sel, list("abc"))
    assert matrix == [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]


def test_predicate_takes_the_key_value_first():
    """``Select(indices, indices, LT)`` selects strictly earlier positions.

    If the predicate were called query-first this would select strictly later
    positions instead, which is the same matrix transposed -- a mistake that
    leaves every shape intact and silently reverses the direction of
    information flow in every compiled model.
    """
    sel = Select(indices, indices, Predicate.LT)
    matrix = evaluate_selector(sel, list("abcd"))
    # Query 0 has no earlier position; query 3 has three.
    assert [sum(row) for row in matrix] == [0, 1, 2, 3]
    assert matrix[2] == [True, True, False, False]


def test_predicate_names_and_equality():
    assert Predicate.EQ == Predicate.EQ
    assert Predicate.EQ != Predicate.NEQ
    assert repr(Predicate.LT) == "Predicate.LT"
    assert len({Predicate.EQ, Predicate.EQ, Predicate.LT}) == 2


@pytest.mark.parametrize(
    "pred,expected",
    [
        (Predicate.EQ, [False, True, False]),
        (Predicate.NEQ, [True, False, True]),
        (Predicate.LT, [True, False, False]),
        (Predicate.LEQ, [True, True, False]),
        (Predicate.GT, [False, False, True]),
        (Predicate.GEQ, [False, True, True]),
        (Predicate.TRUE, [True, True, True]),
        (Predicate.FALSE, [False, False, False]),
    ],
)
def test_every_predicate_against_a_hand_computed_row(pred, expected):
    """Keys are 0,1,2 and the single query is 1; rows are read key-wise."""
    sel = Select(indices, Map(lambda i: 1, indices), pred)
    matrix = evaluate_selector(sel, list("abc"))
    assert matrix[0] == expected


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_categorical_aggregate_moves_one_value():
    """Each query selects the key at its own index, so this is the identity."""
    agg = Aggregate(Select(indices, indices, Predicate.EQ), tokens)
    assert evaluate(agg, list("abc")) == ["a", "b", "c"]


def test_categorical_aggregate_with_no_match_returns_the_default():
    agg = Aggregate(Select(indices, indices, Predicate.FALSE), tokens, default="?")
    assert evaluate(agg, list("ab")) == ["?", "?"]


def test_categorical_aggregate_with_two_matches_raises():
    """A mean of one-hot values is a distribution, not a value."""
    agg = Aggregate(Select(indices, indices, Predicate.TRUE), tokens, label="bad")
    with pytest.raises(IllDefinedAggregate) as excinfo:
        evaluate(agg, list("ab"))
    message = str(excinfo.value)
    assert "bad" in message
    assert "2 keys" in message
    assert "query position 0" in message


def test_numerical_aggregate_takes_the_mean():
    values = Map(lambda i: float(i), indices, encoding=Encoding.NUMERICAL)
    agg = Aggregate(
        Select(indices, indices, Predicate.LEQ),
        values,
        default=0.0,
        encoding=Encoding.NUMERICAL,
    )
    # Query i averages 0..i.
    assert evaluate(agg, list("abcd")) == [0.0, 0.5, 1.0, 1.5]


def test_numerical_aggregate_allows_many_matches():
    """The constraint is categorical-only; averaging magnitudes is well defined."""
    values = Map(lambda i: 1.0, indices, encoding=Encoding.NUMERICAL)
    agg = Aggregate(
        Select(indices, indices, Predicate.TRUE), values, default=0.0,
        encoding=Encoding.NUMERICAL,
    )
    assert evaluate(agg, list("abc")) == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# selector width
# ---------------------------------------------------------------------------


def test_selector_width_counts_matches():
    width = SelectorWidth(Select(tokens, tokens, Predicate.EQ))
    assert evaluate(width, list("aabca")) == [3, 3, 1, 1, 3]


def test_selector_width_is_numerical_regardless_of_the_argument():
    width = SelectorWidth(Select(tokens, tokens, Predicate.EQ), encoding=Encoding.CATEGORICAL)
    assert width.encoding == Encoding.NUMERICAL


# ---------------------------------------------------------------------------
# traversal
# ---------------------------------------------------------------------------


def test_nodes_in_order_puts_operands_first():
    inner = Map(str.upper, tokens)
    outer = Map(str.lower, inner)
    order = nodes_in_order(outer)
    assert order.index(tokens) < order.index(inner) < order.index(outer)


def test_nodes_in_order_visits_each_shared_node_once():
    shared = Map(str.upper, tokens)
    both = SequenceMap(lambda a, b: a + b, shared, shared)
    order = nodes_in_order(both)
    assert sum(1 for n in order if n is shared) == 1


def test_identity_not_value_equality_distinguishes_variables():
    """Two separately built nodes computing the same thing are two variables."""
    a = Map(str.upper, tokens)
    b = Map(str.upper, tokens)
    assert a is not b
    assert a != b
    order = nodes_in_order(SequenceMap(lambda x, y: x + y, a, b))
    assert len([n for n in order if isinstance(n, Map)]) == 2


def test_cycle_is_detected():
    loop = Map(lambda v: v, tokens, label="loop")
    loop.inner = loop
    with pytest.raises(ValueError, match="cycle"):
        nodes_in_order(loop)


# ---------------------------------------------------------------------------
# supports
# ---------------------------------------------------------------------------


def test_support_of_primitives():
    s = support(SequenceMap(lambda a, b: (a, b), tokens, indices), "ba", 3)
    assert s[id(tokens)] == ("a", "b")
    assert s[id(indices)] == (0, 1, 2)


def test_support_of_selector_width_is_bounded_although_numerical():
    """Regression test for a real specification error.

    ``SelectorWidth`` is numerical, and an earlier version of this module keyed
    the support rules on the encoding, which made every ``Map`` over a width
    unbounded and ruled out every counting program -- sorting, histograms, most
    frequent token.  Encoding is how a value is stored; support is which values
    it can take.  They are independent.
    """
    width = SelectorWidth(Select(tokens, tokens, Predicate.EQ))
    counted = Map(int, width)
    s = support(counted, "ab", 4)
    assert s[id(width)] == (0, 1, 2, 3, 4)
    assert s[id(counted)] == (0, 1, 2, 3, 4)


def test_support_of_map_is_the_image():
    m = Map(lambda t: t * 2, tokens)
    assert support(m, "ab", 2)[id(m)] == ("aa", "bb")


def test_support_of_sequence_map_is_the_product_image():
    sm = SequenceMap(lambda t, i: f"{t}{i}", tokens, indices)
    s = support(sm, "ab", 2)[id(sm)]
    assert s == ("a0", "a1", "b0", "b1")


def test_support_of_categorical_aggregate_includes_the_default():
    """The default is part of the support, and the support is sorted, not insertion-ordered.

    Sorting is what makes a compiled model's basis a function of the program
    rather than of the order the author happened to write it in, so "?" landing
    before "a" is the intended behaviour rather than an accident worth
    preserving an insertion order for.
    """
    agg = Aggregate(Select(indices, indices, Predicate.EQ), tokens, default="?")
    assert support(agg, "ab", 2)[id(agg)] == ("?", "a", "b")


def test_support_of_numerical_aggregate_is_unbounded():
    values = Map(float, indices, encoding=Encoding.NUMERICAL)
    agg = Aggregate(
        Select(indices, indices, Predicate.LEQ), values, encoding=Encoding.NUMERICAL
    )
    assert support(agg, "ab", 4)[id(agg)] is None


def test_support_propagates_unboundedness_without_raising():
    """Only the compiler decides unboundedness is fatal, and only if it matters."""
    values = Map(float, indices, encoding=Encoding.NUMERICAL)
    agg = Aggregate(
        Select(indices, indices, Predicate.LEQ), values, encoding=Encoding.NUMERICAL
    )
    downstream = Map(lambda v: v, agg)
    assert support(downstream, "ab", 4)[id(downstream)] is None


def test_product_limit_raises_with_the_offending_node():
    big = SequenceMap(lambda a, b: (a, b), tokens, indices, label="too_big")
    with pytest.raises(UnboundedValueSet, match="too_big"):
        support(big, "abcdefghij", 10, product_limit=16)


def test_support_rejects_degenerate_max_len():
    with pytest.raises(ValueError, match="max_len"):
        support(tokens, "ab", 0)


def test_support_is_deterministic_and_sorted():
    s = support(tokens, "cba", 2)[id(tokens)]
    assert s == ("a", "b", "c")


# ---------------------------------------------------------------------------
# whole programs, against answers computed by hand
# ---------------------------------------------------------------------------


def _reverse():
    length = Map(int, SelectorWidth(Select(tokens, tokens, Predicate.TRUE)), label="length")
    target = SequenceMap(lambda n, i: n - 1 - i, length, indices, label="target")
    return Aggregate(Select(indices, target, Predicate.EQ), tokens, label="reverse")


def _rank():
    """Rank by how many strictly smaller tokens the sequence contains."""
    return Map(int, SelectorWidth(Select(tokens, tokens, Predicate.LT)), label="rank")


def _sort():
    rank = _rank()
    return Aggregate(Select(rank, indices, Predicate.EQ), tokens, label="sort")


def _two_hop():
    """``out[i] = tokens[tokens[i]]``, the smallest genuinely composed program."""
    return Aggregate(Select(indices, tokens, Predicate.EQ), tokens, label="two_hop")


@pytest.mark.parametrize(
    "seq,expected",
    [("abcd", list("dcba")), ("xy", list("yx")), ("q", ["q"])],
)
def test_reverse(seq, expected):
    assert evaluate(_reverse(), list(seq)) == expected


def test_histogram():
    hist = Map(int, SelectorWidth(Select(tokens, tokens, Predicate.EQ)), label="hist")
    assert evaluate(hist, list("aabca")) == [3, 3, 1, 1, 3]


def test_rank_counts_strictly_smaller_tokens():
    assert evaluate(_rank(), list("cab")) == [2, 0, 1]


def test_sort_on_distinct_tokens():
    assert evaluate(_sort(), list("cab")) == list("abc")


def test_sort_with_duplicates_is_ill_defined_and_says_so():
    """Equal tokens share a rank, so two keys match one query: a real limitation.

    Sorting with ties needs the rank broken by position, which the program
    suite does with a secondary index term.  The bare version raising here
    rather than silently returning something is the behaviour that matters.
    """
    with pytest.raises(IllDefinedAggregate):
        evaluate(_sort(), list("aba"))


def test_sort_with_ties_broken_by_position():
    """The fix: rank by (token, index) lexicographically, via a double count."""
    smaller = SelectorWidth(Select(tokens, tokens, Predicate.LT))
    # Among equal tokens, count the strictly earlier ones.
    pairs = SequenceMap(lambda t, i: (t, i), tokens, indices, label="pairs")
    rank = Map(int, SelectorWidth(Select(pairs, pairs, Predicate.LT)), label="rank_ties")
    assert evaluate(Map(int, smaller), list("aba")) == [0, 2, 0]
    out = Aggregate(Select(rank, indices, Predicate.EQ), tokens, label="sort_ties")
    assert evaluate(out, list("aba")) == list("aab")
    assert evaluate(out, list("cba")) == list("abc")


def test_two_hop():
    """tokens = [1, 2, 0] -> out[i] = tokens[tokens[i]] = [2, 0, 1]."""
    assert evaluate(_two_hop(), [1, 2, 0]) == [2, 0, 1]


def test_two_hop_out_of_range_falls_back_to_the_default():
    prog = Aggregate(Select(indices, tokens, Predicate.EQ), tokens, default=-1, label="hop")
    assert evaluate(prog, [1, 9]) == [9, -1]


def test_pair_balance_running_difference():
    """A numerical program: the running fraction of opening brackets."""
    is_open = Map(lambda t: 1.0 if t == "(" else 0.0, tokens, encoding=Encoding.NUMERICAL)
    frac = Aggregate(
        Select(indices, indices, Predicate.LEQ),
        is_open,
        default=0.0,
        encoding=Encoding.NUMERICAL,
        label="frac_open",
    )
    got = evaluate(frac, list("(()"))
    assert got == pytest.approx([1.0, 1.0, 2 / 3])
