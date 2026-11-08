"""The compiler.

Verification here is **exhaustive, not sampled**.  For sequences up to length
four over a two-symbol vocabulary there are thirty inputs per program; running
all of them costs milliseconds and is the difference between a compiler that
is right and one that is right on the inputs somebody happened to try.  The
second kind is the most dangerous component this project could contain,
because the ground truth would be wrong rather than the model, and nothing
downstream would reveal it.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from circuitry.model.hooks import capture, intervene, replace_at
from circuitry.symbolic import programs
from circuitry.symbolic.compiler import CompileError, compile_program
from circuitry.symbolic.lang import (
    Aggregate,
    IllDefinedAggregate,
    Map,
    Predicate,
    Select,
    SelectorWidth,
    SequenceMap,
    evaluate,
    indices,
    tokens,
)

MAX_LEN = 4
ALL = programs.names()


def _inputs(vocab, max_len=MAX_LEN):
    for length in range(1, max_len + 1):
        yield from (list(s) for s in itertools.product(vocab, repeat=length))


# ---------------------------------------------------------------------------
# the central claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_compiled_matches_interpreter_exhaustively(name):
    """Every input, every program, exact equality with the reference interpreter."""
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    numerical = compiled.root.encoding == "numerical"
    for sequence in _inputs(vocab):
        expected = evaluate(program, sequence)
        got = compiled.run(sequence)
        if numerical:
            assert got == pytest.approx(expected, abs=1e-6), sequence
        else:
            assert got == expected, sequence


@pytest.mark.parametrize("name", ALL)
def test_verification_is_on_by_default(name):
    """compile_program runs the exhaustive check itself unless told not to."""
    program, vocab = programs.build(name)
    compile_program(program, vocab, 3)  # would raise on any disagreement


def test_verification_catches_a_deliberately_broken_model():
    """The check has to be able to fail, or it is decoration."""
    program, vocab = programs.build("shift")
    compiled = compile_program(program, vocab, 3, verify=False)
    with torch.no_grad():
        compiled.model.blocks[0].mlp.W_out.zero_()
    from circuitry.symbolic.compiler import _verify_exhaustively

    with pytest.raises(CompileError, match="disagrees with the interpreter"):
        _verify_exhaustively(compiled, compiled.root, vocab, 3)


# ---------------------------------------------------------------------------
# the layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_basis_blocks_are_disjoint_and_cover_the_width(name):
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    layout = compiled.layout
    claimed = [False] * layout.width
    claimed[layout.bos_dim] = True
    for space in layout.all_spaces():
        for dim in range(space.start, space.stop):
            assert not claimed[dim], f"{space.name} overlaps at dimension {dim}"
            claimed[dim] = True
    assert all(claimed), "the layout leaves dimensions unused"


@pytest.mark.parametrize("name", ALL)
def test_every_variable_decodes_to_the_interpreters_value(name):
    """Not just the output: every intermediate must be what the program says.

    A compiled model could in principle get the right answer through a
    different internal route, which would make the ground-truth graph a
    fiction.  Reading each block back and comparing it to the corresponding
    interpreter value rules that out.
    """
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    from circuitry.symbolic.lang import Selector, nodes_in_order

    nodes = [n for n in nodes_in_order(program) if not isinstance(n, Selector)]
    for sequence in _inputs(vocab, 3):
        traced = compiled.trace(sequence)
        for node in nodes:
            space = compiled.layout.of(node)
            if space.name not in traced:
                continue
            expected = evaluate(node, sequence)
            got = traced[space.name]
            if space.values is None:
                assert got == pytest.approx(
                    [float(v) for v in expected], abs=1e-6
                ), (name, space.name, sequence)
            else:
                assert got == list(expected), (name, space.name, sequence)


def test_block_width_is_the_support_size():
    program, vocab = programs.build("hist")
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    space = compiled.layout.of(program)
    assert space.width == MAX_LEN + 1  # a count takes values 0..max_len
    assert space.values == tuple(range(MAX_LEN + 1))


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


def test_ill_defined_aggregate_raises_over_the_whole_input_space():
    """Ranking by token alone ties on a repeat, which has no categorical answer."""
    rank = SelectorWidth(Select(tokens, tokens, Predicate.LT), label="naive_rank")
    program = Aggregate(Select(rank, indices, Predicate.EQ), tokens, label="bad_sort")
    with pytest.raises(CompileError, match="ill defined on input"):
        compile_program(program, ("a", "b"), 3)


def test_the_interpreter_agrees_that_it_is_ill_defined():
    rank = SelectorWidth(Select(tokens, tokens, Predicate.LT), label="naive_rank")
    program = Aggregate(Select(rank, indices, Predicate.EQ), tokens, label="bad_sort")
    with pytest.raises(IllDefinedAggregate):
        evaluate(program, ["a", "a"])


def test_unbounded_numerical_map_is_rejected_with_a_remedy():
    from circuitry.symbolic.lang import Encoding

    step = Map(lambda t: 1.0, tokens, encoding=Encoding.NUMERICAL, label="step")
    mean = Aggregate(
        Select(indices, indices, Predicate.LEQ), step, default=0.0,
        encoding=Encoding.NUMERICAL, label="mean",
    )
    program = Map(lambda m: m > 0, mean, label="positive")
    with pytest.raises(CompileError, match="declared_support"):
        compile_program(program, ("a", "b"), 3, verify=False)


def test_a_wrong_declared_support_is_caught():
    """A declaration is a claim, and the compiler checks it."""
    program = Map(lambda t: t * 2, tokens, declared_support=("aa",), label="doubled")
    with pytest.raises(CompileError, match="outside its own declared support"):
        compile_program(program, ("a", "b"), 2, verify=False)


def test_sequence_map_over_a_magnitude_is_rejected_with_a_remedy():
    from circuitry.symbolic.lang import Encoding

    step = Map(lambda t: 1.0, tokens, encoding=Encoding.NUMERICAL, label="step")
    program = SequenceMap(lambda a, b: a, step, tokens, label="mixed")
    with pytest.raises(CompileError, match="decode the magnitude"):
        compile_program(program, ("a", "b"), 2, verify=False)


def test_empty_vocabulary_and_bad_length_are_rejected():
    program, _ = programs.build("hist")
    with pytest.raises(CompileError, match="vocabulary is empty"):
        compile_program(program, (), 3)
    with pytest.raises(CompileError, match="max_len"):
        compile_program(program, ("a",), 0)


def test_oversized_verification_refuses_rather_than_hanging():
    program, _ = programs.build("hist")
    with pytest.raises(CompileError, match="exhaustive verification"):
        compile_program(program, tuple("abcdefghij"), 6)


# ---------------------------------------------------------------------------
# the beginning-of-sequence position
# ---------------------------------------------------------------------------


def test_bos_absorbs_an_empty_selection():
    """With nothing selected the head must read the default, not an average.

    Without a BOS position every score is equal, softmax spreads attention
    over the whole sequence, and the aggregate is a meaningless mean.  ``shift``
    at position 0 has nothing to look back at, so it exercises exactly this.
    """
    program, vocab = programs.build("shift")
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    assert compiled.run(["a", "b"])[0] == "_"
    assert compiled.run(["b", "a"])[0] == "_"


def _aggregate_pattern(compiled, sequence, variable):
    """The attention pattern of the head implementing ``variable``.

    Found from the ground-truth graph rather than assumed to be layer 0.  The
    scheduler places a head at the earliest layer its operands allow, and
    ``shift``'s aggregate reads a variable an MLP produces, so it sits at layer
    1 with layer 0's head unused -- which is exactly the kind of thing the
    graph exists to record.
    """
    site = next(
        node.site.replace(".attn.z", ".attn.pattern")
        for node, name in compiled.graph.implements
        if name == variable and ".attn.z" in node.site
    )
    head = next(
        dict(node.index)["head"]
        for node, name in compiled.graph.implements
        if name == variable and ".attn.z" in node.site
    )
    with capture(compiled.model, site) as store, torch.inference_mode():
        compiled.model(compiled.encode(sequence))
    return store[site][0, head]


def test_bos_weight_vanishes_when_something_is_selected():
    """It is below any match by design, so it must not contaminate the value."""
    program, vocab = programs.build("shift")
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    pattern = _aggregate_pattern(compiled, ["a", "b", "a"], "shift")
    # Query at position 2 looks back to position 1; BOS is column 0.  The
    # tolerance is 1e-5 rather than 1e-10 because BOS sits at half score, so
    # its residual weight is O(exp(-lambda/2)) -- about 3e-7 at the default
    # scale, not 1e-13.  See the compiler's module docstring.
    assert float(pattern[2, 0]) < 1e-5
    assert float(pattern[2, 1]) > 1 - 1e-5


def test_bos_dominates_when_nothing_is_selected():
    program, vocab = programs.build("shift")
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    pattern = _aggregate_pattern(compiled, ["a", "b"], "shift")
    assert float(pattern[1, 0]) > 1 - 1e-5  # query 1 has no predecessor
    assert float(pattern[1, 1:].abs().max()) < 1e-5


def test_the_empty_selection_path_converges_at_half_the_rate():
    """Measured, because it sets the accuracy of the whole construction.

    A match beats a non-match by lambda; the beginning-of-sequence position
    beats one by lambda/2.  So the empty-selection error is the square root of
    the ordinary one, and it is the first thing to fail as the scale drops.
    """
    program, vocab = programs.build("shift")
    errors = {}
    for scale in (8.0, 16.0, 32.0):
        compiled = compile_program(program, vocab, 3, logit_scale=scale, verify=False)
        pattern = _aggregate_pattern(compiled, ["a", "b"], "shift")
        errors[scale] = 1.0 - float(pattern[1, 0])
    assert errors[8.0] > errors[16.0] > errors[32.0]
    # Halving the exponent means each doubling of the scale squares the error.
    assert errors[16.0] == pytest.approx(errors[8.0] ** 2, rel=0.6)


# ---------------------------------------------------------------------------
# the ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_unused_components_have_no_influence(name):
    """The true-negative set must really be causally inert.

    All of them are ablated at once rather than one at a time: an individually
    inert head could still matter in combination, and precision is only well
    defined if the whole set is genuinely dead.
    """
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    if not compiled.graph.unused:
        pytest.skip("this program uses every head")
    sequence = list(vocab) * 2
    sequence = sequence[:3]
    tokens_in = compiled.encode(sequence)
    with torch.inference_mode():
        before = compiled.model(tokens_in)
    hooks = {}
    for node in compiled.graph.unused:
        head = dict(node.index)["head"]
        hooks.setdefault(node.site, []).append(replace_at(node.site, {"head": head}, 0.0))
    with intervene(compiled.model, hooks), torch.inference_mode():
        after = compiled.model(tokens_in)
    assert torch.allclose(before, after, atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", ALL)
def test_ground_truth_edges_match_program_dependencies(name):
    """Every edge in the emitted graph corresponds to a real data dependency."""
    from circuitry.symbolic.compiler import _dependencies
    from circuitry.symbolic.lang import Selector, nodes_in_order

    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    computed = [
        n
        for n in nodes_in_order(program)
        if not isinstance(n, Selector) and id(n) in compiled.schedule
    ]
    # Every dependency between two computed variables must appear as an edge
    # between the components that compute them.
    node_for = {}
    for gt, variable in compiled.graph.implements:
        node_for.setdefault(variable, gt)
    edges = compiled.graph.edge_set()
    for node in computed:
        dst = node_for[compiled.layout.of(node).name]
        for dep in _dependencies(node):
            if id(dep) not in compiled.schedule:
                continue
            src = node_for[compiled.layout.of(dep).name]
            if src != dst:
                assert (src, dst) in edges, f"{src} -> {dst} missing for {name}"


@pytest.mark.parametrize("name", ALL)
def test_graph_is_internally_consistent(name):
    program, vocab = programs.build(name)
    graph = compile_program(program, vocab, MAX_LEN, verify=False).graph
    assert graph.n_nodes >= 2
    assert set(graph.unused).isdisjoint(graph.nodes)
    assert graph.describe()


def test_graph_rejects_an_edge_outside_its_node_set():
    from circuitry.symbolic.truth import GroundTruthGraph, Node

    a, b = Node("embed"), Node("blocks.0.mlp.out")
    with pytest.raises(AssertionError, match="outside the graph"):
        GroundTruthGraph((a,), ((a, b),), (), (), "x", ())


# ---------------------------------------------------------------------------
# the model's own shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_compiled_models_are_not_causal(name):
    """The language lets a position read any other, so the mask has to go.

    ``reverse`` asks position 1 for the contents of position n-2; with a causal
    mask it returned its default wherever the lookup pointed forwards.
    """
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    assert compiled.cfg.causal is False
    assert compiled.cfg.use_ln is False
    assert compiled.cfg.act_fn == "relu"
    assert compiled.cfg.dtype == "float64"


def test_encode_rejects_a_symbol_outside_the_vocabulary():
    program, vocab = programs.build("hist")
    compiled = compile_program(program, vocab, MAX_LEN, verify=False)
    with pytest.raises(KeyError, match="not in the compiled vocabulary"):
        compiled.encode(["z"])


def test_encode_rejects_an_overlong_sequence():
    program, vocab = programs.build("hist")
    compiled = compile_program(program, vocab, 3, verify=False)
    with pytest.raises(ValueError, match="exceeds the compiled maximum"):
        compiled.encode(["a"] * 4)


def test_decoding_a_numerical_root_raises_with_the_alternative():
    program, vocab = programs.build("pair_balance")
    compiled = compile_program(program, vocab, 3, verify=False)
    with pytest.raises(TypeError, match="read_numerical"):
        compiled.decode(torch.zeros(1, 2, 3))
