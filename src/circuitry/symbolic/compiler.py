"""Turning a program into exact transformer weights.

Every construct in the language has one hardware realisation, and the
realisations are chosen so the resulting network computes the program
*exactly* rather than approximately.  Three decisions make that possible, and
each one is a constraint on the compiled model's configuration rather than a
free choice:

* **No LayerNorm.**  The MLP lookup tables below read one-hot patterns
  straight out of the residual stream.  A LayerNorm in front of them would
  centre and rescale by the stream's RMS, which depends on how many variables
  happen to be active, and the reads would no longer be exact.  Compiled
  models therefore set ``use_ln=False``.
* **ReLU, not GELU.**  A lookup-table neuron is supposed to be off for every
  input but one.  GELU is not zero for negative inputs, so "off" would leak a
  small amount from every non-matching entry, and the leak would grow with the
  size of the domain.  ReLU is exactly zero, so the table is exact.
* **float64.**  The attention logit scale needs to be large enough that
  softmax is effectively hard, and the measured headroom before float32
  overflows is small (see ``experiments/compiler_fidelity.py``).
* **No causal mask.**  The language lets a position read any other position,
  and ``reverse`` asks position 1 for the contents of position ``n-2``.  Those
  programs cannot be realised with a causal mask at all -- the first attempt
  at this compiler produced a ``two_hop`` model that was right wherever the
  lookup happened to point backwards and returned the default wherever it
  pointed forwards.  Compiled models therefore set ``causal=False``.  This is
  a real difference between this regime and the trained decoder-only models of
  phase 8, and it is in the caveats rather than glossed over.

The constructions
-----------------

**``Select`` + ``Aggregate`` -> one attention head.**  Let the key variable
have support ``K`` and the query variable support ``Q``.  Keys are read as
one-hot vectors, and the query projection is set to ``c * P`` where ``P`` is
the predicate evaluated over ``Q x K``, so that

    q_i . k_j = c * P[query_value_i, key_value_j]

and after the ``1/sqrt(d_head)`` the score is ``lambda`` where the predicate
holds and ``0`` where it does not.  Values are read one-hot from the
aggregated variable's block and written to the output block, so the head moves
a value rather than averaging one.

**The beginning-of-sequence position, at half score.**  A selector that
matches nothing is the awkward case: with every score equal, softmax spreads
attention over the whole sequence and the aggregate is a meaningless average
rather than the program's default.  So a BOS position is prepended, its key is
a dedicated dimension, and the query gives it score ``lambda/2`` -- below any
match and above any non-match.  When something matches, BOS's weight is
``O(e^(-lambda/2))`` and vanishes; when nothing does, BOS dominates and the
head reads the default out of it.  One position, and the empty case stops
being special.

The half score has a cost worth stating, because it sets the accuracy of the
whole construction.  A *match* beats a non-match by ``lambda``, so the error
on an ordinary aggregate is ``O(m e^(-lambda))``.  But BOS beats a non-match
by only ``lambda/2``, so the error on the empty case is
``O(m e^(-lambda/2))`` -- the square root of the other.  At the default scale
of 30 that is about ``6e-7`` rather than ``1e-13``: still far inside any
threshold the decoders use, but it means **the empty-selection path converges
at half the rate of everything else** and is the first thing to fail as the
scale is lowered.  ``experiments/compiler_fidelity.py`` measures exactly that.

**``Map`` and ``SequenceMap`` -> an MLP lookup table.**  One neuron per
element of the domain, with a bias that makes its pre-activation positive for
exactly that element: ``-0.5`` against a single one-hot (``1 - 0.5`` against
``0 - 0.5``), and ``-1.5`` against a pair (``2 - 1.5`` against at most
``1 - 1.5``).  The firing value is then divided out of the output weights --
measured from the configured activation function rather than assumed, so that
changing it cannot silently break the compiler.

**``SelectorWidth`` -> a head and an MLP.**  Attend to the selected keys *and*
to BOS at equal score; BOS's weight is then ``1/(n+1)`` where ``n`` is the
number of matches.  Read that weight into a scratch dimension, and decode it
with the step decoder below.

**Decoding a magnitude.**  A count arrives in the residual stream as one
number, and everything downstream of it -- a lookup table, a selector key --
wants a one-hot.  The general tool is a *step decoder*: given a variable whose
achievable values are known and finite, build one unit step per value at the
midpoint between consecutive values, and sum them telescopically so that the
output at ``v_j`` is exactly ``f(v_j)``.  Each step is two ReLU neurons whose
separation is half the smallest gap in the support, which makes the step
exactly ``0`` or ``1`` on the values that actually occur and never in between.
Two neurons per value, and it serves both the ``SelectorWidth`` inversion and
any ``Map`` over a magnitude whose support is bounded.

Scheduling
----------

Attention runs before the MLP within a block, so a variable computed by a head
at layer ``L`` is readable by the MLP at layer ``L``, and one computed by that
MLP is readable from layer ``L+1``.  Nodes are placed at the earliest slot
their operands allow, several to a layer where they fit -- different heads for
different aggregates, disjoint neuron ranges for different maps -- and the
model's depth is whatever the program's dependency chain requires.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch

from ..model.config import ModelConfig
from ..model.transformer import Transformer, act_fn
from .bases import BasisAllocator, Layout, VariableSpace
from .lang import (
    Aggregate,
    CompileError,
    Encoding,
    IllDefinedAggregate,
    Map,
    Select,
    Selector,
    SelectorWidth,
    SequenceMap,
    SOp,
    _Indices,
    _Tokens,
    evaluate,
    nodes_in_order,
    support,
)
from .truth import GroundTruthGraph, Node

__all__ = [
    "DEFAULT_LOGIT_SCALE",
    "CompileError",
    "CompiledModel",
    "compile_program",
]

#: The attention logit scale, chosen from the sweep in
#: ``experiments/compiler_fidelity.py`` rather than guessed.
#:
#: Measured there: the worst error over all inputs falls by a factor of
#: ``e^4`` for every 8 units of scale -- that is, as ``e^(-lambda/2)``, the
#: half rate the beginning-of-sequence construction predicts -- holding over
#: three decades.  Eight of the nine programs in the suite are exactly right
#: from a scale of 4, because a one-hot argmax is unbothered by a perturbation
#: of ``1e-7``.  The binding constraint is ``pair_balance``, the one program
#: whose output is a magnitude read back directly: its tolerance is ``1e-6``,
#: so it needs ``lambda > 2 ln(1e6) = 27.6``, and it is measured to fail at 24
#: and succeed at 32.
#:
#: 40 is that threshold with margin, giving a predicted error around
#: ``2e-9``.  There is no upper bound to respect: float32 does not overflow
#: here at any scale tested up to 100, because softmax subtracts the row
#: maximum before exponentiating.
DEFAULT_LOGIT_SCALE = 40.0

#: How much larger the correct output logit is than the rest.  Only the
#: ordering matters for decoding, but a metric defined on logit *differences*
#: wants a scale that is not within rounding of zero.
UNEMBED_SCALE = 10.0

_ATTN, _MLP = 0, 1


@dataclass
class CompiledModel:
    """A transformer that provably computes its source program.

    ``run`` is the convenience path: it encodes a sequence, runs the model,
    and decodes the output values, so a caller can compare against
    :func:`circuitry.symbolic.evaluate` directly.
    """

    model: Transformer
    layout: Layout
    graph: GroundTruthGraph
    root: SOp
    vocab: tuple
    bos_id: int
    output_values: tuple
    logit_scale: float
    schedule: dict = field(default_factory=dict)

    @property
    def cfg(self) -> ModelConfig:
        return self.model.cfg

    def encode(self, sequence) -> torch.Tensor:
        """``[1, len+1]`` token ids, with the beginning-of-sequence symbol first."""
        ids = [self.bos_id]
        for symbol in sequence:
            try:
                ids.append(self.vocab.index(symbol))
            except ValueError:
                raise KeyError(
                    f"{symbol!r} is not in the compiled vocabulary {self.vocab!r}"
                ) from None
        if len(ids) > self.cfg.n_ctx:
            raise ValueError(
                f"sequence of length {len(sequence)} exceeds the compiled maximum of "
                f"{self.cfg.n_ctx - 1}"
            )
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, logits: torch.Tensor) -> list:
        """Output values for the real positions, dropping the BOS position."""
        if self.root.encoding == Encoding.NUMERICAL:
            raise TypeError(
                f"the program's root {self.root.label!r} is numerical; read it with "
                "read_numerical rather than by decoding logits"
            )
        picks = logits[0, 1:, : len(self.output_values)].argmax(dim=-1).tolist()
        return [self.output_values[i] for i in picks]

    def read_numerical(self, sequence) -> list[float]:
        """The root's value per position, read out of the residual stream."""
        from ..model.hooks import capture

        space = self.layout.of(self.root)
        site = f"blocks.{self.cfg.n_layers - 1}.resid_post"
        with capture(self.model, site) as store, torch.inference_mode():
            self.model(self.encode(sequence))
        return store[site][0, 1:, space.start].tolist()

    def trace(self, sequence) -> dict[str, list]:
        """Every variable's value at every position, decoded out of the stream.

        The debugging tool this compiler could not have been written without.
        When a compiled program disagrees with its source, the question is
        always *which variable first went wrong*, and reading them all out of
        the final residual stream answers it in one run -- the blocks are
        disjoint and nothing is ever overwritten, so the last layer still
        carries every intermediate.
        """
        from ..model.hooks import capture

        site = f"blocks.{self.cfg.n_layers - 1}.resid_post"
        with capture(self.model, site) as store, torch.inference_mode():
            self.model(self.encode(sequence))
        resid = store[site][0]
        out: dict[str, list] = {}
        for space in self.layout.all_spaces():
            block = resid[1:, space.start : space.stop]
            if space.values is None:
                out[space.name] = [round(float(v), 6) for v in block[:, 0]]
            else:
                out[space.name] = [space.values[int(row.argmax())] for row in block]
        return out

    def run(self, sequence) -> list:
        """Encode, run, decode.  The thing to compare against the interpreter."""
        if self.root.encoding == Encoding.NUMERICAL:
            return self.read_numerical(sequence)
        with torch.inference_mode():
            return self.decode(self.model(self.encode(sequence)))


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------


def _is_attention(node: SOp) -> bool:
    return isinstance(node, (Aggregate, SelectorWidth))


def _is_mlp(node: SOp) -> bool:
    return isinstance(node, (Map, SequenceMap))


def _dependencies(node: SOp) -> list[SOp]:
    """The variables a node reads, with selectors flattened into their operands."""
    return [dep for dep, _port in _dependency_ports(node)]


def _dependency_ports(node: SOp) -> list[tuple[SOp, str]]:
    """Each variable a node reads, and *which input* it is read through.

    The compiler knows this exactly and nothing downstream can recover it: a
    head reading a variable as its query is a different edge from one reading
    the same variable as its value, and that distinction is most of what a
    circuit is.  ``Select(keys, queries, pred)`` supplies the key and query
    projections; ``Aggregate``'s own operand is what the head moves, so it is
    the value.
    """
    if isinstance(node, Aggregate):
        selector = node.selector
        return [
            (selector.keys, "k"),
            (selector.queries, "q"),
            (node.inner, "v"),
        ]
    if isinstance(node, SelectorWidth):
        selector = node.selector
        return [(selector.keys, "k"), (selector.queries, "q")]
    if isinstance(node, Map):
        return [(node.inner, "mlp_in")]
    if isinstance(node, SequenceMap):
        return [(node.lhs, "mlp_in"), (node.rhs, "mlp_in")]
    return []


def _schedule(order: list[SOp]) -> dict[int, tuple[int, int]]:
    """Place every computed node at the earliest slot its operands allow."""
    placed: dict[int, tuple[int, int]] = {}

    def earliest_reader(dep: SOp) -> tuple[int, int]:
        """The first slot at which ``dep``'s value is present in the stream.

        A head at layer ``L`` writes before that layer's MLP reads, so its
        output is readable at ``(L, mlp)``.  An MLP's output is readable from
        ``L+1``.

        ``SelectorWidth`` is the exception and the reason this function has a
        docstring: it occupies *both* slots of its layer -- a head to get the
        beginning-of-sequence weight, and that layer's MLP to decode it into a
        count -- so its value is written by the MLP and is not available until
        ``L+1``.  Treating it as a plain head put every ``Map`` over a count
        into the same MLP that produces the count, where it read a block that
        was still zero.  ``reverse`` compiled to a model that returned its
        default everywhere, and ``sort`` only worked because its consumer
        happened to be an aggregate, which was bumped to the next layer for an
        unrelated reason.
        """
        if isinstance(dep, (_Tokens, _Indices)):
            return (0, _ATTN)
        layer, kind = placed[id(dep)]
        if isinstance(dep, SelectorWidth):
            return (layer + 1, _ATTN)
        return (layer, _MLP) if kind == _ATTN else (layer + 1, _ATTN)

    for node in order:
        if isinstance(node, (_Tokens, _Indices)):
            continue
        deps = _dependencies(node)
        need = max((earliest_reader(d) for d in deps), default=(0, _ATTN))
        layer, kind = need
        if _is_attention(node):
            if kind == _MLP:
                layer, kind = layer + 1, _ATTN
        elif _is_mlp(node):
            kind = _MLP
        else:  # pragma: no cover - guards a future construct
            raise CompileError(f"no hardware realisation for {type(node).__name__}")
        placed[id(node)] = (layer, kind)
    return placed


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------


def _predicate_matrix(selector: Select, layout: Layout) -> torch.Tensor:
    keys = layout.of(selector.keys)
    queries = layout.of(selector.queries)
    if keys.values is None or queries.values is None:
        raise CompileError(
            f"selector {selector.label!r} compares {selector.keys.label!r} with "
            f"{selector.queries.label!r}, and a selector needs both sides categorical "
            "with a bounded support so the predicate can be written as a matrix"
        )
    matrix = torch.zeros(len(queries.values), len(keys.values), dtype=torch.float64)
    for qi, qv in enumerate(queries.values):
        for ki, kv in enumerate(keys.values):
            matrix[qi, ki] = 1.0 if selector.predicate(kv, qv) else 0.0
    return matrix


def _head_widths(node: SOp, layout: Layout) -> int:
    """How many head dimensions this construct needs."""
    selector = node.selector
    keys = layout.of(selector.keys)
    need_qk = keys.width + 1  # the key one-hot plus the BOS dimension
    if isinstance(node, SelectorWidth):
        need_v = 1
    else:
        need_v = layout.of(node).width
    return max(need_qk, need_v)


def _mlp_neurons(node: SOp, layout: Layout, max_len: int, supports) -> int:
    if isinstance(node, Map):
        inner = layout.of(node.inner)
        if inner.values is None:
            bounded = supports.get(id(node.inner))
            if bounded is None:
                raise CompileError(
                    f"Map {node.label!r} reads the unbounded numerical variable "
                    f"{node.inner.label!r}; bound it with a declared_support"
                )
            return 2 * len(bounded)  # two ReLU neurons per step
        return inner.width
    if isinstance(node, SequenceMap):
        lhs, rhs = layout.of(node.lhs), layout.of(node.rhs)
        if lhs.values is None or rhs.values is None:
            raise CompileError(
                f"SequenceMap {node.label!r} reads a numerical operand. A two-input lookup "
                "table needs both sides one-hot; decode the magnitude with a Map first."
            )
        return lhs.width * rhs.width
    if isinstance(node, SelectorWidth):
        return 2 * (max_len + 1)  # two ReLU neurons per achievable count
    raise AssertionError(type(node).__name__)  # pragma: no cover


# ---------------------------------------------------------------------------
# the compiler
# ---------------------------------------------------------------------------


def compile_program(
    root: SOp,
    vocab,
    max_len: int,
    *,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
    product_limit: int = 4096,
    verify: bool = True,
) -> CompiledModel:
    """Compile ``root`` into a transformer that computes it exactly.

    ``verify`` checks the result against the reference interpreter over the
    entire input space when that is small enough to enumerate, and raises on
    any disagreement.  It defaults on: a compiler that is right on the inputs
    somebody happened to try is the single most dangerous component this
    project could contain, because the ground truth would be wrong rather than
    the model, and nothing downstream would reveal it.
    """
    vocab = tuple(dict.fromkeys(vocab))
    if not vocab:
        raise CompileError("the vocabulary is empty")
    if max_len < 1:
        raise CompileError(f"max_len must be at least 1, got {max_len}")

    supports = support(root, vocab, max_len, product_limit=product_limit)
    order = [n for n in nodes_in_order(root) if not isinstance(n, Selector)]
    _check_well_defined(root, vocab, max_len)

    # --- allocate ------------------------------------------------------
    alloc = BasisAllocator()
    for node in order:
        alloc.add(node, supports.get(id(node)))
    scratch: dict[int, VariableSpace] = {}
    for node in order:
        if isinstance(node, SelectorWidth):
            scratch[id(node)] = alloc.add_scratch(f"{node.label}.bos_weight", 1)
    layout = alloc.layout

    # --- schedule and size --------------------------------------------
    placed = _schedule(order)
    computed = [n for n in order if id(n) in placed]
    n_layers = max((layer for layer, _ in placed.values()), default=0) + 1

    heads_per_layer: dict[int, list[SOp]] = {i: [] for i in range(n_layers)}
    mlps_per_layer: dict[int, list[SOp]] = {i: [] for i in range(n_layers)}
    for node in computed:
        layer, kind = placed[id(node)]
        if kind == _ATTN:
            heads_per_layer[layer].append(node)
            if isinstance(node, SelectorWidth):
                mlps_per_layer[layer].append(node)
        else:
            mlps_per_layer[layer].append(node)

    n_heads = max((len(v) for v in heads_per_layer.values()), default=0) or 1
    d_head = max((_head_widths(n, layout) for n in computed if _is_attention(n)), default=1)
    d_mlp = max(
        (sum(_mlp_neurons(n, layout, max_len, supports) for n in nodes)
         for nodes in mlps_per_layer.values()),
        default=1,
    ) or 1

    root_support = supports.get(id(root))
    output_values = tuple(root_support) if root_support is not None else ()
    d_vocab = max(len(vocab) + 1, len(output_values), 1)

    cfg = ModelConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=layout.width,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        n_ctx=max_len + 1,
        d_head=d_head,
        act_fn="relu",
        use_ln=False,
        positional="learned",
        tie_unembed=False,
        # The language is not causal; see the module docstring.
        causal=False,
        dtype="float64",
    )
    model = Transformer(cfg)
    bos_id = len(vocab)

    # --- weights -------------------------------------------------------
    with torch.no_grad():
        _build_embeddings(model, layout, vocab, bos_id, max_len)
        head_slots: dict[int, int] = {}
        for layer in range(n_layers):
            for slot, node in enumerate(heads_per_layer[layer]):
                head_slots[id(node)] = slot
                _build_head(model, layout, scratch, node, layer, slot, logit_scale, cfg)
            offset = 0
            for node in mlps_per_layer[layer]:
                offset = _build_mlp(
                    model, layout, scratch, node, layer, offset, max_len, cfg, supports
                )
        _build_unembedding(model, layout, root, output_values)
    model.eval()

    graph = _build_graph(
        layout, computed, placed, head_slots, heads_per_layer, n_layers, n_heads, root
    )
    compiled = CompiledModel(
        model=model,
        layout=layout,
        graph=graph,
        root=root,
        vocab=vocab,
        bos_id=bos_id,
        output_values=output_values,
        logit_scale=logit_scale,
        schedule={id(n): placed[id(n)] for n in computed},
    )
    if verify:
        _verify_exhaustively(compiled, root, vocab, max_len)
    return compiled


def _check_well_defined(root: SOp, vocab, max_len: int) -> None:
    """Reject a categorical aggregate that is ill defined anywhere in the input space.

    Over the *whole* space, not over sampled inputs.  A selector that matches
    two keys for some query has no categorical answer there, and a compiler
    that emitted weights anyway would produce a model computing a distribution
    where the program says a value -- wrong in a way the ground truth would not
    record.
    """
    for length in range(1, max_len + 1):
        for sequence in itertools.product(vocab, repeat=length):
            try:
                evaluate(root, list(sequence))
            except IllDefinedAggregate as exc:
                raise CompileError(
                    f"the program is ill defined on input {list(sequence)!r}: {exc}"
                ) from exc


def _build_embeddings(model, layout: Layout, vocab, bos_id: int, max_len: int) -> None:
    model.W_E.zero_()
    model.W_pos.zero_()
    tokens_space = next(
        (s for s in layout.all_spaces() if s.name == "tokens"), None
    )
    indices_space = next(
        (s for s in layout.all_spaces() if s.name == "indices"), None
    )
    if tokens_space is not None:
        for i, symbol in enumerate(vocab):
            model.W_E[i, tokens_space.dim_of(symbol)] = 1.0
    model.W_E[bos_id, layout.bos_dim] = 1.0
    if indices_space is not None:
        # Position 0 holds BOS and has no program index; position p holds input
        # element p-1.  Getting this offset wrong is the quietest possible bug:
        # every program still runs and `indices` is simply off by one.
        for p in range(1, max_len + 1):
            model.W_pos[p, indices_space.dim_of(p - 1)] = 1.0


def _build_head(model, layout, scratch, node, layer, slot, lam, cfg) -> None:
    attn = model.blocks[layer].attn
    selector = node.selector
    keys = layout.of(selector.keys)
    queries = layout.of(selector.queries)
    matrix = _predicate_matrix(selector, layout)
    bos_col = keys.width
    scale = lam * math.sqrt(cfg.d_head)

    # keys: the key variable one-hot, plus the BOS indicator in its own column
    for i in range(keys.width):
        attn.W_K[slot, keys.start + i, i] = 1.0
    attn.W_K[slot, layout.bos_dim, bos_col] = 1.0

    # queries: the predicate matrix, scaled; BOS at half score for an
    # aggregate (below any match, above any non-match) and at full score for a
    # width count (so its weight is exactly 1/(n+1)).
    bos_score = 1.0 if isinstance(node, SelectorWidth) else 0.5
    for qi in range(queries.width):
        for ki in range(keys.width):
            attn.W_Q[slot, queries.start + qi, ki] = scale * matrix[qi, ki]
        attn.W_Q[slot, queries.start + qi, bos_col] = scale * bos_score

    if isinstance(node, SelectorWidth):
        # Read the weight that landed on BOS; an MLP inverts it below.
        attn.W_V[slot, layout.bos_dim, 0] = 1.0
        attn.W_O[slot, 0, scratch[id(node)].start] = 1.0
        return

    out = layout.of(node)
    inner = layout.of(node.inner)
    if node.encoding == Encoding.CATEGORICAL:
        for value in inner.values:
            attn.W_V[slot, inner.dim_of(value), out.values.index(value)] = 1.0
        attn.W_V[slot, layout.bos_dim, out.values.index(node.default)] = 1.0
        for j in range(out.width):
            attn.W_O[slot, j, out.start + j] = 1.0
    else:
        attn.W_V[slot, inner.start, 0] = 1.0
        default = 0.0 if node.default is None else float(node.default)
        attn.W_V[slot, layout.bos_dim, 0] = default
        attn.W_O[slot, 0, out.start] = 1.0


def _firing_value(cfg) -> float:
    """What a lookup neuron's activation is when it fires, measured not assumed.

    The pre-activation is ``+0.5`` for the matching input and ``-0.5`` or less
    otherwise, by construction of the biases.  The value the activation
    function takes at ``+0.5`` is then divided out of the output weights.
    Measuring it from the configured function rather than hard-coding ``0.5``
    means changing ``act_fn`` cannot silently scale every lookup table.
    """
    value = float(act_fn(cfg.act_fn, torch.tensor([0.5], dtype=torch.float64)))
    off = float(act_fn(cfg.act_fn, torch.tensor([-0.5], dtype=torch.float64)))
    if abs(off) > 1e-12:
        raise CompileError(
            f"activation {cfg.act_fn!r} is {off:.3g} at -0.5 rather than zero, so a "
            "lookup-table neuron would leak from every non-matching entry. Compiled "
            "models need an activation that is exactly zero for negative inputs."
        )
    if value <= 0:
        raise CompileError(f"activation {cfg.act_fn!r} is {value:.3g} at +0.5; cannot normalise")
    return value


def _step_decode(mlp, src_dim: int, out: VariableSpace, pairs, offset: int, label: str) -> int:
    """Write ``f(x)`` into ``out``, for ``x`` a magnitude with a known finite support.

    ``pairs`` is ``[(input_value, output_value), ...]``.  One unit step per
    input value, placed at the midpoint between it and the previous one, and
    the output weights are *differences* between consecutive outputs so the
    steps telescope: at ``v_j`` every step up to ``j`` is on and the rest are
    off, so the sum is exactly ``f(v_j)``.

    The step width is half the smallest gap in the support.  That is what makes
    each step exactly zero or one on the values that occur -- never a fraction,
    which is what a single ReLU ramp would give and which would then be
    multiplied into every downstream weight.
    """
    ordered = sorted(pairs, key=lambda kv: float(kv[0]))
    values = [float(v) for v, _ in ordered]
    if len(values) != len(set(values)):
        raise CompileError(f"{label!r} has a repeated input value in its support: {values}")
    gaps = [b - a for a, b in itertools.pairwise(values)]
    delta = min(gaps) / 2.0 if gaps else 1.0
    if delta <= 0:
        raise CompileError(f"{label!r} has a non-increasing support: {values}")

    def contribution(value) -> dict[int, float]:
        if out.values is None:
            return {out.start: float(value)}
        return {out.dim_of(value): 1.0}

    previous: dict[int, float] = {}
    neuron = offset
    for i, (_input_value, result) in enumerate(ordered):
        threshold = values[0] - delta if i == 0 else (values[i - 1] + values[i]) / 2.0
        # step(x >= t) = (relu(x - t + delta) - relu(x - t)) / delta
        mlp.W_in[src_dim, neuron] = 1.0
        mlp.b_in[neuron] = -(threshold - delta)
        mlp.W_in[src_dim, neuron + 1] = 1.0
        mlp.b_in[neuron + 1] = -threshold
        current = contribution(result)
        for dim in set(current) | set(previous):
            change = current.get(dim, 0.0) - previous.get(dim, 0.0)
            if change:
                mlp.W_out[neuron, dim] += change / delta
                mlp.W_out[neuron + 1, dim] -= change / delta
        previous = current
        neuron += 2
    return neuron


def _build_mlp(model, layout, scratch, node, layer, offset, max_len, cfg, supports) -> int:
    mlp = model.blocks[layer].mlp
    fire = _firing_value(cfg)

    if isinstance(node, SelectorWidth):
        # The head put 1/(n+1) in a scratch dimension; decode it into n.
        src = scratch[id(node)].start
        pairs = [(1.0 / (n + 1), n) for n in range(max_len + 1)]
        return _step_decode(mlp, src, layout.of(node), pairs, offset, node.label)

    out = layout.of(node)
    if isinstance(node, Map):
        inner = layout.of(node.inner)
        if inner.values is None:
            # A magnitude whose achievable values are known: the step decoder
            # handles it.  A count declared numerical arrives here.
            inner_support = supports.get(id(node.inner))
            if inner_support is None:
                raise CompileError(
                    f"Map {node.label!r} reads {node.inner.label!r}, which is numerical and "
                    "whose support could not be bounded, so there is no finite set of "
                    "inputs to build a table over. Bound it with a declared_support."
                )
            pairs = [(v, node.fn(v)) for v in inner_support]
            return _step_decode(mlp, inner.start, out, pairs, offset, node.label)
        for i, value in enumerate(inner.values):
            neuron = offset + i
            mlp.W_in[inner.dim_of(value), neuron] = 1.0
            mlp.b_in[neuron] = -0.5
            _write_lookup_output(mlp, out, neuron, node.fn(value), fire, node.label)
        return offset + inner.width

    if isinstance(node, SequenceMap):
        lhs, rhs = layout.of(node.lhs), layout.of(node.rhs)
        if lhs.values is None or rhs.values is None:
            raise CompileError(
                f"SequenceMap {node.label!r} reads a numerical operand; a lookup table "
                "needs both inputs categorical with bounded support"
            )
        neuron = offset
        for a in lhs.values:
            for b in rhs.values:
                mlp.W_in[lhs.dim_of(a), neuron] = 1.0
                mlp.W_in[rhs.dim_of(b), neuron] = 1.0
                # Two one-hots sum to 2 only when both match; at most 1 otherwise.
                mlp.b_in[neuron] = -1.5
                _write_lookup_output(mlp, out, neuron, node.fn(a, b), fire, node.label)
                neuron += 1
        return neuron

    raise AssertionError(type(node).__name__)  # pragma: no cover


def _write_lookup_output(mlp, out: VariableSpace, neuron: int, value, fire: float, label) -> None:
    if out.values is None:
        mlp.W_out[neuron, out.start] = float(value) / fire
        return
    try:
        mlp.W_out[neuron, out.dim_of(value)] = 1.0 / fire
    except KeyError as exc:
        raise CompileError(
            f"{label!r} produced {value!r}, which is outside its own declared support. "
            f"A declared_support is a claim the compiler checks, and this one is wrong: {exc}"
        ) from exc


def _build_unembedding(model, layout: Layout, root: SOp, output_values) -> None:
    model.W_U.zero_()
    model.b_U.zero_()
    space = layout.of(root)
    if space.values is None:
        # A numerical root has one logit proportional to its value; there is
        # nothing to take an argmax over.
        model.W_U[space.start, 0] = UNEMBED_SCALE
        return
    for i, value in enumerate(output_values):
        model.W_U[space.dim_of(value), i] = UNEMBED_SCALE


def _build_graph(layout, computed, placed, head_slots, heads_per_layer, n_layers, n_heads, root):
    nodes: list[Node] = []
    implements: list[tuple[Node, str]] = []
    by_variable: dict[int, Node] = {}

    input_node = Node("embed")
    nodes.append(input_node)
    implements.append((input_node, "tokens, indices and the BOS indicator"))

    for node in computed:
        layer, kind = placed[id(node)]
        if kind == _ATTN:
            slot = head_slots[id(node)]
            gt = Node(f"blocks.{layer}.attn.z", (("head", slot),))
        else:
            gt = Node(f"blocks.{layer}.mlp.out")
        if gt not in nodes:
            nodes.append(gt)
        implements.append((gt, layout.of(node).name))
        by_variable[id(node)] = gt

    edges: list[tuple[Node, Node, str]] = []
    for node in computed:
        dst = by_variable[id(node)]
        for dep, port in _dependency_ports(node):
            src = by_variable.get(id(dep), input_node)
            if src != dst and (src, dst, port) not in edges:
                edges.append((src, dst, port))
        # A SelectorWidth's own MLP decodes its own head's output.
        if isinstance(node, SelectorWidth):
            layer, _ = placed[id(node)]
            mlp_node = Node(f"blocks.{layer}.mlp.out")
            if mlp_node not in nodes:
                nodes.append(mlp_node)
            if (dst, mlp_node, "mlp_in") not in edges:
                edges.append((dst, mlp_node, "mlp_in"))

    # The program's answer is read out through the unembedding, which is a real
    # edge that discovery will and should find.
    sink = Node("logits")
    nodes.append(sink)
    implements.append((sink, f"the read-out of {root.label}"))
    root_component = by_variable.get(id(root), input_node)
    edges.append((root_component, sink, "resid"))

    used_heads = {
        (layer, head_slots[id(n)]) for layer in range(n_layers) for n in heads_per_layer[layer]
    }
    unused = tuple(
        Node(f"blocks.{layer}.attn.z", (("head", slot),))
        for layer in range(n_layers)
        for slot in range(n_heads)
        if (layer, slot) not in used_heads
    )
    return GroundTruthGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        variables=layout.all_spaces(),
        unused=unused,
        program_repr=f"{root.label} = {type(root).__name__}(...)",
        implements=tuple(implements),
    )


def _verify_exhaustively(compiled: CompiledModel, root: SOp, vocab, max_len: int) -> None:
    """Compare against the interpreter on every input, and say which one failed."""
    total = sum(len(vocab) ** n for n in range(1, max_len + 1))
    if total > 20000:
        raise CompileError(
            f"exhaustive verification would need {total} inputs; call compile_program with "
            "verify=False and check a sample yourself, stating the sample in the result"
        )
    numerical = root.encoding == Encoding.NUMERICAL
    for length in range(1, max_len + 1):
        for sequence in itertools.product(vocab, repeat=length):
            expected = evaluate(root, list(sequence))
            got = compiled.run(list(sequence))
            if numerical:
                ok = all(
                    abs(float(a) - float(b)) < 1e-6
                    for a, b in zip(expected, got, strict=True)
                )
            else:
                ok = list(expected) == list(got)
            if not ok:
                raise CompileError(
                    f"the compiled model disagrees with the interpreter on "
                    f"{list(sequence)!r}: expected {expected!r}, got {got!r}"
                )
