"""Components, edges, and a subgraph of them.

The unit is a *component* -- a thing that writes into the residual stream --
and an *edge* carries a port, so a circuit can say that one head reads
another's output through its query input and not through its value input.
Edges between components without ports cannot express that, and it is
precisely the distinction the published circuits in this area are made of.

The search space is the full dependency DAG the architecture allows:

* a head at layer ``L`` can read, through any of ``q``, ``k`` and ``v``, from
  the embedding and from every head and MLP at a layer below ``L``;
* an MLP at layer ``L`` can read, through ``mlp_in``, from the embedding, from
  every head at layer ``L`` or below, and from every MLP below ``L``;
* the logits read from everything.

Stating it this way makes the denominator of "precision" explicit.  A
discovery method is choosing a subset of these edges, so how many there are is
part of what any precision number means, and :func:`all_edges` is where that
number comes from rather than being implicit in a loop somewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Literal

__all__ = [
    "PORTS",
    "Circuit",
    "Component",
    "Edge",
    "all_components",
    "all_edges",
    "component_of_node",
]

Port = Literal["q", "k", "v", "mlp_in", "resid"]
PORTS: tuple[str, ...] = ("q", "k", "v", "mlp_in", "resid")


@dataclass(frozen=True, order=True)
class Component:
    """Something that writes into the residual stream, or reads it at the end.

    Ordered so that sorting a set of components gives a stable, readable
    listing; the sort key puts the embedding first and the logits last because
    ``layer`` is ``-1`` and a large number respectively.
    """

    layer: int
    kind: str  # "embed" | "attn" | "mlp" | "logits"
    head: int | None = None

    @staticmethod
    def embed() -> Component:
        return Component(-1, "embed")

    @staticmethod
    def logits(n_layers: int) -> Component:
        return Component(n_layers, "logits")

    @staticmethod
    def attn(layer: int, head: int) -> Component:
        return Component(layer, "attn", head)

    @staticmethod
    def mlp(layer: int) -> Component:
        return Component(layer, "mlp")

    @property
    def output_site(self) -> str:
        """The site whose value *is* this component's contribution to the stream.

        ``attn.z`` rather than ``attn.out`` for a head, because ``out`` is the
        sum over heads and patching it would move every head at once.  Scaling
        one head's ``z`` scales exactly that head's contribution, since
        ``out = sum_h z_h W_O_h + b_O``.
        """
        if self.kind == "embed":
            return "embed"
        if self.kind == "attn":
            return f"blocks.{self.layer}.attn.z"
        if self.kind == "mlp":
            return f"blocks.{self.layer}.mlp.out"
        if self.kind == "logits":
            return "logits"
        raise ValueError(f"unknown component kind {self.kind!r}")

    @property
    def write_index(self) -> dict[str, int]:
        return {"head": self.head} if self.head is not None else {}

    @property
    def label(self) -> str:
        if self.kind == "embed":
            return "embed"
        if self.kind == "logits":
            return "logits"
        if self.kind == "attn":
            return f"a{self.layer}.h{self.head}"
        return f"m{self.layer}"

    def __repr__(self) -> str:
        return f"Component({self.label})"


@dataclass(frozen=True, order=True)
class Edge:
    """A dependency from one component's output to another's named input."""

    source: Component
    target: Component
    port: str

    #: Which ports each kind of receiver has.  Checked at construction, because
    #: an edge whose port does not match its target is not merely unusual -- the
    #: ablation engine looks edges up by ``(layer, port, head)`` and would
    #: silently never find such an edge, so a typo would quietly remove an edge
    #: from a circuit rather than raising.
    _PORTS_BY_KIND: ClassVar[dict[str, tuple[str, ...]]] = {
        "attn": ("q", "k", "v"),
        "mlp": ("mlp_in",),
        "logits": ("resid",),
    }

    def __post_init__(self) -> None:
        if self.port not in PORTS:
            raise ValueError(f"port must be one of {PORTS}, got {self.port!r}")
        if self.target.kind == "embed":
            raise ValueError("the embedding has no inputs, so nothing can be an edge into it")
        allowed = self._PORTS_BY_KIND.get(self.target.kind)
        if allowed is None:
            raise ValueError(f"unknown receiver kind {self.target.kind!r}")
        if self.port not in allowed:
            raise ValueError(
                f"{self.target.label} is a {self.target.kind} receiver, whose ports are "
                f"{allowed}; got {self.port!r}"
            )
        if self.source.kind == "logits":
            raise ValueError("the logit sink has no output, so nothing can be an edge from it")

    @property
    def label(self) -> str:
        return f"{self.source.label} -{self.port}-> {self.target.label}"

    def __repr__(self) -> str:
        return f"Edge({self.label})"


def all_components(cfg, *, include_logits: bool = True) -> tuple[Component, ...]:
    out = [Component.embed()]
    for layer in range(cfg.n_layers):
        out += [Component.attn(layer, h) for h in range(cfg.n_heads)]
        out.append(Component.mlp(layer))
    if include_logits:
        out.append(Component.logits(cfg.n_layers))
    return tuple(out)


def _writers_below(cfg, layer: int, *, include_same_layer_attn: bool) -> list[Component]:
    out = [Component.embed()]
    for other in range(cfg.n_layers):
        if other < layer:
            out += [Component.attn(other, h) for h in range(cfg.n_heads)]
            out.append(Component.mlp(other))
        elif other == layer and include_same_layer_attn:
            out += [Component.attn(other, h) for h in range(cfg.n_heads)]
    return out


def all_edges(cfg) -> tuple[Edge, ...]:
    """Every edge the architecture permits.  The denominator of any precision."""
    out: list[Edge] = []
    for layer in range(cfg.n_layers):
        senders = _writers_below(cfg, layer, include_same_layer_attn=False)
        for head in range(cfg.n_heads):
            target = Component.attn(layer, head)
            for port in ("q", "k", "v"):
                out += [Edge(s, target, port) for s in senders]
        mlp_senders = _writers_below(cfg, layer, include_same_layer_attn=True)
        out += [Edge(s, Component.mlp(layer), "mlp_in") for s in mlp_senders]
    sink = Component.logits(cfg.n_layers)
    out += [
        Edge(s, sink, "resid")
        for s in _writers_below(cfg, cfg.n_layers, include_same_layer_attn=False)
    ]
    return tuple(out)


def component_of_node(node) -> Component:
    """Translate a ground-truth :class:`circuitry.symbolic.truth.Node`.

    The compiler emits its graph in the engine's site names so that no
    translation layer is needed to compare a discovered circuit with the true
    one.  This is that non-layer: it reads the site string rather than
    maintaining a parallel vocabulary.
    """
    site = node.site
    index = dict(node.index)
    if site == "embed":
        return Component.embed()
    if site == "logits":
        raise ValueError("the logit sink has no layer; construct it with Component.logits")
    parts = site.split(".")
    layer = int(parts[1])
    if ".attn." in site:
        return Component.attn(layer, index["head"])
    if ".mlp." in site:
        return Component.mlp(layer)
    raise ValueError(f"cannot interpret {site!r} as a component")


@dataclass(frozen=True)
class Circuit:
    """A subgraph: some components and some edges between them."""

    components: frozenset[Component]
    edges: frozenset[Edge]
    label: str = ""

    @staticmethod
    def of(components=(), edges=(), *, label: str = "") -> Circuit:
        comps = set(components)
        for edge in edges:
            comps.add(edge.source)
            comps.add(edge.target)
        return Circuit(frozenset(comps), frozenset(edges), label)

    @staticmethod
    def complete(cfg, *, label: str = "complete") -> Circuit:
        return Circuit.of(all_components(cfg), all_edges(cfg), label=label)

    @staticmethod
    def empty(*, label: str = "empty") -> Circuit:
        return Circuit(frozenset(), frozenset(), label)

    @staticmethod
    def from_ground_truth(graph, cfg, *, label: str = "ground truth") -> Circuit:
        """The compiler's graph as a circuit, for scoring a discovery against."""
        sink = Component.logits(cfg.n_layers)
        nodes = {}
        for node in graph.nodes:
            try:
                nodes[node] = component_of_node(node)
            except ValueError:
                nodes[node] = sink
        edges = set()
        for src, dst, port in graph.edges:
            if nodes[src] != nodes[dst]:
                edges.add(Edge(nodes[src], nodes[dst], port))
        return Circuit.of(set(nodes.values()), edges, label=label)

    # -- set algebra ------------------------------------------------------

    def __or__(self, other: Circuit) -> Circuit:
        return Circuit(self.components | other.components, self.edges | other.edges)

    def __and__(self, other: Circuit) -> Circuit:
        return Circuit(self.components & other.components, self.edges & other.edges)

    def __sub__(self, other: Circuit) -> Circuit:
        return Circuit(self.components - other.components, self.edges - other.edges)

    def __len__(self) -> int:
        return len(self.edges) or len(self.components)

    def __contains__(self, item) -> bool:
        return item in self.components or item in self.edges

    # -- scoring ----------------------------------------------------------

    def score_against(self, truth: Circuit) -> dict[str, float]:
        """Precision, recall and F1, at node level and at edge level separately.

        Separately because they can disagree sharply: recovering every
        component while getting their connections wrong scores well on nodes
        and badly on edges, and a single number would hide which happened.
        """
        out = {}
        for name, mine, theirs in (
            ("node", self.components, truth.components),
            ("edge", self.edges, truth.edges),
        ):
            hits = len(mine & theirs)
            precision = hits / len(mine) if mine else 0.0
            recall = hits / len(theirs) if theirs else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            out[f"{name}_precision"] = precision
            out[f"{name}_recall"] = recall
            out[f"{name}_f1"] = f1
            out[f"{name}_found"] = len(mine)
            out[f"{name}_true"] = len(theirs)
        return out

    def jaccard(self, other: Circuit, *, on: str = "edge") -> float:
        mine, theirs = (
            (self.edges, other.edges) if on == "edge" else (self.components, other.components)
        )
        union = len(mine | theirs)
        return len(mine & theirs) / union if union else 1.0

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "components": [
                {"layer": c.layer, "kind": c.kind, "head": c.head, "label": c.label}
                for c in sorted(self.components)
            ],
            "edges": [
                {"source": e.source.label, "target": e.target.label, "port": e.port}
                for e in sorted(self.edges)
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def describe(self) -> str:
        lines = [f"{self.label or 'circuit'}: {len(self.components)} components, "
                 f"{len(self.edges)} edges"]
        lines += [f"  {e.label}" for e in sorted(self.edges)]
        return "\n".join(lines)
