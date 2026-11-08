"""The computational graph the compiler built, as the thing to be scored against.

A ground-truth node is a ``(site, index)`` pair in the *engine's* own naming
scheme, not in a separate vocabulary of the compiler's own.  That is the whole
design decision in this file: a discovered circuit and the true circuit are
then directly comparable, with no translation layer in between to disagree
about.

``unused`` is what makes precision well defined.  A compiled model has spare
capacity -- heads in a layer that no program variable needed -- and those are
genuine true negatives: a discovery method that returns them is wrong, and
without them "precision" would be a question of where you chose to draw the
boundary of the search space.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GroundTruthGraph", "Node"]


@dataclass(frozen=True, order=True)
class Node:
    """A component of the compiled model, named as the engine names it."""

    site: str
    index: tuple[tuple[str, int], ...] = ()

    @property
    def label(self) -> str:
        if not self.index:
            return self.site
        inner = ", ".join(f"{k}={v}" for k, v in self.index)
        return f"{self.site}[{inner}]"

    def __repr__(self) -> str:
        return f"Node({self.label})"


@dataclass(frozen=True)
class GroundTruthGraph:
    """What the compiled model computes, and with which components."""

    nodes: tuple[Node, ...]
    edges: tuple[tuple[Node, Node], ...]
    variables: tuple  # VariableSpace, one per allocated block
    unused: tuple[Node, ...]
    program_repr: str
    implements: tuple[tuple[Node, str], ...]  # node -> the variable it computes

    def __post_init__(self) -> None:
        known = set(self.nodes)
        for src, dst in self.edges:
            if src not in known or dst not in known:
                raise AssertionError(f"edge {src} -> {dst} names a node outside the graph")
        overlap = known & set(self.unused)
        if overlap:
            raise AssertionError(f"nodes marked both used and unused: {sorted(overlap)}")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def edge_set(self) -> frozenset[tuple[Node, Node]]:
        return frozenset(self.edges)

    def describe(self) -> str:
        lines = [self.program_repr, ""]
        for node, variable in self.implements:
            lines.append(f"  {node.label:34s} computes {variable}")
        if self.unused:
            lines.append("")
            lines.append(f"  unused (true negatives): {len(self.unused)}")
            for node in self.unused:
                lines.append(f"    {node.label}")
        lines.append("")
        for src, dst in self.edges:
            lines.append(f"  {src.label}  ->  {dst.label}")
        return "\n".join(lines)
