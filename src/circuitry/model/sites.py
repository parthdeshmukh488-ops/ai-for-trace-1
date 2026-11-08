"""Names for every tensor inside the model, and a grammar for selecting them.

Written before the forward pass, because everything in the package is
organised around these strings and letting them accrete one at a time is how
you end up with ``blocks.0.attn.pattern`` next to ``block_0_attn_probs``.

Three things this module provides that a bare naming convention does not:

* **Expansion that cannot silently match nothing.**  ``resolve`` raises on a
  pattern with no matches.  A silently-empty expansion is how an experiment
  ends up patching the empty set and reporting a null result -- the plot looks
  fine, every effect is zero, and the conclusion is that the circuit is
  elsewhere.
* **Index spaces.**  A site is indexed by position, by position and head, or
  by head and two sequence axes, and patching needs to iterate over whichever
  it is.  :func:`index_space` makes that generic instead of a special case per
  site, which is what lets ``patch`` take "every head at every position"
  without knowing what kind of site it was given.
* **A wildcard that will not bankrupt you.**  ``blocks.*.attn.result`` is
  151 MB per layer for GPT-2 small at batch 8 and 512 positions, so
  ``attn.result`` is excluded from wildcard expansion unless asked for by name
  or with ``include_expensive=True``.  You can still capture it; you cannot
  capture it by accident while reaching for something else.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Sequence

__all__ = [
    "EXPENSIVE",
    "SiteNotFound",
    "all_sites",
    "axis_dim",
    "families",
    "index_space",
    "is_expensive",
    "parse",
    "resolve",
    "shape",
]


class SiteNotFound(KeyError):
    """A site name that does not exist, or a pattern that matched nothing."""


#: Sites whose memory cost is a multiple of the obvious one, and which are
#: therefore left out of wildcard expansion.  ``attn.result`` is ``n_heads``
#: times ``attn.out``; ``attn.scores`` and ``attn.pattern`` are quadratic in
#: sequence length but are needed often enough that excluding them would be
#: more annoying than helpful, so they stay in.
EXPENSIVE = frozenset({"attn.result"})

# leaf name -> (index space, shape template).  The templates are evaluated
# against (batch, seq, cfg) in `shape`; keeping them as a table rather than a
# chain of conditionals is what makes it hard for the shape used by a hook and
# the shape asserted by a test to drift apart.
_PER_LAYER: dict[str, tuple[tuple[str, ...], str]] = {
    "resid_pre": (("pos",), "B,T,d"),
    "ln1.scale": (("pos",), "B,T,1"),
    "ln1.normalized": (("pos",), "B,T,d"),
    "attn.q": (("pos", "head"), "B,T,H,dh"),
    "attn.k": (("pos", "head"), "B,T,H,dh"),
    "attn.v": (("pos", "head"), "B,T,H,dh"),
    "attn.scores": (("head", "query", "key"), "B,H,T,T"),
    "attn.pattern": (("head", "query", "key"), "B,H,T,T"),
    "attn.z": (("pos", "head"), "B,T,H,dh"),
    "attn.result": (("pos", "head"), "B,T,H,d"),
    "attn.out": (("pos",), "B,T,d"),
    "resid_mid": (("pos",), "B,T,d"),
    "ln2.scale": (("pos",), "B,T,1"),
    "ln2.normalized": (("pos",), "B,T,d"),
    "mlp.pre": (("pos", "neuron"), "B,T,dm"),
    "mlp.post": (("pos", "neuron"), "B,T,dm"),
    "mlp.out": (("pos",), "B,T,d"),
    "resid_post": (("pos",), "B,T,d"),
}

_GLOBAL: dict[str, tuple[tuple[str, ...], str]] = {
    "embed": (("pos",), "B,T,d"),
    "pos_embed": (("pos",), "B,T,d"),
    "ln_final.scale": (("pos",), "B,T,1"),
    "ln_final.normalized": (("pos",), "B,T,d"),
    "logits": (("pos", "vocab"), "B,T,V"),
}

_LAYER_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def families() -> tuple[str, ...]:
    """Every leaf name, per-layer and global, sorted."""
    return tuple(sorted({*_PER_LAYER, *_GLOBAL}))


def parse(name: str) -> tuple[int | None, str]:
    """``(layer, leaf)`` for a concrete site name; layer is ``None`` if global."""
    match = _LAYER_RE.match(name)
    if match:
        leaf = match.group(2)
        if leaf not in _PER_LAYER:
            raise SiteNotFound(
                f"{name!r} names layer {match.group(1)} but {leaf!r} is not a per-layer "
                f"site; the per-layer sites are: {', '.join(sorted(_PER_LAYER))}"
            )
        return int(match.group(1)), leaf
    if name in _GLOBAL:
        return None, name
    raise SiteNotFound(
        f"{name!r} is not a site name. Global sites: {', '.join(sorted(_GLOBAL))}. "
        f"Per-layer sites take the form blocks.<layer>.<leaf> with leaf one of: "
        f"{', '.join(sorted(_PER_LAYER))}"
    )


def index_space(name: str) -> tuple[str, ...]:
    """The axes a site is indexed by, excluding batch.

    This is what lets patching iterate generically.  ``resid_post`` is indexed
    by position alone; ``attn.z`` by position and head; ``attn.pattern`` by
    head and the two sequence axes, which is why patching an attention pattern
    is a different kind of operation from patching a residual stream and the
    API has to know.
    """
    layer, leaf = parse(name)
    table = _PER_LAYER if layer is not None else _GLOBAL
    return table[leaf][0]


def axis_dim(name: str, axis: str) -> int:
    """Which tensor dimension an index-space axis occupies.

    Uniform across every site in the table: the axes of the index space are
    the first ``len(index_space)`` dimensions after batch, in order.  That
    holds for ``[B,T,d]`` indexed by position, for ``[B,T,H,dh]`` indexed by
    position and head, and for ``[B,H,T,T]`` indexed by head, query and key --
    which is what lets one indexed-write combinator serve all of them instead
    of a special case per site family.
    """
    space = index_space(name)
    try:
        return 1 + space.index(axis)
    except ValueError:
        raise SiteNotFound(
            f"site {name!r} has index space {space}, which has no axis {axis!r}"
        ) from None


def is_expensive(name: str) -> bool:
    """Whether a site is excluded from wildcard expansion.  See :data:`EXPENSIVE`."""
    _, leaf = parse(name)
    return leaf in EXPENSIVE


def shape(name: str, cfg, batch: int, seq: int) -> tuple[int, ...]:
    """The tensor shape a site holds, as a tuple of ints."""
    layer, leaf = parse(name)
    if layer is not None and not 0 <= layer < cfg.n_layers:
        raise SiteNotFound(
            f"{name!r} names layer {layer} but the model has {cfg.n_layers} layers "
            f"(0 to {cfg.n_layers - 1})"
        )
    table = _PER_LAYER if layer is not None else _GLOBAL
    template = table[leaf][1]
    env = {
        "B": batch,
        "T": seq,
        "d": cfg.d_model,
        "dh": cfg.d_head,
        "H": cfg.n_heads,
        "dm": cfg.d_mlp,
        "V": cfg.d_vocab,
        "1": 1,
    }
    return tuple(env[token] for token in template.split(","))


def all_sites(cfg, *, include_expensive: bool = False) -> tuple[str, ...]:
    """Every concrete site in a model of this shape, in forward-pass order.

    Forward order rather than alphabetical, because a reader scanning a
    captured store or a patching result wants to see the model's own sequence,
    and because the attribution code walks sites in this order to build its
    graph.
    """
    out: list[str] = ["embed", "pos_embed"]
    for layer in range(cfg.n_layers):
        for leaf in _PER_LAYER:
            if leaf in EXPENSIVE and not include_expensive:
                continue
            out.append(f"blocks.{layer}.{leaf}")
    out += ["ln_final.scale", "ln_final.normalized", "logits"]
    return tuple(out)


def _expand_ranges(pattern: str) -> list[str]:
    """``blocks.0-2.mlp.post`` -> three patterns, one per layer."""
    parts = pattern.split(".")
    for i, part in enumerate(parts):
        match = _RANGE_RE.match(part)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            if lo > hi:
                raise SiteNotFound(
                    f"pattern {pattern!r} has the range {part!r} with its bounds "
                    "the wrong way round"
                )
            return [
                expanded
                for value in range(lo, hi + 1)
                for expanded in _expand_ranges(".".join([*parts[:i], str(value), *parts[i + 1 :]]))
            ]
    return [pattern]


def resolve(
    patterns: str | Iterable[str],
    cfg,
    *,
    include_expensive: bool = False,
) -> tuple[str, ...]:
    """Expand glob patterns to concrete site names, in forward-pass order.

    Supports ``*`` and ``?`` anywhere, and ``lo-hi`` ranges in the layer
    position.  Raises :class:`SiteNotFound` when a pattern matches nothing --
    see the module docstring for why that is worth an exception rather than an
    empty tuple.

    A pattern that names an expensive site explicitly gets it; a wildcard does
    not, unless ``include_expensive``.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns = list(patterns)
    if not patterns:
        raise SiteNotFound("no patterns given")

    universe = all_sites(cfg, include_expensive=True)
    cheap = set(all_sites(cfg, include_expensive=include_expensive))

    selected: dict[str, None] = {}
    for raw in patterns:
        matched: list[str] = []
        for pattern in _expand_ranges(raw):
            has_wildcard = any(ch in pattern for ch in "*?[")
            for name in universe:
                if not fnmatch.fnmatchcase(name, pattern):
                    continue
                # An explicit name always wins; a wildcard respects the budget
                # guard so that reaching for attn.* cannot pull in attn.result.
                if has_wildcard and name not in cheap:
                    continue
                matched.append(name)
        if not matched:
            hint = ""
            if any(is_expensive(name) for name in universe if fnmatch.fnmatchcase(name, raw)):
                hint = (
                    "; it matches only expensive sites, which wildcards skip -- "
                    "name one explicitly or pass include_expensive=True"
                )
            raise SiteNotFound(
                f"pattern {raw!r} matched no site in a {cfg.n_layers}-layer model{hint}. "
                f"Leaf names are: {', '.join(families())}"
            )
        for name in matched:
            selected[name] = None

    order = {name: i for i, name in enumerate(universe)}
    return tuple(sorted(selected, key=order.__getitem__))


def validate(names: Sequence[str], cfg) -> tuple[str, ...]:
    """Check concrete names exist in this model, returning them in forward order."""
    universe = all_sites(cfg, include_expensive=True)
    order = {name: i for i, name in enumerate(universe)}
    for name in names:
        layer, _ = parse(name)
        if layer is not None and not 0 <= layer < cfg.n_layers:
            raise SiteNotFound(
                f"{name!r} names layer {layer} but the model has {cfg.n_layers} layers"
            )
    return tuple(sorted(dict.fromkeys(names), key=order.__getitem__))
