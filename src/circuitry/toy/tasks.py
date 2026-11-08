"""Matched clean and counterfactual prompts, with the degenerate ones counted.

A patching experiment needs pairs of prompts that differ in their *answer*.
Generating them by sampling two inputs at random does not work: for ``sort``,
``abab`` and ``baba`` have the same sorted output, so the metric scores both
runs identically and there is nothing a patch could restore.  Those pairs are
not merely useless -- normalising by a near-zero gap is what produces patching
plots with effects of 40 and conclusions about circuits that are really
conclusions about one bad pair.

So the generator searches for pairs that differ, records the position where
they differ so the metric can be evaluated there, and **reports how many
candidates it discarded**.  The discard rate is a property of the program:
``hist`` is easy to make differ, ``two_hop`` over a small vocabulary is not,
and a reader who is not told the rate cannot tell which they are looking at.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..attrib.metric import PromptPair, logit_diff

__all__ = ["TaskReport", "compiled_pairs"]


@dataclass(frozen=True)
class TaskReport:
    """The pairs, and an honest account of what it took to find them."""

    pair: PromptPair
    n_requested: int
    n_found: int
    n_discarded: int
    length: int
    positions: tuple[int, ...]
    program: str

    @property
    def discard_rate(self) -> float:
        total = self.n_found + self.n_discarded
        return self.n_discarded / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "program": self.program,
            "n_requested": self.n_requested,
            "n_found": self.n_found,
            "n_discarded": self.n_discarded,
            "discard_rate": self.discard_rate,
            "length": self.length,
            "positions": list(self.positions),
        }


def compiled_pairs(
    compiled,
    *,
    n_pairs: int = 16,
    length: int | None = None,
    generator: torch.Generator | None = None,
    max_attempts: int | None = None,
) -> TaskReport:
    """Clean and counterfactual inputs for a compiled program, with a metric.

    The metric is the logit difference between the clean answer and the
    counterfactual answer, evaluated at the position where they differ -- which
    varies from pair to pair, hence the per-item position support in
    :func:`circuitry.attrib.metric.logit_diff`.
    """
    if compiled.root.encoding != "categorical":
        raise ValueError(
            f"{compiled.root.label!r} has a numerical output, so there is no logit "
            "difference to take; patch it with a metric defined on the read-out instead"
        )
    vocab = compiled.vocab
    length = length if length is not None else compiled.cfg.n_ctx - 1
    if length < 1 or length > compiled.cfg.n_ctx - 1:
        raise ValueError(
            f"length must be between 1 and {compiled.cfg.n_ctx - 1}, got {length}"
        )
    generator = generator or torch.Generator().manual_seed(0)
    attempts = max_attempts if max_attempts is not None else 200 * n_pairs

    cleans: list[list] = []
    corrupts: list[list] = []
    positions: list[int] = []
    correct: list[int] = []
    incorrect: list[int] = []
    discarded = 0

    def sample() -> list:
        idx = torch.randint(0, len(vocab), (length,), generator=generator).tolist()
        return [vocab[i] for i in idx]

    for _ in range(attempts):
        if len(cleans) >= n_pairs:
            break
        a, b = sample(), sample()
        out_a, out_b = compiled.run(a), compiled.run(b)
        differing = [i for i, (x, y) in enumerate(zip(out_a, out_b, strict=True)) if x != y]
        if not differing:
            discarded += 1
            continue
        # The first differing output position, deterministically, so a seed
        # reproduces the whole task and not merely the inputs.
        at = differing[0]
        cleans.append(a)
        corrupts.append(b)
        # Output position `at` is model position `at + 1`: position 0 holds the
        # beginning-of-sequence symbol, which the program never sees.
        positions.append(at + 1)
        correct.append(compiled.output_values.index(out_a[at]))
        incorrect.append(compiled.output_values.index(out_b[at]))

    if not cleans:
        raise ValueError(
            f"found no pair of length {length} with differing outputs for "
            f"{compiled.root.label!r} after {attempts} attempts; the program may be "
            "constant on this vocabulary"
        )

    clean_tokens = torch.cat([compiled.encode(s) for s in cleans], dim=0)
    corrupt_tokens = torch.cat([compiled.encode(s) for s in corrupts], dim=0)
    metric = logit_diff(correct, incorrect, position=torch.tensor(positions))
    pair = PromptPair(clean_tokens, corrupt_tokens, metric, compiled.root.label)
    return TaskReport(
        pair=pair,
        n_requested=n_pairs,
        n_found=len(cleans),
        n_discarded=discarded,
        length=length,
        positions=tuple(positions),
        program=compiled.root.label,
    )
