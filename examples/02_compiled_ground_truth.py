"""A transformer whose algorithm is known, because it was written down first.

This is the thing the rest of the project is measured against.  A program is
written in a small sequence language, compiled to exact weights, and then
three claims are checked: the model agrees with the program on every input,
every intermediate variable is readable out of the residual stream, and the
heads the compiler says are unused really are causally dead.

    python examples/02_compiled_ground_truth.py
"""

from __future__ import annotations

import itertools

import torch

from circuitry import compile_program, intervene
from circuitry.model.hooks import replace_at
from circuitry.symbolic import evaluate, programs

torch.set_grad_enabled(False)

NAME = "dyck1"
program, vocab = programs.build(NAME)
spec = programs.PROGRAMS[NAME]
compiled = compile_program(program, vocab, max_len=5)

cfg = compiled.cfg
print(f"{NAME}: {spec.exercises}")
print(
    f"compiled to {cfg.n_layers} layers, {cfg.n_heads} heads, d_model={cfg.d_model}, "
    f"d_mlp={cfg.d_mlp}, not causal\n"
)

# ---------------------------------------------------------------------------
# 1. the residual stream, variable by variable
# ---------------------------------------------------------------------------
print("the residual stream is a direct sum of the program's variables:")
print(compiled.layout.describe())
print()

# ---------------------------------------------------------------------------
# 2. it agrees with its source on every input
# ---------------------------------------------------------------------------
checked = 0
for length in range(1, 6):
    for sequence in itertools.product(vocab, repeat=length):
        assert compiled.run(list(sequence)) == evaluate(program, list(sequence))
        checked += 1
print(f"exact agreement with the interpreter on all {checked} inputs of length 1 to 5\n")

# ---------------------------------------------------------------------------
# 3. every intermediate is readable, not just the answer
# ---------------------------------------------------------------------------
sequence = list("(())(")
print(f"input {''.join(sequence)}")
for name, values in compiled.trace(sequence).items():
    print(f"   {name:18s} {values}")
print()

# ---------------------------------------------------------------------------
# 4. the ground-truth graph, and its true negatives
# ---------------------------------------------------------------------------
graph = compiled.graph
print(f"ground truth: {graph.n_nodes} nodes, {graph.n_edges} edges, "
      f"{len(graph.unused)} unused heads")
for node, variable in graph.implements:
    print(f"   {node.label:30s} computes {variable}")
print()

tokens_in = compiled.encode(sequence)
before = compiled.model(tokens_in)
hooks: dict[str, list] = {}
for node in graph.unused:
    head = dict(node.index)["head"]
    hooks.setdefault(node.site, []).append(replace_at(node.site, {"head": head}, 0.0))
with intervene(compiled.model, hooks):
    after = compiled.model(tokens_in)
print(f"ablating all {len(graph.unused)} unused heads at once changes the output by "
      f"{float((before - after).abs().max()):.1e}")
print("-- which is what makes them true negatives, and precision well defined")
