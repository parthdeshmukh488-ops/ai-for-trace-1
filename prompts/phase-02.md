Read AGENT_BRIEF.md. Phase 1's gate must be green, including the cross-check.
This is phase 2 of docs/BUILD_PLAN.md: the symbolic compiler.

This is the phase the rest of the project's credibility rests on, because it
produces the ground truth everything downstream is scored against. Implement
SPEC §2 in full.

Order of work:

1. `symbolic/lang.py` — the SOp algebra and, crucially, a **reference
   interpreter** that evaluates a program directly on a sequence. Write and
   test the interpreter before the compiler. It is the oracle for the
   compiler, and if it is wrong, every ground truth in the repository is wrong
   in the same direction and nothing downstream will reveal it.
2. `symbolic/bases.py` — `VariableSpace` and the allocator. Assert disjointness
   as an invariant inside the allocator, not only in a test.
3. `symbolic/compiler.py` — `Select`/`Aggregate` to a head, `Map`/`SequenceMap`
   to an MLP lookup table, `SelectorWidth` to head plus MLP, per SPEC §2.4.
   Two things to get right rather than approximately right:
   - the well-definedness check on categorical `Aggregate` (SPEC §2.2), which
     must raise over the whole input space rather than on the inputs you
     happened to try;
   - the MLP output scaling, which must be derived numerically from the
     configured activation function at the measured firing margin, so that
     changing `act_fn` cannot silently break the compiler.
4. `symbolic/truth.py` — the `GroundTruthGraph`, with `unused` populated and
   verified empty of causal influence.
5. `symbolic/programs.py` — all nine programs from SPEC §2.6.

Then `experiments/compiler_fidelity.py` per docs/EXPERIMENTS.md#e1, and
`examples/02_compiled_ground_truth.py`.

Verification is exhaustive, not sampled: for `T <= 5` over a 4-symbol
vocabulary, every input, every program, exact equality with the interpreter.

Gate:

```bash
pytest tests/test_compiler.py -q
PYTHONPATH=experiments python experiments/compiler_fidelity.py
```

All nine programs at 1.0 exact-match at a single shared logit scale.

Report: the chosen logit scale and the measured fidelity curve that justifies
it; which programs needed the largest scale and whether that matches the
`m*exp(-lambda)/n` prediction in SPEC §2.4; where float32 overflow bounds the
scale from above; and any program in the suite you could not express in the
language, with what was missing.
