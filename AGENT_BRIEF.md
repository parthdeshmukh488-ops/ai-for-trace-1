# Agent brief — read this before writing any code

This repository is specified but not built. You are implementing it. The
specification is complete enough that you should not need to invent scope; if
you think you do, the answer is in [`docs/SPEC.md`](docs/SPEC.md) or it is a
question worth asking rather than guessing.

Work through [`prompts/`](prompts/) in order. Each file is one phase with its
own exit gate. Do not start a phase before the previous phase's gate is green.

---

## 1. What this project is

A mechanistic interpretability library whose distinguishing property is that
its methods are **calibrated against ground truth before being applied**. The
three regimes, and what counts as truth in each, are in the README. Everything
in the repository exists to serve one argument: an interpretability claim
should be checkable against something that could contradict it.

Concretely, the dependency order of the whole build is:

```
  transformer engine  ->  cross-checked against an independent implementation
          |
  symbolic compiler   ->  cross-checked against its own source program
          |
  attribution methods ->  cross-checked against exact patching on compiled nets
          |
  dictionary learning ->  cross-checked against a known feature basis
          |
  GPT-2 application   ->  no ground truth; carries the error bars from above
```

Nothing downstream is trustworthy if something upstream is not verified. Build
in this order and resist the temptation to jump to the interesting part.

---

## 2. Hard rules

These are not stylistic. Breaking any of them makes the repository worthless
for its stated purpose.

**R1 — Never write a number you did not measure.** No example number, no
"typical" value, no plausible-looking result in a docstring, README, comment
or JSON file. The README ships with `[[Mx]]` placeholders; you replace a
placeholder only with the value your experiment actually printed, in the same
commit that produces the `results/*.json` it came from. If an experiment has
not been run, the placeholder stays.

**R2 — Report disagreement, do not resolve it quietly.** If a method you
implemented performs worse than the naive baseline, if a published result does
not reproduce, if a metric contradicts the hypothesis the experiment was
written to test — that is a finding and it goes in the write-up. The existing
portfolio's credibility rests on these being present. Write them up in the
relevant `docs/` file and in the experiment's docstring.

**R3 — Every claim needs an oracle that could have contradicted it.** Before
implementing a measurement, write down what independent thing it is checked
against. If the answer is "nothing", you are building a proxy, and the
write-up must say so in those words.

**R4 — Determinism is a tested property, not an aspiration.** Every
stochastic entry point takes an explicit `seed`. There is a test that runs the
same pipeline twice and asserts bitwise-identical output. Global RNG state is
never read or mutated by library code: pass generators explicitly. See
`docs/ENGINEERING.md#determinism`.

**R5 — No silent device or dtype coercion.** Library functions do not call
`.cuda()`, do not promote to float64 behind the caller's back, and do not fall
back from a requested device. Dtype and device come from the model or from an
explicit argument, and mismatches raise.

**R6 — Interventions must be exactly reversible.** The hook system's contract
is that after any intervention context exits, the model produces bitwise the
same output as before it was entered. There is a test for this and it is not
optional, because every number in the repository is downstream of it.

**R7 — The test suite is the correctness claim.** Coverage floor is 93% and CI
enforces it. A test that asserts a function "runs" is not a test. Prefer:
properties that must hold for all inputs, agreement with a closed form,
agreement with a brute-force implementation of the same thing, and exact
recovery on cases constructed to have a known answer.

**R8 — Nothing about the tooling used to write this code appears in the
repository.** No tool names, no assistant attribution, no generated-by notices
in files, commit messages, or metadata. Commits are authored normally.

---

## 3. House style

Match the author's existing repositories. The style is specific and consistent.

**Module docstrings argue for the design.** They do not describe the file;
they explain why the file is the way it is, usually with a short bulleted case
and a named alternative that was rejected. Good opening lines read like
`"""The Black formula, in the one parameterisation that stays accurate.` —
a claim, not a label.

**Comments explain reasoning, especially for constants.** A magic number gets
a comment saying how it was chosen and what measurement justifies it. If you
chose it by trying things until it worked, say that.

**Prose wraps at 78 columns. Code is `line-length = 100`, ruff-clean.**

**`from __future__ import annotations` at the top of every module. `__all__`
is explicit and sorted.**

**Public API is small and flat.** Users should be able to do real work from
`circuitry.<thing>` without reaching into submodules. Internal helpers are
`_`-prefixed and not re-exported.

**Experiment scripts open with the question, not the method.** The docstring
states what is being measured, what it is compared against, why that
comparison is the informative one, and ends with the usage line:

```
    python experiments/attribution_error.py [--quick]
```

Every experiment:
- takes `--quick`, which must exercise every code path at reduced size,
- takes `--seed`, defaulting to a fixed value,
- writes `experiments/results/<name>.json` with the raw numbers plus a
  `metadata` block (seed, git SHA, device, dtype, package versions, wall time),
- writes `experiments/results/<name>.png` through `experiments/style.py`,
- prints the headline numbers to stdout in a form you can paste into a table,
- and under `--quick` writes `<name>_quick.json`, which is gitignored because
  it is a smoke test and never a result.

**Figures are plotted against the informative axis.** The existing repo plots
accuracy against *conditioning* rather than against moneyness, because that
separates how hard the problem is from how well the method handled it. Look
for the equivalent axis every time: effect size, sparsity, compute, not just
layer index.

---

## 4. Definition of done, per component

A component is done when all five hold:

1. It is implemented and its public API matches `docs/SPEC.md`.
2. Its tests pass, including at least one test that would fail if the
   component were subtly wrong rather than obviously broken.
3. Its oracle cross-check passes (R3).
4. Its docstrings argue for the design (§3) and contain no unmeasured numbers.
5. `pytest tests/ -q --cov=circuitry --cov-fail-under=93` is green and
   `ruff check .` is clean.

---

## 5. Commits

Small, self-contained, one idea each. Present tense, lowercase after the
first word, no trailing period, no scope prefixes:

```
Add the hook registry and its reversibility test
Measure attribution-patching error against exact patching
Fix the LayerNorm fold; the direct-logit path was off by the scale factor
```

Commit the `results/*.json` and `results/*.png` that back a README number in
the same commit that fills in the placeholder. Never commit weights,
checkpoints or activation caches — `.gitignore` already covers them, keep it
that way.

---

## 6. Compute

Phases 0 through 5 run on CPU in minutes to tens of minutes. Phase 6 (GPT-2
small) wants a GPU but will complete on CPU; phase 7 (dictionaries on GPT-2)
effectively needs one. `docs/ENGINEERING.md#compute-tiers` gives the budget for
each experiment at each tier, including the ones that should be skipped at the
CPU tier rather than run badly.

If a run is too expensive to complete, reduce its *scope* explicitly and say
so in the results JSON (`"scope": "reduced: 200 of 1000 prompts"`) rather than
quietly running less and reporting it as the full measurement.

---

## 7. When you are stuck

The specification is wrong sometimes. If you find an inconsistency between
`docs/SPEC.md` and reality — an algorithm that cannot work as described, a
metric that is not well-defined, a ground truth that turns out not to be
ground truth — stop, write down the problem, fix the specification, and note
the change in `docs/SPEC.md` under a `## Revisions` heading at the bottom.
That is a better outcome than an implementation that quietly diverges from its
own documentation.
