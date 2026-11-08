Read AGENT_BRIEF.md. All previous gates must be green. This is phase 10 of
docs/BUILD_PLAN.md: the write-up.

The implementation is done. What remains is making the repository say true
things about itself, which is the part that decides whether any of it was
worth doing.

1. **Every placeholder replaced.** `grep -c '\[\[M' README.md` prints 0. Each
   number traceable to a committed `experiments/results/*.json`. If a
   measurement was never run, the honest move is to remove the claim from the
   README, not to leave a placeholder or invent a value.

2. **`docs/EXPERIMENTS.md` completed.** Every `Result` block filled, and for
   each pre-registered hypothesis an explicit statement of whether it survived.

3. **The `## Where the measurements disagreed` section written**, in
   docs/EXPERIMENTS.md and summarised in the README. Candidates you should
   already have: whether the attribution-patching fixes helped; whether the
   proxy metrics predicted dictionary recovery; whether greedy discovery beat
   edge ranking at matched cost; whether the modular-addition seeds agreed;
   which published IOI heads did not reproduce. If this section is empty, the
   experiments were not sharp enough — that is a finding about the project,
   and saying so is better than an empty section.

4. **Rewrite the README's prose around what was actually found.** It currently
   argues for a programme; it should argue for a result. Keep the structure and
   the house voice, change the claims to the measured ones. If a headline claim
   did not survive contact with the measurement, the README leads with what
   did.

5. **Examples run from a clean checkout.** All three, no manual setup beyond
   the install.

6. **Coverage and lint.** 93% floor enforced in CI, `ruff check .` clean.

7. **A `docs/LIMITATIONS.md`**, which the specification does not yet contain
   and should: what the compiled-model regime does not tell you about trained
   models, what the toy feature basis does not tell you about real features,
   what the IOI result does not tell you about GPT-2 in general, and which of
   the repository's numbers would change most if the metric's noise floor were
   taken more seriously. Write it as the strongest honest case *against* the
   project's conclusions. If you cannot make that case, you do not yet
   understand the results well enough to publish them.

Gate:

```bash
pytest tests/ -q -m "not gpu" --cov=circuitry --cov-fail-under=93
ruff check .
grep -c '\[\[M' README.md    # must print 0
```

Report: the gate output, the contents of `docs/LIMITATIONS.md` in summary, and
the three findings you would lead with if this were being presented. Then say
which one you are least confident in and what measurement would settle it.
