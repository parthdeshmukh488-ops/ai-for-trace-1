Read AGENT_BRIEF.md. Phase 3's gate must be green. This is phase 4 of
docs/BUILD_PLAN.md, and it produces the project's central result.

Implement SPEC §3.5: `attrib/gradients.py` — attribution patching, integrated
gradients, softmax recomputation, and gradient dropout. Then
`experiments/attribution_error.py` per docs/EXPERIMENTS.md#e2.

Build the validation test before the experiment. Construct a config with the
nonlinearities removed — identity activation, and a single-position input so
the softmax is a no-op — where the first-order estimate is *exactly* the
patching effect, and assert agreement to float tolerance. Until that passes,
any error the experiment measures is indistinguishable from a sign bug in the
estimator, and the whole experiment is void.

Then run E2 as specified. Non-negotiable parts of the protocol:

* Results are reported **per program**. Never pool the nine into one headline
  number; two easy programs would carry it.
* `two_hop` and `most_freq` get their own panel, because composition is the
  structure first-order methods are expected to fail on, and a pooled number
  would hide exactly the finding the experiment exists to produce.
* Both directions, denoise and noise, reported separately.
* The error broken down **by the size of the true effect** (`by_effect_decile`
  in the JSON). A method that is accurate on small effects and wrong on large
  ones has a good summary correlation and is useless, and that is the single
  most important thing this experiment can show.
* Count forward and backward passes from actual hook tallies, not estimates,
  and include the compute panel.

The five hypotheses (a) through (e) in docs/EXPERIMENTS.md#e2 were written
before the run. Report which survived. If a fix — softmax recomputation or
gradient dropout — does not help on these programs, that is the result and it
goes in the write-up as such, not into a footnote.

Then fill in `[[M1a]]` through `[[M1i]]` in README.md with the measured
values, in the same commit as the results JSON they come from.

Gate:

```bash
PYTHONPATH=experiments python experiments/attribution_error.py
grep -c '\[\[M1' README.md    # must print 0
```

Report: the filled README table; which of (a)-(e) held; the worst single
program and site type for sign errors; and the compute-versus-error frontier
in one sentence — at what integrated-gradient step count does the estimate
stop improving, and how does that cost compare to exact patching.
