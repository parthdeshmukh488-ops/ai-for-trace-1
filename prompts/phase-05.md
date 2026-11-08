Read AGENT_BRIEF.md. Phase 4's gate must be green. This is phase 5 of
docs/BUILD_PLAN.md: automated discovery and circuit metrics.

Implement SPEC §3.6 and §3.7: `attrib/acdc.py` and `attrib/metrics.py`.

For discovery, two requirements that distinguish this from edge ranking:

* removals must be **committed and persistent** as the pruning walks the graph
  in reverse topological order, so each edge is judged in the context of the
  ones already gone;
* the graph is over **ports**, not components, so a recovered circuit can
  express that one head reads another's output through its query input only.

For the metrics, completeness and minimality are defined over sampled subsets,
so each must report its own Monte Carlo error. A completeness number computed
over 20 subsets without an error bar is the kind of thing this repository
exists not to produce.

Before the sweeps, measure the metric's **noise floor**: rerun the metric on
identical inputs with different batch composition and report the spread. Every
threshold in the sweep should be stated relative to it, because a threshold
below the noise floor returns the whole model and that is a property of the
measurement, not of the circuit.

Then run both experiments:

* `experiments/ablation_choice.py` per docs/EXPERIMENTS.md#e3 — five
  counterfactual arms including the deliberately-wrong mean-over-positions,
  each shown at its own best threshold.
* `experiments/discovery_roc.py` per docs/EXPERIMENTS.md#e4 — greedy with
  exact and with attribution scoring, individual-effect ranking at matched
  edge count, and a random subgraph at matched size. Report F1 against
  forward passes, not F1 alone.

Fill in `[[M2a]]` through `[[M2h]]` in README.md.

Gate:

```bash
PYTHONPATH=experiments python experiments/ablation_choice.py
PYTHONPATH=experiments python experiments/discovery_roc.py
grep -c '\[\[M2' README.md    # must print 0
```

Report: which counterfactual recovered the planted circuit best and by how
much; whether the choice changed the *conclusion* rather than just the numbers
on any program; what mean-over-positions cost; the measured noise floor; and
how much worse the cheap attribution-scored discovery was than the exact one,
against its cost saving.
