# Experiment registry

One section per experiment. Each states the question, the hypothesis being
risked, the oracle that could contradict it, the protocol in enough detail to
rerun, and an empty **Result** block to fill in after the run.

The hypothesis is written down *before* the run. A hypothesis recorded after
seeing the numbers is not a hypothesis, and the point of writing them here is
to make it visible when one fails — which is the most valuable output this
repository can produce.

Common conventions: `--quick` reduces every count below by roughly 20x and
writes `<name>_quick.json`; `--seed` defaults to `0`; every JSON carries the
metadata block from SPEC §7.

---

## E1 — `compiler_fidelity.py`

**Question.** Does a compiled program compute exactly what its source says, and
at what attention-logit scale does the softmax become hard enough for that to
hold?

**Hypothesis.** There is a threshold scale above which exact-match is 100% for
every program in the suite, and below it the failures concentrate on programs
whose selectors match many positions (because the error term goes as
`m e^{-lambda} / n`).

**Oracle.** The reference interpreter in `symbolic/lang.py`, written before the
compiler and tested independently. Exhaustive over the input space, not
sampled.

**Protocol.**

* All nine programs in the suite.
* Vocabulary of 4 symbols plus BOS; sequence lengths `T in {3,4,5}`, every
  input exhaustively (at most 1365 per program per length).
* Logit scale `lambda` swept over `{1, 2, 4, 8, 16, 32, 64, 100}`.
* Also sweep `float32` against `float64` at the largest `lambda`, to separate
  softmax hardness from floating-point saturation. At large `lambda` the
  exponentials overflow before the softmax is hard, and the measurement must
  show where that boundary is.

**Outputs.** `results/compiler_fidelity.{json,png}`. JSON keys:
`per_program[program][lambda] -> {exact_match_rate, n_inputs, max_abs_error}`,
plus `chosen_lambda` and the reason. Figure: exact-match rate against
`lambda`, one line per program, directly labelled; second panel showing
maximum numerical error against `lambda` on a log axis, with the fp32 overflow
boundary marked.

**Pass condition.** Every program at 1.0 exact match for at least one
`lambda`, and `chosen_lambda` inside that region for all nine simultaneously.

**Compute.** CPU, minutes.

**Result.** *(fill in: chosen scale, which programs needed the most, where fp32
gave out)*

---

## E2 — `attribution_error.py`

**Question.** How wrong is a gradient approximation to a patching effect, when
the true effect is known?

**Hypothesis.** Median relative error will look reassuring and the tail will
not. Specifically: (a) errors concentrate on large-effect nodes, so the summary
correlation overstates usefulness; (b) sign errors appear on edges whose two
paths partly cancel; (c) `two_hop` is worse than the single-aggregate programs;
(d) softmax recomputation helps on `q`/`k`/`scores` sites and does nothing
elsewhere; (e) integrated gradients' error falls roughly as `1/n` until it
hits the exact-patching floor.

**Oracle.** Exact patching on compiled models, where the effect is a
re-measured forward pass and not an estimate at all.

**Protocol.**

* Programs: all nine, reported separately; `two_hop` and `most_freq` are the
  compositional cases and get their own panel.
* For each program: 256 clean/corrupt input pairs, non-degenerate by §3.1.
* Nodes: every head and MLP at every position. Edges: every port-level edge in
  the model's graph. For the compiled models this is small enough to enumerate
  exhaustively, which is the other advantage of this regime.
* Estimators: exact patching (ground truth), attribution patching, attribution
  patching + softmax recomputation, + gradient dropout, + both, and integrated
  gradients at `n in {2, 4, 8, 16, 32, 64}`.
* Metric: normalised logit difference, both directions (denoise and noise),
  reported separately — a method may be fine in one direction and not the
  other, and pooling them would hide it.

**Outputs.** `results/attribution_error.{json,png}`. JSON:
`per_program[program][estimator] -> {median_rel_err, p95_rel_err,
max_rel_err, sign_error_frac, spearman, pearson, forward_passes,
backward_passes, flops}`, plus `by_effect_decile` giving the error broken down
by the size of the true effect.

Figure, four panels: (1) estimated against true effect, log-log, with the
diagonal, coloured by site type; (2) relative error against true effect
magnitude — the panel that shows whether the method is wrong where it matters;
(3) sign-error rate by site type and by program; (4) error against compute, for
integrated-gradient step counts and exact patching, which is the frontier that
actually decides what to use.

**Pass condition.** The experiment cannot fail; it measures. It is invalid if
the linear-model test from phase 4 is not passing, since then the error is a
bug rather than curvature.

**Compute.** CPU, tens of minutes. The exhaustive edge enumeration is the cost
driver; `--quick` reduces to three programs and 32 pairs.

**Result.** *(fill in: the table in the README, plus which of (a)-(e) survived)*

---

## E3 — `ablation_choice.py`

**Question.** How much of a discovered circuit is an artefact of the chosen
counterfactual?

**Hypothesis.** Zero ablation recovers a *larger* circuit than the planted one
(it damages the model generally, so many nodes look important) and resample
ablation is closest to the truth. Mean ablation sits between them and is worse
than its popularity suggests.

**Oracle.** The planted graph from the compiler.

**Protocol.**

* Programs: all nine. Trained planted-algorithm models (induction, two-hop)
  included as a second row, since compiled models are off-distribution in a way
  that may flatter resampling.
* Counterfactuals: zero, mean over the prompt distribution, mean over
  positions (the common bug, included deliberately as a fourth arm to measure
  what it costs), resample, Gaussian at matched variance.
* Discovery: `discover` with `scorer="exact"`, threshold swept; the operating
  point for the table is the threshold that maximises F1 *per counterfactual*,
  so each is shown at its best rather than at a shared threshold that favours
  one.
* Report pairwise Jaccard between the recovered subgraphs.

**Outputs.** `results/ablation_choice.{json,png}`. Figure: grouped bars of
precision/recall per counterfactual per program, plus a Jaccard heatmap.

**Pass condition.** Measurement only. The write-up must answer explicitly
whether the choice changes the *conclusion* — not just the numbers — on at
least one program.

**Compute.** CPU, tens of minutes.

**Result.** *(fill in)*

---

## E4 — `discovery_roc.py`

**Question.** How well does greedy edge pruning recover a known circuit, and
is it better than ranking edges by their individual effect?

**Hypothesis.** The greedy procedure beats individual-effect ranking on
programs with redundancy (where two paths each look dispensable alone) and ties
with it elsewhere. Recall falls off sharply for edges whose individual effect
is below the noise floor of the metric.

**Oracle.** The planted graph.

**Protocol.**

* Programs: all nine, plus the two trained planted-algorithm models.
* Methods: greedy pruning with `scorer="exact"`; greedy with
  `scorer="attribution"` (the cheap version, whose degradation relative to
  exact is the useful number); individual-effect ranking at a matched edge
  count; and a random-subgraph baseline at matched size.
* Threshold swept over 20 log-spaced values; curves are precision against
  recall with the threshold as the parameter.
* Report the cost of each method in forward passes alongside its F1, since the
  cheap method being 90% as good at 1% of the cost is the decision-relevant
  fact.

**Outputs.** `results/discovery_roc.{json,png}`. Figure: PR curves per
program; second panel of best-F1 against forward passes across methods.

**Compute.** CPU, ~an hour at full scope.

**Result.** *(fill in)*

---

## E5 — `dictionary_recovery.py`

**Question.** Do sparse autoencoders recover a known feature basis, and does
the architecture ranking under the usual proxy metrics match the ranking under
actual recovery?

**Hypothesis.** The proxy ranking and the recovery ranking disagree. TopK and
BatchTopK will look best on the L0-MSE frontier; recovery will depend more on
sparsity and expansion than on architecture, and at high superposition
(`n/m` large, sparsity low) all four will recover poorly while still producing
respectable reconstruction loss. The gap between "reconstructs well" and
"recovered the features" is the result.

**Oracle.** The planted feature directions `W[:, i]` in `toy/superposition.py`.

**Protocol.**

* Toy model: `n in {32, 64, 128}` features, `m in {8, 16, 32}` hidden,
  sparsity `S in {0.7, 0.9, 0.99}`, importance uniform and decaying — a grid,
  reported as a grid, not averaged.
* Dictionaries: all four architectures, expansion `in {1, 2, 4, 8}`, sparsity
  knob swept to produce matched-L0 comparisons.
* Each configuration trained 3 times with different seeds; report the spread.
  A single-seed SAE comparison is noise.
* Metrics: MMCS both directions, Hungarian-matched mean cosine, L0, MSE,
  explained variance, dead-feature fraction under no remedy / resampling /
  AuxK.
* Second half: the same four architectures on GPT-2 small `blocks.6.resid_post`
  at expansion 8, reporting only the frontier metrics and `downstream_kl`,
  with the explicit note that no recovery number exists in this regime.

**Outputs.** `results/dictionary_recovery.{json,png}`. Figure: recovery against
sparsity with one line per architecture and a panel per `n/m`; a second figure
of the L0-MSE frontier; a third of recovery against MSE, which is the panel
that shows whether the proxy predicts the thing.

**Compute.** Toy grid: CPU, ~an hour. The GPT-2 half wants a GPU; at the CPU
tier run it at expansion 2 on 20M tokens and mark `scope` accordingly.

**Result.** *(fill in)*

---

## E6 — `grokking_circuit.py`

**Question.** Does the trained network implement the known closed-form
algorithm, and how much of its behaviour does that form account for?

**Hypothesis.** Explained variance above 90% for a handful of frequencies;
most MLP neurons assigned to a single frequency; and the circuit's analytic
signatures emerging before the test loss falls. The last part is the one most
likely to fail, and if it does the trajectory figure is the finding.

**Oracle.** The closed form, validated in phase 7 against synthetic weights
built to implement it exactly.

**Protocol.**

* `p = 113`, one layer, `H = 4`, `d = 128`, `dm = 512`, no LayerNorm, training
  fraction 0.3, AdamW `lr = 1e-3`, weight decay `1.0`, full batch, 25k steps,
  3 seeds.
* Checkpoint every 100 steps; run the full analysis on every checkpoint.
* Measurements per SPEC §5.2.
* Negative control: the same analysis applied to a network trained on a
  *random* permutation of the labels, which must show no frequency structure.
  Without this control, the Fourier structure could be an artefact of the
  analysis method applied to any smooth function on a cyclic group.

**Outputs.** `results/grokking_circuit.{json,png}`. Figure: train/test loss and
the circuit signatures on a shared time axis; a second figure of the logit
spectrum; a third of the neuron-frequency assignment histogram.

**Compute.** CPU-feasible (~30 min for 25k full-batch steps at this size),
GPU in a few minutes.

**Result.** *(fill in, including whether all 3 seeds found the same
frequencies — they need not, and that is interesting)*

---

## E7 — `gpt2_ioi.py`

**Question.** Applying the calibrated stack to a real model: what circuit does
it find, how does it compare to the published one, and what do the error bars
from E2–E4 imply about the answer?

**Hypothesis.** The main published heads are recovered; the disagreements will
be at the margin — small-effect heads whose inclusion depends on the threshold
and the counterfactual. Attribution patching will rank the top heads correctly
and misorder the tail, consistent with E2.

**Oracle.** None, and the write-up says so in those words. The comparisons
available are: an independently published circuit, internal consistency
between exact and approximate methods, and the faithfulness family — all
proxies. This section exists to make the difference between regimes 1-2 and
regime 3 unmistakable.

**Protocol.**

* 512 IOI prompt pairs from the generator, templates and name sets stated in
  the results JSON, non-degenerate pairs only.
* Metric: normalised logit difference between the indirect object and the
  subject.
* Discovery at three thresholds chosen from E4's F1-optimal region, under all
  three counterfactuals, both directions.
* Head-level comparison table against the published circuit, with every
  difference listed.
* Transcoders on all 12 MLPs, expansion 8; attribution graph for 32 prompts;
  report the error-node influence fraction and the intervention-agreement
  correlation (SPEC §4.5).

**Outputs.** `results/gpt2_ioi.{json,png}`. Figure: the recovered circuit as a
head-by-layer heatmap of effects, with the published circuit's heads outlined;
a second panel of faithfulness/completeness/minimality under each
counterfactual; a third of attribution-graph intervention agreement.

**Compute.** GPU strongly preferred. CPU tier: 128 prompts, exact patching at
head granularity only, transcoders skipped — and `scope` must say so.

**Result.** *(fill in, with the disagreements enumerated)*

---

## E8 — `steering_cost.py`

**Question.** What does a steering intervention cost in collateral damage per
unit of intended effect, against honest baselines?

**Hypothesis.** A dictionary feature direction beats the mean-difference
direction at low strength and converges to it at high strength, and a random
direction of matched norm accounts for a non-trivial share of the apparent
effect at the strengths usually demonstrated.

**Oracle.** The random-direction control at matched norm, and the unmodified
model's distribution on held-out text.

**Protocol.**

* Site: `blocks.6.resid_post` on GPT-2 small, positions `"all"` and `"last"`
  reported separately.
* Directions: a dictionary feature selected by its activation on target text;
  mean difference between two prompt sets; a trained linear probe; a random
  direction, 20 draws, matched `L2` norm.
* Strengths: 12 values log-spaced, covering the range from no measurable effect
  to visible degradation.
* `target_eval`: the probability of the intended continuation over 128 prompts.
  `control_eval`: mean KL from the unmodified model on 256 held-out sequences
  of unrelated text.

**Outputs.** `results/steering_cost.{json,png}`. Figure: intended effect
against collateral KL, one curve per direction type, with the random-direction
band shaded — a curve inside that band is not a result.

**Compute.** Minutes on GPU, under an hour on CPU.

**Result.** *(fill in)*

---

## Where the measurements disagreed

Collect here, as they are found, the places where a measurement contradicted
the expected answer — including expectations taken from the literature. Each
entry: what was expected, what was measured, and what was done about it.

This section being empty at the end of phase 10 means the experiments were not
sharp enough, not that everything worked.
