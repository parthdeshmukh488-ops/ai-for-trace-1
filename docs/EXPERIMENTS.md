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

**Result.** All nine programs reach exact agreement with the interpreter, and
a single shared scale of **40** is sufficient for all of them.

| quantity | measured |
|---|---|
| programs exact at every scale from 4 upward | 8 of 9 |
| the exception, and the scale it needs | `pair_balance`, 32 |
| error decay with scale (`pair_balance`) | `exp(-0.51 * lambda)` |
| predicted decay from the half-score BOS path | `exp(-0.50 * lambda)` |
| float32 worst error at scale 100 | 3.3e-8 (`pair_balance`); 0 for the other eight |
| float64 worst error at scale 100 | 1.1e-16 |

All three parts of the hypothesis survived, and the second was sharper than
expected:

* There is a threshold above which every program is exactly right, and it is
  40 with margin.
* The failures do concentrate on the empty-selection path, and the reason is
  quantitative: the beginning-of-sequence position sits at *half* score so
  that it loses to any match and beats any non-match, which makes its residual
  weight `exp(-lambda/2)` rather than `exp(-lambda)`. The measured decay of
  0.51 against a predicted 0.50 confirms it, and the analytic threshold of
  27.6 brackets the measured fail-at-24 / pass-at-32 exactly.
* What actually binds is the *read-out*, not the attention. Eight programs end
  in a categorical argmax, which tolerates a 1e-7 perturbation without
  noticing; the ninth reads a magnitude back and is compared at 1e-6, so it
  sets the scale for the whole suite.

**Where the protocol was wrong.** It predicted float32 would overflow and
bound the scale from above. It does not — softmax subtracts the row maximum
before exponentiating, so there is nothing to overflow, and float32 is exactly
right for every program at every scale tested. The only float32 cost is
precision on the one magnitude read-out. Compiled models remain float64, for a
smaller reason than the specification gave.

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

**Oracle.** Exact patching, which is the definition of the effect rather than
an estimate of it, and is therefore an oracle on any substrate.

**Substrate — changed from the specification, for a reason that matters.** The
protocol originally named the compiled programs, on the grounds that the
ground-truth circuit is known there. That is the wrong substrate, and the
reason is a fact about the whole compiled approach: a compiled model is exact
*because* its lookup tables are piecewise constant, and the derivative of a
piecewise-constant function is zero. The gradient inside a compiled model is
identically zero at between a third and two thirds of its sites, rising with
depth. Every gradient estimate there is exactly 0 against a large true effect,
so the measured "error" would be that artefact and not curvature.

So this experiment runs on models with smooth nonlinearities, and the
dead-gradient fractions are reported as their own arm so the exclusion is on
the record. Compiled models remain the right substrate for discovery, where no
gradient is involved.

**Protocol.**

* Substrates: random transformers at depth 2, 4 and 8, three seeds each, with
  depth as the available proxy for composition depth; plus a
  saturated-attention variant, built by scaling the query and key projections,
  which is the regime the softmax fix exists for. Reported per substrate, never
  pooled into one number.
* 8 clean/corrupt pairs per substrate, non-degenerate by §3.1.
* Sites: seven families — `attn.z`, `attn.q`, `attn.k`, `attn.scores`,
  `attn.pattern`, `mlp.out`, `resid_post` — at every position, reported by
  family so the answer says which *kind* of site a method fails on.
* Estimators: exact patching (ground truth), first-order attribution, first
  order + softmax recomputation, + gradient dropout, + both, and integrated
  gradients at `n in {2, 4, 8, 16, 32}`.
* Metric: normalised logit difference, both directions, reported separately.
* Error as absolute error and as a fraction of the largest true effect in that
  substrate. Elementwise relative error is *not* reported: most effects are
  near zero and dividing by them gives ratios that say nothing. Sign errors are
  counted only over effects above 5% of the largest.
* Compute in **batch items processed**, not forward passes: integrated
  gradients batches its own interpolation steps, so a pass count would make it
  look free.

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

**Result.** Averaged over six smooth substrates (depth 2, 4 and 6; two seeds
each), denoising, errors as fractions of the largest true effect:

| | first order | + softmax | + dropout | + both | IG n=4 | IG n=16 | exact |
|---|---|---|---|---|---|---|---|
| median error | 0.0098 | 0.0056 | 0.0105 | 0.0089 | 0.0023 | 0.0006 | 0 |
| worst error | 0.94 | 0.93 | 1.01 | 1.01 | 0.18 | 0.044 | 0 |
| wrong sign | 14.9% | 13.3% | 13.7% | 11.4% | 0.5% | 0% | 0% |
| Spearman | 0.693 | 0.735 | 0.702 | 0.740 | 0.969 | 0.992 | 1 |
| batch items | 32 | 32 | 64 | 64 | 6,688 | 26,656 | 256 |

Error by decile of the true effect's size (first order, deepest substrate):

| decile | 1 | 3 | 5 | 7 | 9 (largest) |
|---|---|---|---|---|---|
| mean \|true\| | 0.0005 | 0.0055 | 0.0160 | 0.0715 | 0.3826 |
| median error | 0.0002 | 0.0028 | 0.0088 | 0.0273 | **0.1426** |

Wall time 69 seconds on four CPU cores.

**The hypotheses.**

*(a) Errors concentrate on large-effect nodes, so the summary correlation
overstates usefulness.* **Survived, decisively.** The median error rises
monotonically across every decile, from 0.0002 to 0.1426 — a factor of 700.
The Spearman correlation of 0.693 is the number usually quoted and it coexists
with 14.9% of decision-relevant effects having the wrong sign. The same point
again from a different angle: pooling a head's contribution over all positions
instead of patching one position at a time takes the median error from 0.038
to 0.368 and the sign-error rate from 17.7% to 27.4%, with nothing changed but
the size of the displacement.

*(b) Sign errors appear where two paths partly cancel.* **Not tested as
stated**, and weak evidence against. The measurement here is over nodes, not
edges, so the claim as written was not put at risk. What was tested is the
probe designed to detect that cancellation — gradient dropout — and it made
the estimate worse rather than better, which is not what one would expect if
cancellation were the dominant mechanism.

*(c) `two_hop` is worse than the single-aggregate programs.* **Void.**
Compiled models turned out to be an impossible substrate for this experiment
(see the substrate note above), so there are no per-program numbers. Depth is
the substitute axis, and the error does grow with it.

*(d) Softmax recomputation helps on `q`/`k`/`scores` and nothing elsewhere.*
**Survived, both halves, and the first half is dramatic.** Overall it cuts the
median error by 43% and the sign-error rate from 14.9% to 13.3% at no cost at
all. On the saturated-attention substrate, where the fix is meant to matter,
the worst error on `attn.scores` falls from **27.1 to 0.89** — a factor of
thirty. A first-order estimate of a saturated softmax can be wrong by
twenty-seven times the largest real effect in the model, and recomputing one
softmax fixes it. The second half holds too: on `attn.z` and `mlp.out` the fix
changes the estimate by less than 1e-12, which is a test rather than a
measurement.

*(e) Integrated gradients' error falls as 1/n until it hits a floor.* **The
rate survived; the premise did not.** The error falls as exactly 1/n — measured
ratios of 2.29, 2.08, 2.04, 2.02, 2.01, 2.01, 2.00 for each doubling — and
there is no floor, because it converges to the exact effect. The surprise is on
the other axis: both methods need one evaluation per index and integrated
gradients needs *n* of them, so it is **strictly dominated by exact patching**
— 104 times the compute at n=16, and still not exact. Only the plain
first-order method buys anything, since its cost does not grow with the number
of indices at all. This was not anticipated by the hypothesis and is arguably
the most decision-relevant line in the table.

**An unanticipated finding: gradient dropout does not help.** Taking the
largest magnitude over per-layer gradient cuts makes the median error worse
(0.0105 against 0.0098) and the worst case worse (1.01 against 0.94), in
exchange for 1.2 points of sign-error rate and twice the compute. Combined
with the softmax fix it gives the best sign-error rate in the table (11.4%)
and still the worst maximum error. It is reported because it was expected to
help; on these models it is not worth its cost.

A bug worth recording, because it would have produced a false negative rather
than a wrong number: the first implementation skipped the gradient cut whenever
the site being cut was also a site being measured, which on a full sweep is
every site — so the method silently did nothing while appearing to run. It was
found by reading the code rather than by any test, and the fix changed the
measured outcome from "no effect" to "a small harm".

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

### Phase 3 — edge effects do not sum to node effects, and the compiled regime cannot show it

*Expected:* the build plan asked for one number — the discrepancy between the
sum of a component's edge effects and its node effect — and expected it to be
a modest remainder from the nonlinearities.

*Measured:* it is not modest. On an untrained model, whose softmaxes and MLP
activations are nowhere near saturated, one head with four outgoing edges
gives:

| direction | node effect | sum of edges | discrepancy |
|---|---|---|---|
| denoise | −0.139 | +0.241 | +0.380, i.e. **273%** of the node effect |
| noise | −0.142 | −0.277 | −0.135, i.e. **95%** of the node effect |

The denoising row is the striking one: patching the head alone *hurts*, while
patching any single path from it *helps*. A decomposition into edges is not a
small perturbation of the node measurement — on this model the two disagree
about the sign.

*And the part that matters for the whole project:* **on a compiled model the
discrepancy is exactly 0.** It would be a serious error to report that as
evidence that edge decomposition is sound. Compiled lookup tables are
piecewise constant, so a patch either flips a switch or does nothing, and
every nonlinear interaction vanishes by construction. The compiled regime
therefore *cannot exhibit* this phenomenon, and any claim about edge
decomposition measured there carries no information about trained models.

*Done about it:* the number is reported in both regimes, with the caveat
stated wherever it appears; `tests/test_paths.py` contains both measurements
and the docstring of the compiled one says in those words that it is a caveat
on the regime rather than a result about decomposition. Phase 5's
planted-algorithm arm — trained models, unsaturated — is where the honest
version of this measurement lives.

*A second finding, from the same work:* the level/effect distinction. The
normalised metric runs 0-is-corrupt to 1-is-clean in both directions, so the
*unpatched* run reads 0 when denoising and **1** when noising. Reporting the
level as the effect therefore makes a noising patch that destroys nothing look
maximally important, and summing such numbers over a component's edges gives
an answer larger than the whole model — which is exactly what the first
version of this measurement did, giving a "sum of edges" of 3.72 against a
node effect of 0.86. `effect_of` now exists solely to make that mistake hard.

### Phase 1 — an absolute tolerance was the wrong unit, and folding costs precision

*Expected:* the forward-pass cross-check would agree with the reference to
better than 1e-4 absolute, as SPEC §1.7 specified, and folding would make no
difference to that.

*Measured:* neither. On real GPT-2 the deviation is 4.9e-4 absolute, which
sounds like a failure and is not: the residual stream there has an RMS of 41
and entries above 3000, so 4.9e-4 is 1.19e-5 *relative*, about 100x float32
epsilon and exactly what twelve layers of accumulation produce. The bound was
in the wrong unit. And folding, which is algebraically exact, is **3.3x worse
conditioned in float32** — 7.7e-5 relative against 2.3e-5 unfolded — because
absorbing the LayerNorm gain into `W_Q` changes the order the products
accumulate in.

*Done about it:* the bound is now relative to each site's own RMS, and the
exactness claim is checked in float64 where it can actually be seen (6.4e-14
on real GPT-2 — the identity holds). The float32 folded figure is reported
rather than bounded, because it measures conditioning rather than correctness.

*Why it matters downstream:* every attribution number in this repository is
computed on a folded model, so the 3.3x is a floor under the precision of all
of them. Phase 4 measures attribution errors in the 1e-2 range, so it is not
binding there — but it is worth knowing, and it is the reason the compiled
models in phase 2 are kept in float64.
