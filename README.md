# circuitry — mechanistic interpretability, measured against circuits that are known

An interpretability result is a claim about what a network is *actually doing*
inside. The awkward part is that almost none of those claims can be
contradicted by anything. The usual evidence is a faithfulness score: delete
the proposed circuit, watch the behaviour fall over, conclude the circuit was
the cause. But that number has no scale. It does not say whether the circuit is
the right one, whether it is minimal, or whether a different method pointed at
the same model would have produced a different subgraph with an equally good
score. The field's standard tools are reported without error bars because
there is nothing to measure the error against.

This project builds the missing reference. Three regimes, in increasing order
of realism and decreasing order of certainty:

1. **Compiled transformers.** A small symbolic language is *compiled* into
   exact transformer weights. The resulting network's algorithm is known
   because it was written down first: which head implements which comparison,
   which MLP implements which map, which residual subspace carries which
   variable. The compiler's symbol table *is* the ground-truth circuit.
2. **Planted algorithms.** Networks trained from scratch on tasks whose
   minimal algorithm is known analytically — modular addition, sorting,
   bracket matching, two-hop lookup. Here the truth is a family of equivalent
   algorithms rather than one graph, and it is verified through closed-form
   signatures rather than by fiat.
3. **A real model.** GPT-2 small on indirect object identification. No ground
   truth exists here, which is the point: the error bars measured in regimes 1
   and 2 are what make the regime-3 numbers interpretable at all.

Every attribution method in the repository is implemented from first
principles and then **measured in regime 1 before it is trusted in regime 3**.

---

## The central measurement

Exact activation patching — actually re-running the forward pass with one
activation swapped — is the definition of a causal effect in these models. It
is also expensive: one forward pass per node, per edge, per prompt. So the
field mostly uses **attribution patching**, a first-order Taylor expansion
that estimates every patching effect at once from a single backward pass:

```
  effect(node) = L(x_clean with node <- a_corrupt) - L(x_clean)
               ~= (a_corrupt - a_clean) . dL/da |_clean        [first order]
```

The approximation is exactly as good as the metric is locally linear in that
activation, and the places where it breaks are precisely the interesting ones:
saturated softmaxes, LayerNorm's normalisation, and any node whose effect is
large — which is to say, the nodes that end up in the circuit.

Exact patching is not an estimate of that equation's left-hand side — it *is*
the left-hand side, re-measured. So the error is computable and not a matter
of opinion. The headline measurement, over seven families of site at every
position of transformers swept across depth:

| | first order | + softmax fix | + gradient dropout | integrated, n=16 | exact |
|---|---|---|---|---|---|
| median error | **0.0098** | **0.0056** | 0.0105 | 0.0006 | 0 by definition |
| worst error | **0.94** | 0.93 | 1.01 | 0.044 | 0 |
| **wrong sign** | **14.9%** | 13.3% | 13.7% | 0% | 0% |
| compute | 32 | 32 | 64 | 26,656 | 256 |

Errors are fractions of the largest true effect in that substrate, averaged
over six models spanning depth 2 to 6. An elementwise relative error would be
meaningless: most effects are near zero, and dividing by them gives a summary
dominated by the smallest number in the model. Sign errors are counted only
over effects above 5% of the largest, because the sign of a near-zero effect
is not a claim anybody makes. Compute is batch items processed, not forward
passes — integrated gradients batches its own interpolation steps, and a pass
count would make it look free.

Four things the measurement says that the usual summary statistics do not.

**The median is reassuring and the tail is not.** A median error of 1% of the
largest effect sits alongside a worst case of 94% of it, and a Spearman
correlation of 0.69 — the number usually quoted — alongside **14.9% of
decision-relevant effects coming out with the wrong sign.** A sign error does
not shrink a circuit; it puts the wrong component in it.

**The error is concentrated exactly where it does damage.** Broken down by the
size of the true effect, the median error rises monotonically from 0.0002 in
the smallest decile to 0.1426 in the largest — a factor of **700**. The
estimate is accurate about the components that do not matter and inaccurate
about the ones that do, which is the precise sense in which "attribution
patching correlates well with patching" is true on average and false where it
is used.

**It depends on how much you patch at once.** A first-order expansion is a
claim about small displacements. Patching one head at one position gives a
median error of 0.038 and 17.7% wrong signs; pooling that head's whole
contribution across every position gives **0.368 and 27.4%**. Nothing changed
but the size of the displacement.

**Integrated gradients is dominated by being exact.** Both need one evaluation
per index; integrated gradients needs *n* of them. At 16 steps it reaches a
median error of 0.0006 for 26,656 batch items, against exact patching's zero
error for **256** — 104 times the compute, and still not exact. Only the plain
first-order method buys anything, because its cost does not grow with the
number of indices at all.

### The two fixes, measured separately

**Softmax recomputation works, and costs nothing.** It cuts the median error
by 43% and the sign-error rate from 14.9% to 13.3%, with no extra forward or
backward pass, by recomputing the attention pattern exactly instead of
expanding through it. On the saturated-attention substrate, where the fix is
supposed to matter most, it is dramatic: the worst error on `attn.scores`
falls from **27.1 to 0.89**, a factor of thirty. A first-order estimate of a
saturated softmax can be wrong by twenty-seven times the largest real effect
in the model.

**Gradient dropout does not.** Taking the largest magnitude over per-layer
gradient cuts makes the median error *worse* (0.0105 against 0.0098) and the
worst case worse (1.01 against 0.94), buying a 1.2-point improvement in sign
errors for twice the compute. Combined with the softmax fix it gives the best
sign-error rate in the table, 11.4%, and still the worst maximum error. On
these models it is not worth its cost — reported here because it was expected
to help.

And one finding about the laboratory itself, which changed where this
experiment can run at all. A compiled model is exact *because* its lookup
tables are piecewise constant — and the derivative of a piecewise-constant
function is zero. The gradient inside a compiled model is identically zero at
**up to 67%** of its sites (`dyck1`, `hist` and `unique`; 0% only for the
shallowest program in the suite). Gradient methods cannot be measured there at
all: the "error" would be that artefact rather than curvature. So this
experiment runs on models with smooth nonlinearities, and regime 1 is reserved
for the questions that do not involve a gradient — which is what discovery is.

*Measured by* [`experiments/attribution_error.py`](experiments/attribution_error.py).

---

## What the counterfactual choice is worth

Patching needs something to patch *in*. Zero ablation, mean ablation and
resample ablation are all in common use and they are not the same experiment:
zeroing an activation puts the network in a state it never occupies on any
input, so the behaviour it destroys may be nothing more than the network
objecting to being off-distribution.

Because regime 1 has a true circuit, the choices can be scored rather than
argued about: run the same discovery procedure under each and compare the
subgraph that comes back. Eight programs, each counterfactual shown at its own
best threshold, averaged:

| counterfactual | precision | recall | edge F1 | ceiling | recall ÷ ceiling | edges kept |
|---|---|---|---|---|---|---|
| zero | 0.744 | 0.846 | **0.788** | 0.846 | **1.000** | 10.5 |
| mean, over the prompt distribution | **0.877** | 0.595 | 0.682 | 0.595 | **1.000** | 6.1 |
| mean, over positions *(the common bug)* | 0.698 | 0.722 | 0.698 | 0.802 | **0.875** | 8.2 |
| resample | **0.877** | 0.595 | 0.682 | 0.595 | **1.000** | 6.1 |
| Gaussian, matched scale | 0.586 | 0.860 | 0.677 | 0.860 | **1.000** | 16.6 |

The "ceiling" column is the fraction of planted edges that have *any* causal
signal on these prompts. A planted circuit is the program's whole dependency
graph; a given prompt sample exercises part of it, and an edge with no signal
cannot be found by any causal method. Reporting recall without that ceiling
makes a correct method look worse the more thorough the ground truth is.

Three findings, and the first reframes the question.

**The counterfactual does not change how well discovery works. It changes what
is findable.** Four of the five arms reach their own ceiling on **8 of 8
programs** — exactly, every time. The large spread in raw recall, from 0.595
to 0.860, is entirely a spread in the *ceiling*. Zero and Gaussian ablation
perturb harder, so more edges register as causally live; resample perturbs
least, so fewer do. None of them is better at discovery.

**Zero ablation wins on F1, and for a reason that should make you uneasy.** It
gets the best score against the planted graph (0.788) because putting the
network in a state it occupies on no input makes more of the circuit
measurable. That is the same property that makes it suspect: the extra signal
is off-distribution response, not necessarily circuit structure. Gaussian
ablation shows the cost of pushing that further — the highest ceiling of all
(0.860) and the worst precision (0.586), keeping 16.6 edges where resample
keeps 6.1.

**The deliberately-wrong arm is the only one that actually degrades
discovery.** Averaging over positions rather than over the prompt distribution
destroys positional information while looking like it preserved the
distribution. It is the single arm that fails to reach its ceiling — 0.875,
and 7 of 8 programs instead of 8 — so the common mistake does cost something
measurable, and it is the only thing here that does.

And the choice genuinely changes the answer. Pairwise agreement between the
recovered circuits runs from 0.442 (resample against Gaussian) to 1.000 —
**resample and mean-over-the-prompt-distribution produce the identical circuit
on every program**, which is worth knowing given how much ink is spent
choosing between them.

One caveat on this regime, measured: the metric's sampling noise floor is
**exactly zero** on every compiled program. The output is a one-hot argmax, so
every prompt pair gives the same normalised level and there is no spread to
bootstrap. That makes thresholds unusually easy to choose here and is another
way the compiled laboratory is cleaner than a real model, where the floor is
what sets the threshold.

*Measured by* [`experiments/ablation_choice.py`](experiments/ablation_choice.py).

---

## Do sparse dictionaries recover the features that are there?

A sparse autoencoder is supposed to undo superposition: to recover the
network's true, sparsely-active features from the dense directions they are
packed into. Whether it does is normally judged by reconstruction loss and L0,
neither of which is the thing being claimed.

In a toy model with a *known* feature basis — features planted by
construction, packed into fewer dimensions than there are features — recovery
is directly measurable: for each true feature, the cosine similarity to its
best match in the learned dictionary.

| architecture | mean max cosine to true features | L0 at matched MSE | dead features |
|---|---|---|---|
| ReLU + L1 | `[[M3a]]` | `[[M3b]]` | `[[M3c]]` |
| TopK | `[[M3d]]` | `[[M3e]]` | `[[M3f]]` |
| BatchTopK | `[[M3g]]` | `[[M3h]]` | `[[M3i]]` |
| JumpReLU (straight-through) | `[[M3j]]` | `[[M3k]]` | `[[M3l]]` |

The same comparison then runs on GPT-2 small residual streams, where no true
basis exists, and only the frontier quantities (L0 against reconstruction
loss, and downstream KL) can be reported. The gap between the two tables is
the honest measure of how much the toy result transfers.

*Measured by* [`experiments/dictionary_recovery.py`](experiments/dictionary_recovery.py).

---

## The analytic case: modular addition

A one-layer transformer trained on `(a + b) mod p` is the rare case where the
learned algorithm is known in closed form: the embedding becomes a set of
discrete Fourier modes, attention and the MLP compute products of cosines, and
the logit for answer `c` is a sum over a handful of key frequencies `k` of
`cos(w_k (a + b - c))`.

That makes it a test the project cannot talk its way out of. The closed form
is fitted to the trained network's own logits, and the fraction of variance it
explains is reported:

* variance of the trained logits explained by the closed form: `[[M4a]]`
* number of key frequencies found, and the fraction of total logit norm they
  carry: `[[M4b]]`
* fraction of MLP neurons assigned to a single frequency: `[[M4c]]`
* edge precision/recall of automated discovery against the analytic circuit:
  `[[M4d]]`

*Measured by* [`experiments/grokking_circuit.py`](experiments/grokking_circuit.py).

---

## And then the real model

Only after the stack has error bars does it get pointed at GPT-2 small. The
task is indirect object identification — "When Mary and John went to the
store, John gave a drink to ___" — chosen because an independently published
circuit exists to disagree with.

What gets reported, with the regime-1 error bars attached:

* faithfulness, completeness and minimality of the recovered subgraph, each
  under all three counterfactuals: `[[M5a]]`
* agreement with the published circuit, head by head, and every disagreement
  listed rather than summarised: `[[M5b]]`
* the same circuit recovered by exact patching and by attribution patching,
  and the measured difference between them: `[[M5c]]`
* an attribution graph over transcoder features for the same prompts, with the
  fraction of the model's effect it accounts for: `[[M5d]]`

*Measured by* [`experiments/gpt2_ioi.py`](experiments/gpt2_ioi.py).

---

## Steering, and what it costs

A feature direction found in the dictionary can be added back into the
residual stream to push behaviour around. Single-example demonstrations of
this are easy and nearly meaningless. The measurement that matters is the
trade: how much of the intended effect per unit of collateral damage, where
damage is the KL divergence from the unmodified model on text the steer has no
business touching.

* intended-effect / collateral-KL frontier across steering strengths: `[[M6a]]`
* strength at which the model's loss on unrelated text degrades by 1%: `[[M6b]]`
* comparison against the obvious baselines — a mean-difference direction and a
  trained linear probe — on the same axes: `[[M6c]]`

*Measured by* [`experiments/steering_cost.py`](experiments/steering_cost.py).

---

## Layout

```
src/circuitry/
  model/       a from-scratch GPT-2-compatible transformer whose every
               activation site is addressable, readable and writable
  symbolic/    the program language, the compiler that turns a program into
               exact weights, and the ground-truth graph it emits
  attrib/      direct logit attribution, exact patching, path patching,
               gradient approximations, automated discovery, and the metrics
  dictlearn/   sparse autoencoders, transcoders, their training and the
               recovery evaluation; attribution graphs over features
  toy/         the superposition model with a known feature basis, modular
               addition, and the algorithmic task suite
  steer.py     interventions and their cost curves
experiments/   one script per measurement above; writes results/*.json + *.png
benchmarks/    the forward-pass cross-check against the reference implementation
examples/      short, runnable, public-API-only walkthroughs
docs/          SPEC.md, BUILD_PLAN.md, EXPERIMENTS.md, ENGINEERING.md
prompts/       the staged build transcript
```

## Running it

```bash
pip install -e ".[dev,plot,oracle]"
pytest tests/ -q
```

```bash
PYTHONPATH=experiments python experiments/attribution_error.py
```

Every experiment takes `--quick` for a short run that exercises the whole
pipeline without producing a publishable number; CI runs all of them that way
on every push, so no figure in this README can go stale without the build
going red.

## A note on method

No number appears in this README until the experiment that produces it has
been run; until then it is a bracketed placeholder. One of those is
embarrassing in
a way that a plausible invented number is not, which is the only reason the
convention holds.

The repository's one real commitment is that an interpretability claim should
be checkable against something that could contradict it — a compiler's symbol
table for a discovered circuit, a known feature basis for a dictionary, a
closed form for a trained network's logits, an independently written
transformer for a transformer. Where a measurement disagreed with the method
everyone uses, the disagreement is in the write-up, not smoothed out of it.

## License

MIT
