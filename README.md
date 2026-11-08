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

In regime 1 both sides of that equation are computable, so the error is not a
matter of opinion. The headline measurement reports, over the full set of
nodes and edges of compiled programs:

| quantity | attribution patching | integrated gradients (n=16) | exact patching |
|---|---|---|---|
| median relative error vs exact | `[[M1a]]` | `[[M1b]]` | 0 by definition |
| 95th percentile relative error | `[[M1c]]` | `[[M1d]]` | 0 |
| fraction of edges with the **wrong sign** | `[[M1e]]` | `[[M1f]]` | 0 |
| forward passes per circuit | `[[M1g]]` | `[[M1h]]` | `[[M1i]]` |

A sign error is the one that matters: it does not shrink a circuit, it puts
the wrong edge in it. The experiment reports where in the model sign errors
concentrate and what effect size they survive to — the honest version of
"attribution patching correlates well with patching", which is true on average
and false exactly where it is used.

*Measured by* [`experiments/attribution_error.py`](experiments/attribution_error.py).

---

## What the counterfactual choice is worth

Patching needs something to patch *in*. Zero ablation, mean ablation and
resample ablation are all in common use and they are not the same experiment:
zeroing an activation puts the network in a state it never occupies on any
input, so the behaviour it destroys may be nothing more than the network
objecting to being off-distribution.

Because regime 1 has a true circuit, the three can be scored rather than
argued about: run the same discovery procedure under each counterfactual and
compare the recovered subgraph to the planted one.

| counterfactual | edge precision | edge recall | Jaccard vs resample |
|---|---|---|---|
| zero ablation | `[[M2a]]` | `[[M2b]]` | `[[M2c]]` |
| mean ablation (over the prompt distribution) | `[[M2d]]` | `[[M2e]]` | `[[M2f]]` |
| resample ablation (matched counterfactual prompts) | `[[M2g]]` | `[[M2h]]` | 1 |

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
