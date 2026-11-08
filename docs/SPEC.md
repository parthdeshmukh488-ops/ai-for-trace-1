# circuitry — technical specification

This document is the contract. Where it names a function signature, a tensor
shape, a loss or a metric, implement exactly that; where it leaves a choice
open it says so. Revisions go at the bottom.

Notation throughout: `B` batch, `T` sequence position, `H` heads, `L` layers,
`d` model width, `dh` head width (`d = H * dh`), `dm` MLP width, `V` vocab.

---

## 0. Package layout and public surface

```
src/circuitry/
  __init__.py            re-exports the flat public API
  model/
    config.py            ModelConfig
    transformer.py       Transformer, Block, Attention, MLP, LayerNorm
    hooks.py             HookPoint, HookRegistry, intervention contexts
    sites.py             the activation-site naming scheme and its parser
    loading.py           external checkpoint import, weight folding
  symbolic/
    lang.py              SOp, Select, Aggregate, Map, SequenceMap, ...
    bases.py             VariableSpace, BasisAllocator
    compiler.py          compile(program, vocab, max_len) -> CompiledModel
    truth.py             GroundTruthGraph
  attrib/
    metric.py            Metric protocol, logit_diff, kl, loss
    dla.py               direct logit attribution, logit lens
    patching.py          exact node patching, counterfactual families
    paths.py             path patching, edge effects
    gradients.py         attribution patching, integrated gradients, AtP*
    acdc.py              greedy edge pruning
    circuit.py           Circuit (a subgraph), set algebra, serialisation
    metrics.py           faithfulness, completeness, minimality, edge PR
  dictlearn/
    sae.py               ReluSae, TopKSae, BatchTopKSae, JumpReluSae
    transcoder.py        Transcoder, CrossLayerTranscoder
    train.py             the training loop, activation streaming, schedules
    evaluate.py          recovery, frontier, dead-feature and KL metrics
    graph.py             attribution graphs with error nodes
  toy/
    superposition.py     the known-feature-basis model
    modadd.py            modular addition: training + the analytic reference
    tasks.py             the algorithmic task suite and its prompt generators
  steer.py               interventions and their cost curves
  util/
    seeding.py           explicit generators, determinism helpers
    flops.py             forward/backward cost accounting
    store.py             activation caches with an explicit memory budget
```

`circuitry.__init__` re-exports, and nothing else: `ModelConfig`,
`Transformer`, `load_gpt2`, `sites`, `patch`, `path_patch`, `attribute`,
`discover`, `Circuit`, `compile_program`, `train_sae`, `steer`.

---

## 1. The transformer engine

### 1.1 Why from scratch

An interpretability engine needs to do three things a standard implementation
will not: expose every intermediate tensor under a stable name, allow any of
them to be *replaced* mid-forward without recomputing what came before it, and
decompose the residual stream into per-component contributions. Wrapping an
existing implementation gets you the first of those and fights you on the
other two. Writing the forward pass means the decomposition is the natural
thing and the hook points are not bolted on.

The cost is that correctness is now your problem, which is what
`benchmarks/cross_check_reference.py` is for (§1.7).

### 1.2 `ModelConfig`

```python
@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    d_head: int | None = None       # defaults to d_model // n_heads
    act_fn: str = "gelu_tanh"       # "gelu_tanh" | "gelu_exact" | "relu" | "solu"
    ln_eps: float = 1e-5
    use_ln: bool = True
    positional: str = "learned"     # "learned" | "none" | "sinusoidal"
    tie_unembed: bool = True
    dtype: str = "float32"
```

`gelu_tanh` is the approximation the GPT-2 checkpoints were trained with, so it
is the default; `gelu_exact` differs from it by up to ~1e-3 in activation
space, which is above the cross-check tolerance and will fail §1.7 if chosen
by mistake. State this in the docstring — it is the first thing that goes
wrong when importing external weights.

### 1.3 Forward pass

Pre-norm, GPT-2 shaped:

```
  resid_0      = W_E[tokens] + W_pos[:T]
  for l in 0..L-1:
      x        = resid_l
      n1       = LN(x)                        scale s1 = 1/sqrt(var + eps)
      q,k,v    = n1 W_Q^l, n1 W_K^l, n1 W_V^l       each [B,T,H,dh]
      scores   = q k^T / sqrt(dh)  + causal_mask    [B,H,T,T]
      pattern  = softmax(scores, dim=-1)
      z        = pattern v                          [B,T,H,dh]
      attn_out = sum_h z[:,:,h] W_O^l[h]            [B,T,d]
      resid_m  = x + attn_out
      n2       = LN(resid_m)                   scale s2
      pre      = n2 W_in^l + b_in^l                 [B,T,dm]
      post     = act(pre)
      mlp_out  = post W_out^l + b_out^l
      resid_l+1= resid_m + mlp_out
  logits       = LN_f(resid_L) W_U + b_U
```

The causal mask is `-inf` above the diagonal, applied **before** softmax and
implemented by `masked_fill` on a cached boolean, not by adding a large
negative constant: with a constant, `scores` at masked positions is a number
the hook system can be asked to patch, and patching a masked position must be
a no-op rather than a quiet behaviour change.

Per-head attention output `attn.result` of shape `[B,T,H,d]` is what lets the
residual stream be decomposed per head, and it costs `H` times the memory of
`attn.out`. It is computed only when a hook or a cache request names it;
default off. This is a documented performance cliff, not an implementation
detail to discover later.

### 1.4 Activation sites

Site names are strings with a fixed grammar, parsed by `model/sites.py`:

```
  embed                       [B,T,d]
  pos_embed                   [B,T,d]
  blocks.{l}.resid_pre        [B,T,d]
  blocks.{l}.ln1.scale        [B,T,1]
  blocks.{l}.ln1.normalized   [B,T,d]
  blocks.{l}.attn.q           [B,T,H,dh]
  blocks.{l}.attn.k           [B,T,H,dh]
  blocks.{l}.attn.v           [B,T,H,dh]
  blocks.{l}.attn.scores      [B,H,T,T]    pre-softmax, post-mask
  blocks.{l}.attn.pattern     [B,H,T,T]    post-softmax
  blocks.{l}.attn.z           [B,T,H,dh]
  blocks.{l}.attn.result      [B,T,H,d]    opt-in, H x memory
  blocks.{l}.attn.out         [B,T,d]
  blocks.{l}.resid_mid        [B,T,d]
  blocks.{l}.ln2.scale        [B,T,1]
  blocks.{l}.ln2.normalized   [B,T,d]
  blocks.{l}.mlp.pre          [B,T,dm]
  blocks.{l}.mlp.post         [B,T,dm]
  blocks.{l}.mlp.out          [B,T,d]
  blocks.{l}.resid_post       [B,T,d]
  ln_final.scale              [B,T,1]
  ln_final.normalized         [B,T,d]
  logits                      [B,T,V]
```

`sites.resolve(pattern)` expands a glob (`"blocks.*.attn.pattern"`,
`"blocks.0-3.mlp.post"`) to a sorted list of concrete names, and raises on a
pattern that matches nothing — a silent empty match is how an experiment ends
up measuring the empty set and reporting it as a null result.

A site has an **index space**: `resid_post` is indexed by position, `attn.z` by
`(position, head)`, `scores` by `(head, query, key)`. `sites.index_space(name)`
returns the tuple of axis names, which is what lets patching iterate over
"every head at every position" generically rather than with a special case per
site.

### 1.5 Hooks

```python
class HookPoint:
    def __call__(self, tensor: Tensor) -> Tensor: ...

@contextmanager
def intervene(model, hooks: Mapping[str, Callable]) -> Iterator[None]: ...

@contextmanager
def capture(model, sites: Sequence[str], *, budget_bytes: int | None = None)
        -> Iterator[ActivationStore]: ...
```

Hook functions receive the tensor and return a replacement; returning `None`
means "observe only". Four prebuilt combinators cover nearly every use:

```python
replace(value)                  # substitute wholesale
replace_at(index, value)        # substitute at an index-space slice
add(vector, *, scale=1.0)       # resid-stream steering
project_out(directions)         # ablate a subspace
freeze(store)                   # pin to a cached value (path patching)
```

Contract (R6 in the agent brief): leaving an `intervene` block restores the
model to a state that produces bitwise-identical logits to before. Implement
by registering and removing handles, never by mutating parameters. The test is
`tests/test_hooks.py::test_intervention_is_exactly_reversible` and it compares
`torch.equal`, not `allclose`.

`capture` with a `budget_bytes` raises `ActivationBudgetExceeded` naming the
sites and the overage rather than letting the process be killed by the OOM
reaper — `attn.result` across all layers of GPT-2 small at `T=512` is already
several gigabytes.

### 1.6 Weight folding

Three transformations make the residual stream analysable. All three are
*exactly* behaviour-preserving, and there is a test for each asserting
identical logits before and after:

**Fold LayerNorm gain and bias.** `LN(x) = s(x) * (x - mean(x)) * gamma + beta`
followed by a linear `W, b`. Since `gamma` is a diagonal scaling applied before
`W`, set `W' = diag(gamma) W` and `b' = b + beta W`. What remains of the
LayerNorm is centring and the scalar `s(x)`. After folding, every weight that
reads the residual stream reads it directly, and direct logit attribution is
exact up to the single scalar `s`.

**Centre writing weights.** LayerNorm removes the residual stream's mean
component, so the mean of any column written into the stream is unobservable
downstream. Subtracting it from `W_O`, `W_out`, `W_E`, `W_pos` and the biases
changes nothing about the model and removes a meaningless degree of freedom
from every decomposition. The test asserts logits unchanged.

**Centre the unembedding.** Adding a constant to every logit does not change
the softmax, so `W_U` may be centred over the vocabulary axis. This makes
logit attributions comparable across prompts.

`load_gpt2(..., fold_ln=True, center_writing_weights=True, center_unembed=True)`
defaults all three on, and the docstring says what each one buys and that each
is exact.

### 1.7 The independent cross-check

`benchmarks/cross_check_reference.py` loads the same checkpoint into
`circuitry.Transformer` and into an independently written implementation, runs
both on the same random and natural inputs, and compares:

1. final logits,
2. every `resid_post` (against the reference's hidden states),
3. every `attn.pattern` (against the reference's attention outputs).

Report the achieved maximum absolute and maximum relative deviation per site
and the position in the network where it is largest. Bitwise equality will not
hold — matmul reduction order differs — so the measurement is the deviation,
and the pass condition is `max_abs < 1e-4` in float32 with the *achieved*
figure written into `benchmarks/results/cross_check.json`. If the deviation
grows monotonically with depth at the rate accumulation predicts, that is the
expected signature; a jump at one layer is a bug, and the script should say
which layer.

Run the comparison with folding both off and on. Off proves the forward pass
is right; on proves the folds are exact.

---

## 2. The symbolic compiler

### 2.1 What it is for

This is the component that makes the rest of the repository falsifiable. A
program is written in a small sequence-processing language, compiled to exact
transformer weights, and the compiler emits the computational graph it built.
Discovery methods are then scored against that graph. Unused heads and MLPs in
the compiled model are genuine true negatives, which is what makes precision
well-defined rather than a matter of where you draw the boundary.

### 2.2 The language

Values are *sequence operations* (SOps): functions from an input sequence of
length `T` to a sequence of length `T`. Primitives:

```python
tokens                  # SOp: the input symbols
indices                 # SOp: 0, 1, ..., T-1
Map(f, x)               # elementwise f applied to x
SequenceMap(f, x, y)    # elementwise binary f over x and y
Select(k, q, pred)      # Selector: a [T,T] boolean matrix, pred(k[j], q[i])
Aggregate(sel, x)       # SOp: mean of x over positions selected per query
SelectorWidth(sel)      # SOp: number of positions selected per query
```

`pred` is one of `EQ, NEQ, LT, LEQ, GT, GEQ, TRUE, FALSE`. Every SOp carries a
declared **encoding**:

* `categorical` — the value set is finite and known at compile time; it is
  represented as a one-hot over `|values|` residual dimensions.
* `numerical` — a real magnitude in one dimension.

`Aggregate` over a categorical SOp is only well-defined when the selector
selects exactly one position per query (a mean of one-hots is otherwise a
distribution, not a value); the compiler must *check* this over the whole
input space and raise `IllDefinedAggregate` naming the input that breaks it.
This check is the difference between a compiler and a thing that silently
produces a model that computes something else.

Version 1 supports categorical throughout plus numerical `Aggregate` and
`SelectorWidth`. Numerical `SequenceMap` is a documented non-goal for v1.

### 2.3 Basis allocation

`bases.BasisAllocator` assigns each SOp a disjoint block of residual
dimensions:

```python
@dataclass(frozen=True)
class VariableSpace:
    name: str
    encoding: Literal["categorical", "numerical"]
    values: tuple[object, ...] | None     # categorical only
    start: int
    width: int                            # len(values), or 1
```

The residual stream is the direct sum of these blocks plus one dimension for a
BOS indicator. Width is `sum(widths)`, which is the compiled model's `d_model`
— so compiled models are small and exact rather than padded to a round number.

The allocation table is the ground truth's backbone: it says which dimensions
mean what, which is a stronger statement than any real model admits and
exactly why this regime is useful.

### 2.4 Compiling each construct

**`Select` + `Aggregate` → one attention head.** Let `k` be categorical with
space `K` and `q` categorical with space `Q`, and let `x` be the aggregated
SOp with space `X`.

* `W_K` reads `K`'s block into `dh` dimensions as the identity on that block.
* `W_Q` reads `Q`'s block.
* The bilinear form `W_Q W_K^T` is set so that `scores[i,j] = lambda` when
  `pred(k[j], q[i])` is true and `0` otherwise. Concretely: build the
  `|Q| x |K|` predicate matrix `P`, and set `W_Q W_K^T = lambda * P` restricted
  to the relevant blocks. Because both are one-hot reads, the product picks out
  exactly `P[q_i, k_j]`.
* `W_V` reads `X`'s block; `W_O` writes to the output SOp's block.
* The BOS position is always selected with score `0`, giving the head somewhere
  to put its attention when nothing matches — without it, softmax over an
  all-zero row attends uniformly to the whole sequence and the aggregate is
  wrong rather than empty.

With `n` positions selected and `lambda` finite, the pattern puts
`e^lambda / (n e^lambda + m)` on each selected position where `m` counts the
unselected (including BOS). The aggregate is therefore the intended mean plus
an error of order `m e^{-lambda} / n`. `compiler_fidelity.py` (§E1) sweeps
`lambda` and measures the exact-match rate; the default is chosen from that
sweep and its comment cites the measured number.

**`Map` / `SequenceMap` → one MLP.** The input is categorical with a finite
domain `D` (for `SequenceMap`, the product of two domains, which is why the
compiler must bound it and raise if `|D|` exceeds a configured limit). The MLP
is a lookup table:

* `W_in` has one neuron per element `d in D`, whose weights are the one-hot
  pattern of `d` and whose bias is chosen so the pre-activation is positive
  only for exactly that `d`. For a single categorical input of one-hot height
  1, `bias = -0.5` separates cleanly; for a product of two one-hots the
  pre-activation is `2` for the matching pair and `<= 1` otherwise, so
  `bias = -1.5`.
* After the activation, exactly one neuron is above zero. Because the chosen
  activation is not a step function, its value at the firing input must be
  measured and divided out: `W_out[d] = onehot(f(d)) / act(margin)`. Do this
  numerically from the actual activation function rather than from an
  analytic value, so changing `act_fn` cannot silently break the compiler.
* The old block's output must also be *cleared* if the SOp overwrites a
  variable; the compiler's convention is that every SOp gets a fresh block and
  nothing is ever overwritten, which costs width and buys the property that no
  compiled component has a side effect.

**`SelectorWidth` → one head plus one MLP.** Attend uniformly over the selected
set together with BOS; the attention weight on BOS is `1/(n+1)`, so reading the
BOS indicator through `W_V` and `W_O` yields `1/(n+1)` in a numerical
dimension, and an MLP inverts it to `n`. The inversion over the finite range
`0..T` is another lookup table.

### 2.5 The ground-truth graph

```python
@dataclass(frozen=True)
class GroundTruthGraph:
    nodes: tuple[Node, ...]          # (site, index) that the program uses
    edges: tuple[tuple[Node, Node], ...]
    variables: tuple[VariableSpace, ...]
    unused: tuple[Node, ...]         # true negatives
    program_repr: str
```

A `Node` is a `(site, index)` pair in the engine's own naming scheme, so a
discovered circuit and the ground truth are directly comparable without a
translation layer. Edges come from the compiler's data dependencies: if SOp `u`
is an input to SOp `v`, there is an edge from `u`'s writing component to `v`'s
reading component.

Three honest caveats the docs must state, because they bound every claim made
in this regime:

1. The compiled model is not a trained model. Its weights are one-hot and its
   attention is near-hard; methods that exploit smoothness may do better or
   worse here than on a trained network. §E4 therefore also runs on *trained*
   planted-algorithm models (§5), and the two are reported side by side.
2. The ground-truth graph is the graph of the *program*. A compiled model may
   admit a second, equivalent computation through unused capacity; the
   `unused` set is verified empty of causal influence by ablating all of it at
   once and asserting the output is unchanged.
3. `pred=TRUE` selectors and degenerate programs compile to models with
   trivially recoverable circuits. The program suite (§2.6) is required to
   span a stated range of depth and fan-in, and the results are reported *per
   program*, never pooled into one headline number that a few easy programs
   could carry.

### 2.6 The program suite

At least these, with the stated minimum graph sizes, in `symbolic/programs.py`:

| program | what it computes | min depth | notes |
|---|---|---|---|
| `reverse` | reverses the sequence | 2 | index arithmetic + one aggregate |
| `hist` | count of each token's occurrences | 2 | `SelectorWidth` with `EQ` |
| `sort` | tokens in sorted order | 3 | rank by `SelectorWidth(LT)`, then gather |
| `most_freq` | tokens by descending frequency | 4 | composes `hist` and `sort` |
| `dyck1` | balanced-bracket validity prefix | 3 | numerical aggregate + threshold |
| `dyck2` | two bracket types | 4 | adds a mismatch detector |
| `shift` | sequence shifted by `k` | 2 | the easy control |
| `pair_balance` | running difference of two tokens | 2 | numerical path |
| `two_hop` | `x[x[i]]` | 3 | the composition case; two chained aggregates |

`two_hop` is the important one: it is the smallest program with a genuine
two-step composition, which is the structure that first-order attribution
methods are expected to get wrong, and §E2 reports it separately for that
reason.

---

## 3. Attribution

### 3.1 Metrics

```python
class Metric(Protocol):
    def __call__(self, logits: Tensor) -> Tensor: ...   # [B] -> scalar per item
```

Provided: `logit_diff(correct, incorrect)`, `logprob(correct)`,
`kl_from(reference_logits)`, `cross_entropy(targets)`.

Patching results are reported in **normalised** form:

```
  m_norm = (M(patched) - M(corrupt)) / (M(clean) - M(corrupt))
```

so `0` is corrupt behaviour and `1` is clean. The denominator must be checked
away from zero per item — a prompt pair where clean and corrupt score the same
has no signal and must be dropped at dataset construction, with the number
dropped reported. Dividing by a near-zero denominator is how a patching plot
acquires enormous spurious effects.

Both directions are first-class and must never be conflated:

* **Denoising** (corrupt run, clean activation patched in) finds sets that are
  *sufficient* to restore the behaviour.
* **Noising** (clean run, corrupt activation patched in) finds sets that are
  *necessary*.

The API requires the direction explicitly: `direction="denoise" | "noise"`,
no default. Results carry it in their metadata and the figures label it.

### 3.2 Counterfactual families

```python
class Counterfactual(Protocol):
    def value(self, site: str, index, clean_store, prompt_batch) -> Tensor: ...
```

Implement `ZeroAblation`, `MeanAblation(over=...)`,
`ResampleAblation(counterfactual_prompts)`, and `GaussianAblation(sigma)`.
`MeanAblation` must state its averaging set — mean over the prompt
distribution, over positions, or both, are three different interventions, and
the common bug is averaging over positions and thereby destroying positional
information while believing you preserved the distribution.

### 3.3 Exact node patching

```python
def patch(
    model, *, clean, corrupt, sites, metric, direction,
    counterfactual=None, index_subset=None, batch_size=None,
) -> PatchResult
```

One forward pass per (site, index) group. Shapes and costs are reported in the
result: `PatchResult.forward_passes` is used by §E2's compute axis and must be
counted, not estimated.

Position handling is explicit. Patching `resid_post` "at layer 8" means
nothing without a position; the API takes an index subset over the site's
index space (§1.4) and defaults to *every* index, which for `resid_post` means
every position separately. A pooled-over-positions patch is available as
`index_subset="all"` and is labelled differently in every output, because the
two answer different questions.

### 3.4 Path patching

The effect of the *edge* from sender `u` to receiver `v`, as distinct from the
effect of `u` on the output through all paths. Three-pass procedure:

1. Run clean, cache everything.
2. Run corrupt, cache everything.
3. Run a third pass in which `u`'s output is the corrupt value, **every other
   input to `v` is frozen at its clean value**, and everything downstream of
   `v` is recomputed normally.

Step 3 is where implementations go wrong. "Freeze every other input to `v`"
means: all other components writing into the residual stream that `v` reads,
and — if `v` is an attention head and the edge is into its value input — `v`'s
own query and key inputs, so the pattern does not move. The API therefore
distinguishes edge *ports*:

```python
def path_patch(model, *, clean, corrupt, sender, receiver, port, metric,
               direction, counterfactual=None) -> PathPatchResult
```

with `port in {"q", "k", "v", "mlp_in", "resid"}`. A test constructs a
compiled program where exactly one path exists, patches it, and asserts the
measured edge effect equals the measured node effect; and a second test
constructs a two-path program and asserts the edge effects sum to the node
effect to within the stated tolerance (they will not sum exactly, because the
composition is nonlinear — the *size* of the discrepancy is itself a reportable
number and §E2 reports it).

### 3.5 Gradient approximations

**Attribution patching.** One forward and one backward pass on the clean run
gives `g = dM/da` at every site; combined with cached corrupt activations:

```
  effect_hat(site, index) = <a_corrupt - a_clean, g>  summed over the
                            non-indexed axes of that site
```

**Integrated gradients** with `n` steps:

```
  effect_hat = <a_corrupt - a_clean,
                (1/n) sum_{i=1}^{n} grad at a_clean + (i/n)(a_corrupt-a_clean)>
```

costing `n` forward-backward pairs, which is the knob §E2 sweeps.

**The two known fixes**, implemented and measured separately so their
contribution is visible:

* *Softmax recomputation* for sites upstream of a softmax: the first-order
  term is badly wrong when the pattern is saturated, because the gradient
  through a saturated softmax is near zero while the actual intervention moves
  the argmax. For `q`, `k` and `scores` sites, recompute the pattern under the
  patched value and take the gradient from there.
* *Gradient dropout*: cancellation between paths makes some estimates small
  for the wrong reason; recompute with a subset of downstream gradient paths
  zeroed and take the maximum magnitude over subsets.

Report, in §E2, the error of plain attribution patching, each fix alone, and
both together, against exact patching. If a fix does not help on these
programs, that is the finding (R2).

### 3.6 Automated discovery

```python
def discover(model, *, clean, corrupt, metric, threshold, counterfactual,
             scorer="exact" | "attribution", direction) -> Circuit
```

Greedy edge pruning in reverse topological order over the model's computational
graph: for each candidate edge, measure the effect of removing it given the
removals already committed; if the degradation is below `threshold`, remove it
permanently. The result is a `Circuit`.

Two implementation requirements:

* The committed removals must persist — the whole point is that an edge is
  judged in the context of the ones already gone, which is what distinguishes
  this from ranking all edges by individual effect. A test asserts the
  procedure is order-dependent in the expected way and that running it with a
  threshold of 0 removes nothing.
* The graph is over edges between *ports* (§3.4), not between components, or
  the recovered circuit cannot express "head 9.9 reads head 7.3's output
  through its query input only".

`threshold` is swept in §E4 to produce precision/recall curves. There is no
default threshold, because choosing one is the entire question and hiding it
in a default would be dishonest.

### 3.7 Circuit metrics

With `M` the full model, `C` a circuit and `m` the normalised metric:

* **faithfulness** — `m` with everything outside `C` ablated. One number.
* **completeness** — for sampled subsets `K` of `C`, the discrepancy
  `|m(C \ K) - m(M \ K)|`; report the distribution and its maximum, not the
  mean alone, since completeness fails on specific subsets.
* **minimality** — for each `v in C`, the largest degradation achievable by
  removing `v` from `C \ K` over sampled `K`; a circuit is minimal if every
  member has a `K` under which it matters.
* **edge precision / recall / F1** against a ground-truth graph, with the
  node-level and edge-level versions reported separately.

Subset sampling needs a stated size and seed, and the metric must report its
own Monte Carlo error: completeness computed over 20 subsets has an error bar
and quoting it without one is the kind of thing this repository exists not to
do.

---

## 4. Dictionary learning

### 4.1 The architectures

All take activations `x in R^d`, produce features `f in R^{F}` with
`F = expansion * d`, and reconstruct `xhat`.

```
  f    = enc(W_e (x - b_d) + b_e)
  xhat = W_d f + b_d
```

The pre-encoder subtraction of the decoder bias is not cosmetic: it makes the
dictionary model `x ~ b_d + sum_i f_i W_d[:,i]`, which is the claim being made,
and without it the bias is absorbed into the features.

| variant | `enc` | sparsity term | notes |
|---|---|---|---|
| `ReluSae` | `ReLU` | `lambda * sum_i f_i ||W_d[:,i]||_2` | the norm factor removes the shrinkage/norm-gaming degeneracy |
| `TopKSae` | keep top `k` of `W_e(...)+b_e`, zero the rest | none | exact `L0 = k`; no sparsity coefficient to tune |
| `BatchTopKSae` | keep the top `k*B` activations across the whole batch | none | per-example `L0` varies; inference uses a global threshold estimated as the running mean of the `k*B`-th value, which **must** be reported since train/inference behaviour differs |
| `JumpReluSae` | `z * H(z - theta)` with learned `theta` | `lambda * sum_i H(z_i - theta_i)` | both the Heaviside and the `L0` count are non-differentiable; use a straight-through estimator with a rectangle kernel of bandwidth `eps`, and state that `eps` trades gradient bias against variance |

Decoder columns are unit-normalised after every step for `ReluSae` (otherwise
the L1 penalty is trivially gamed by scaling), and *not* for `TopKSae`, where
there is no penalty to game. The difference is a footgun worth a comment.

**Dead features.** Track per-feature firing counts over a window. Implement
both standard remedies and compare them rather than picking one:
*resampling* (reinitialise dead features to under-reconstructed inputs) and an
*auxiliary loss* (`AuxK`: reconstruct the residual `x - xhat` using the top
`k_aux` dead features, weighted by a small coefficient). §E5 reports the dead
fraction under each.

### 4.2 Transcoders

A transcoder has the same sparse bottleneck but a different target: it maps an
MLP's **input** to that MLP's **output**.

```
  f    = enc(W_e (ln2_out) + b_e)
  yhat = W_d f + b_d  ~=  mlp_out
```

This is what makes cross-layer circuit tracing tractable. With an SAE on the
residual stream, the path from one feature to another runs through the MLP's
nonlinearity and cannot be read off the weights. With a transcoder, the MLP is
*replaced* by a sparse linear map on its active features, so the direct effect
of feature `i` at layer `l` on feature `j` at layer `l' > l` is a product of
matrices and an attention pattern, computable without another forward pass.

`CrossLayerTranscoder` reads layer `l`'s input and writes to the outputs of
layers `l..L-1`, which captures MLP computation that spans layers. State the
cost: the parameter count grows with the number of output layers, and the
training signal per output layer shrinks.

### 4.3 Training

```python
def train_sae(sae, activations, *, steps, batch_size, lr, warmup,
              lambda_sparsity=None, lambda_warmup=None, seed, device,
              dead_feature_window=..., resample=None, aux_k=None) -> TrainReport
```

Requirements:

* **Activation source is a stream, not a tensor.** Training an SAE on GPT-2
  residuals needs more activations than fit in memory. `util/store.py` provides
  a shuffling buffer: fill from model forward passes, shuffle, yield batches,
  refill. The buffer size and the shuffle matter — consecutive tokens from one
  document are correlated and a small buffer makes the SAE fit document
  structure instead of features. Report the buffer size with every run.
* **Sparsity coefficient warmup** from 0 over the first `lambda_warmup` steps
  for the penalty-based variants, otherwise the dictionary dies at step 1.
* **Normalisation.** Scale activations so `E||x||_2 = sqrt(d)`, record the
  scale factor in the report, and apply the inverse when the dictionary is
  used for intervention. Comparing a reconstruction MSE across layers without
  this is comparing two different units.
* `TrainReport` carries, per logged step: MSE, the sparsity term, measured
  `L0`, dead-feature fraction, and the explained variance.

### 4.4 Evaluation

```python
def recovery(sae, true_directions) -> RecoveryReport     # toy regime only
def frontier(saes, activations, model, site) -> FrontierReport
def downstream_kl(sae, model, site, prompts) -> Tensor
```

* `recovery` — for each true feature direction, the maximum cosine similarity
  over learned decoder columns (MMCS), and the reverse direction too: for each
  learned feature, its best true match. Reporting only the first hides a
  dictionary that learned every true feature plus a hundred spurious ones.
  Also report the **Hungarian-matched** mean cosine, which penalises the
  many-to-one matches that MMCS rewards.
* `frontier` — reconstruction MSE against measured `L0`, swept over the
  sparsity knob, which is the only comparison that makes architectures
  commensurable.
* `downstream_kl` — splice the reconstruction back into the model at `site` and
  measure KL from the unmodified model's next-token distribution. This is the
  metric that matters and the one most likely to disagree with MSE; when it
  does, say so.

### 4.5 Attribution graphs

```python
def attribution_graph(model, transcoders, prompt, *, metric,
                      node_threshold, edge_threshold) -> AttributionGraph
```

Build a *local replacement model* for one prompt: freeze attention patterns and
LayerNorm scales at their values on that prompt, replace each MLP by its
transcoder, and add an **error node** per layer carrying the transcoder's
residual. The resulting model is linear in its features by construction, so
every edge weight is a closed-form product rather than an estimate.

Nodes: active features, error nodes, token embeddings, the logit of interest.
Edges: the direct linear effect of a source node's contribution on a target
feature's pre-activation.

Two numbers make the graph honest and both are mandatory in the report:

* the fraction of total influence on the target that flows through **error
  nodes** — the part the dictionaries failed to explain;
* **intervention agreement**: the graph predicts the effect of ablating a
  feature; actually ablate it in the real model and correlate predicted
  against measured. A graph that predicts badly is a picture, not an
  explanation, and this is the number that distinguishes the two.

---

## 5. Toy and planted-algorithm models

### 5.1 Superposition with a known basis

Following the standard construction: `n` features, importance `I_i`, sparsity
`S` (probability a feature is inactive), hidden width `m < n`.

```
  x in R^n,  x_i = 0 w.p. S, else ~ U[0,1]
  h  = W x,                 W in R^{m x n}
  x' = ReLU(W^T h + b)
  loss = sum_i I_i E[(x_i - x'_i)^2]
```

Train `W` and `b`. The *known ground truth* is `W`'s columns: feature `i`'s
direction in the hidden space is `W[:, i]`. Generate `h` samples, train an SAE
on them, and `recovery` (§4.4) is then a direct measurement with no proxy
anywhere in it.

Report the regime dependence rather than one number: recovery as a function of
`S`, of `n/m`, and of the importance spread. Superposition is a function of
sparsity, so a dictionary's recovery must be too, and a single headline figure
would hide it.

### 5.2 Modular addition

One-layer transformer, no LayerNorm, on `(a + b) mod p` with `p = 113`,
inputs `[a, b, =]`, loss on the final position. Train on a fraction of the
`p^2` pairs with weight decay until the test loss drops long after the train
loss does.

The analytic reference: the trained logits are well approximated by

```
  logit(a, b, c) ~= sum_{k in K} c_k * cos( omega_k * (a + b - c) ),
                    omega_k = 2 pi k / p
```

for a small set of key frequencies `K`. The measurements:

* fit the closed form by least squares over all `(a,b,c)` and report the
  fraction of logit variance explained, with `|K|` chosen by the elbow of the
  explained-variance curve rather than by eye;
* the key frequencies, found from the 2D DFT of the logits over `(a,b)`, and
  the fraction of total spectral power they hold;
* neuron-frequency assignment: the DFT of each MLP neuron's activation over the
  `(a,b)` grid, and the fraction of neurons whose power concentrates on a
  single frequency above a stated threshold;
* the training-time trajectory of all of the above against train and test
  loss, so the claim "the circuit forms before the test loss moves" is either
  supported or not. Report what the trajectory actually shows.

### 5.3 The algorithmic task suite

`toy/tasks.py` provides matched clean/corrupt prompt pairs for: induction
(`A B ... A -> B`), copying, two-hop lookup, bracket matching, sorting, and
indirect object identification on GPT-2. Each task supplies
`clean`, `corrupt`, `correct_token`, `incorrect_token` and a `metric`, and
asserts at construction that the clean/corrupt metric gap is non-degenerate
(§3.1), reporting how many generated pairs were discarded.

---

## 6. Steering

```python
def steer(model, direction, *, site, strength, positions="all") -> ContextManager
def steering_frontier(model, direction, *, site, strengths, target_eval,
                      control_eval) -> FrontierReport
```

`target_eval` measures the intended effect; `control_eval` measures collateral
damage as KL from the unmodified model on held-out text with nothing to do
with the steer. The frontier is the pair, swept over `strengths`.

Baselines, on the same axes, because a steering vector that beats nothing is
not a result: the difference of means between two prompt sets, a trained
linear probe direction, and a random direction of matched norm. The random
baseline is the important one — random directions of sufficient norm degrade
behaviour, and some fraction of a steering result is that effect.

---

## 7. Cross-cutting requirements

**Determinism.** `util/seeding.py` exposes `generator(seed) -> torch.Generator`
and `seed_everything(seed)`, and library code takes generators explicitly.
`tests/test_determinism.py` runs the full pipeline of each experiment under
`--quick` twice and asserts equal JSON. Note in the docs that CUDA matmul
non-determinism means the GPU tier asserts `allclose`, not equality, and that
`torch.use_deterministic_algorithms(True)` is set in the CPU tier.

**Compute accounting.** `util/flops.py` counts forward and backward FLOPs from
the config and sequence length analytically, plus actual forward-pass counts
from the engine. Every experiment that compares methods on cost reports both
*forward passes* and *FLOPs*, because the two orderings differ when one method
needs backward passes.

**Results metadata.** Every `results/*.json` carries:

```json
{"metadata": {"seed": 0, "git_sha": "...", "device": "cpu", "dtype": "float32",
              "versions": {"torch": "...", "numpy": "..."},
              "wall_seconds": 0.0, "scope": "full",
              "command": "python experiments/....py"}}
```

`scope` is `"full"` or a string saying exactly how it was reduced.

---

## 8. Revisions

Append dated entries here when the specification is found to be wrong, with
what was changed and why. An empty section means nothing has been found yet,
not that nothing is wrong.

### 2026-11-08 — value sets are keyed on node type, not on encoding

**Was:** §2.2 and the `value_set` function derived a node's finite value set
from its *encoding*, returning nothing for a numerical node.

**Why that was wrong:** a node's encoding is how it sits in the residual
stream; its support is which values it can take. Those are independent.
`SelectorWidth` is numerical — a count read out of attention is naturally a
magnitude — but its support is exactly `0..max_len` and is perfectly well
known. Keying on the encoding made every `Map` over a width unbounded, which
rules out every counting program in the suite: `hist`, `sort`, `most_freq`,
and the rank construction all of them are built from. The error was found by
the first program written against the interpreter.

**Now:** the function is `support` and its rules key on node type.
`SelectorWidth` reports `0..max_len`. Unboundedness propagates without
raising; only the compiler treats it as fatal, and only for a node it has to
allocate a one-hot block for. `tests/test_lang.py::test_support_of_selector_
width_is_bounded_although_numerical` is the regression test and says why.

### 2026-11-08 — a declared support, for turning a magnitude into a category

**Was:** nothing. §2.2 had no way to discretise a numerical variable.

**Why that was a gap:** a numerical variable's support genuinely is not
enumerable by forward propagation — the mean of a prefix of plus and minus
ones takes a different set of values at every length — but a *discretisation*
of it, `lambda m: "neg" if m < 0 else "pos"`, obviously has a small one.
Without a way to say so, no program that asks a numerical question compiles,
which rules out `dyck1`, `pair_balance` and anything else built on a running
count.

**Now:** `Map` and `SequenceMap` take an optional `declared_support`. It is a
claim by the author, not a derivation, and the compiler verifies it
exhaustively over the input space before emitting any weights — so a wrong
declaration is a compile error rather than a model that is wrong on inputs
nobody tried. The escape hatch is narrow and checked, which is the only reason
it is acceptable.

### 2026-11-08 — `ModelConfig` details settled during phase 0

Three small things the specification left implicit, resolved in code and
recorded here rather than left to be rediscovered:

* `d_head` is allowed to be independent of `d_model / n_heads`. Compiled
  models need this: their width is the sum of their variables' one-hot blocks,
  which is whatever the program needs, and padding the residual stream to a
  multiple of the head count would add dimensions that mean nothing — in the
  one regime whose entire value is that every dimension means something.
* `act_fn` gains `"identity"`. Phase 4's validation test needs a configuration
  in which the model is linear and the first-order attribution estimate is
  therefore *exact*; without it, the error that experiment measures cannot be
  separated from a sign bug in the estimator.
* The activation functions are enumerated in `ACT_FNS` and validated at
  construction, because `gelu_tanh` against `gelu_exact` is a ~1e-3 difference
  in activation space — above the cross-check tolerance in §1.7 — and
  mistyping it should fail loudly rather than quietly.
