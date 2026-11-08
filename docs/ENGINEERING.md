# Engineering notes

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,plot,oracle,data]"
```

On a CPU-only machine install the CPU torch wheels first, or the install pulls
roughly 2 GB of CUDA libraries that will never be used:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Python 3.10 is the floor and CI covers through 3.13. The one 3.10
incompatibility worth knowing about is that `itertools.batched` does not exist
there; write the two-line helper rather than raising the floor.

## Determinism

The guarantee is: **same seed, same machine, same result, bitwise.** Across
machines and across CPU/GPU it is `allclose`, and the docs should not claim
more.

Rules that make it hold:

* Library code never touches global RNG. Functions that sample take a
  `generator: torch.Generator` or a `seed: int` and build one. There is no
  module-level default generator.
* `util/seeding.seed_everything(seed)` seeds Python, NumPy and torch, and on
  the CPU tier also sets `torch.use_deterministic_algorithms(True)`.
* DataLoader-style shuffling uses an explicit generator; the activation
  shuffle buffer takes one too, which matters because a non-deterministic
  buffer makes SAE comparisons irreproducible in a way that is very hard to
  notice.
* `tests/test_determinism.py` runs every experiment's `--quick` path twice in
  one process and asserts the JSON payloads are equal, after dropping the
  `metadata.wall_seconds` field.

On CUDA, `torch.use_deterministic_algorithms(True)` makes some ops raise
rather than run. Do not work around that by relaxing the flag globally; set it
per-tier, and state in any GPU result's metadata that determinism is
`allclose`-level.

## Compute tiers

Three tiers. Every experiment must run at the CPU tier, at reduced scope where
necessary, and must say in its results JSON which tier produced the number.

| tier | hardware | what it is for |
|---|---|---|
| `smoke` | any CPU, no model downloads | `--quick` paths; what CI runs |
| `cpu` | 8-core CPU, 16 GB | everything in regimes 1 and 2 at full scope |
| `gpu` | one 16 GB accelerator | regime 3 and the GPT-2 dictionaries |

Estimated full-scope budgets, to be replaced with measured wall times once the
experiments exist:

| experiment | tier | estimate | dominant cost |
|---|---|---|---|
| E1 compiler fidelity | cpu | minutes | exhaustive input enumeration |
| E2 attribution error | cpu | tens of minutes | exhaustive edge patching |
| E3 ablation choice | cpu | tens of minutes | discovery, five counterfactuals |
| E4 discovery ROC | cpu | ~1 hour | 20 thresholds x 11 models x 4 methods |
| E5 dictionary recovery (toy) | cpu | ~1 hour | 3 seeds x the config grid |
| E5 dictionary recovery (GPT-2) | gpu | ~3 hours | activation collection, then 4 x 4 SAEs |
| E6 grokking | cpu | ~30 min | 25k full-batch steps x 3 seeds |
| E7 GPT-2 IOI | gpu | ~4 hours | transcoder training on 12 MLPs |
| E8 steering | cpu | ~30 min | 12 strengths x 4 direction types |

The GPT-2 numbers assume activations are collected once and cached to disk,
not recollected per SAE. Caching is in `util/store.py` and the cache directory
is gitignored.

If you must reduce scope, reduce it in the dimension that costs the least
information — fewer prompts rather than fewer programs, fewer seeds rather
than fewer sparsity levels — and record the reduction in `metadata.scope` as a
sentence, not a flag.

## Memory

The two cliffs, both documented in the code where they occur:

* `attn.result` is `[B,T,H,d]`. For GPT-2 small at `B=8, T=512` that is
  8 x 512 x 12 x 768 x 4 bytes = 151 MB **per layer**, so capturing it across
  all layers is 1.8 GB. Capture it for the layers you need.
* `attn.pattern` is `[B,H,T,T]`, quadratic in sequence length: at `T=1024`
  that is 8 x 12 x 1024 x 1024 x 4 = 402 MB per layer.

`capture(..., budget_bytes=...)` computes the requirement before allocating
and raises with the breakdown. Experiments set a budget; they do not discover
the limit by being killed.

## Data

* GPT-2 weights come from the standard checkpoint; the loader records the
  checkpoint's hash in the results metadata so a silent upstream change cannot
  be mistaken for a result change.
* Text for the control evaluations and SAE training uses a documented public
  corpus slice, with the exact slice (dataset name, split, row range, token
  count) in the results JSON. "A sample of web text" is not reproducible.
* Nothing downloaded is committed. `weights/`, `activations/` and
  `checkpoints/` are gitignored.

## Performance

In order of how much they matter:

1. **Batch the patching loop.** Patching one node at a time is `O(nodes)`
   forward passes, but many nodes can be patched in parallel across the batch
   dimension by replicating the prompt and patching a different node in each
   replica. This is the single biggest win available and it changes E2 and E4
   from hours to minutes. Implement it in `patching.py` behind the same API
   and assert equality with the unbatched path in a test.
2. **Cache the clean and corrupt runs once per prompt batch**, never per node.
3. `torch.inference_mode()` for everything that does not need gradients;
   `no_grad` is not the same thing and leaves version-counter bookkeeping on.
4. Keep compiled models in `float64` — they are tiny, and the exactness
   claims in E1 are cleaner without fp32 noise. Real models stay `float32`;
   do not use `bfloat16` anywhere results are measured, because the
   attribution errors being measured in E2 are smaller than its precision.

## Debugging recipes

**The cross-check deviation jumps at one layer.** Almost always the activation
function (`gelu_tanh` against `gelu_exact`) or a transposed weight in the
loader. Check `attn.pattern` at layer 0 first: if that matches and `resid_post`
does not, the error is in the MLP or in `W_O`'s reshape.

**Patching everything does not recover the clean metric.** A site was missed,
or the counterfactual is being applied to the clean run and the clean value to
the corrupt run. `test_patching_everything_recovers_clean` localises it.

**An SAE's L0 collapses to zero in the first hundred steps.** The sparsity
coefficient has no warmup, or the decoder is not norm-constrained and the
penalty is being minimised by shrinking features rather than removing them.

**The compiler produces a model that is right on most inputs.** It is the BOS
handling or an `Aggregate` over a selector with multiple matches that the
well-definedness check did not catch. Run the exhaustive test and look at the
smallest failing input.

**Discovery returns the whole model.** The threshold is below the metric's
noise floor. Measure the noise floor explicitly — rerun the metric on
identical inputs with different batch composition — and state it next to the
threshold sweep.

## Code style specifics

* `from __future__ import annotations` first, then stdlib, third-party, local,
  each group separated.
* `__all__` explicit and sorted, immediately after imports.
* Dataclasses for anything with more than two fields; `frozen=True` unless
  mutation is the point.
* Type hints on every public signature. `Tensor` annotations say the shape in
  the docstring, in the `[B,T,d]` notation used throughout.
* No `*args, **kwargs` passthrough in public APIs; it defeats the point of
  having a specification.
* Errors are specific exception types defined in the module that raises them,
  with messages that name the offending value. `ValueError("bad shape")` is
  not acceptable; `ValueError(f"site {name!r} has index space {space}, got
  index {index!r}")` is.
