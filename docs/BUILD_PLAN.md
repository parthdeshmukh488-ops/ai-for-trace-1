# Build plan

Eleven phases. Each has deliverables, the tests that must exist, and an **exit
gate** that is a command with a pass/fail outcome rather than a judgement call.
Do not begin a phase until the previous gate is green; the whole argument of
the project is that each layer is verified before the next one leans on it.

Effort estimates assume someone who knows the material and is not learning
PyTorch at the same time. They are there to flag when a phase has gone wrong,
not as a target.

---

## Phase 0 — bootstrap · ~half a day

**Deliverables**

* `src/circuitry/__init__.py` with the public surface declared (importing
  names that do not exist yet is fine; stub them with `NotImplementedError`).
* `util/seeding.py`, `util/flops.py`, `util/store.py` (the activation buffer
  can be a stub that holds a tensor; the streaming version lands in phase 6).
* `tests/conftest.py` with the shared fixtures: a tiny `ModelConfig`
  (`L=2, H=2, d=16, dm=32, V=20, n_ctx=16`) used by most tests, a fixed
  generator, and a `tmp_results` fixture.
* `ruff` clean, CI green.

**Gate**

```bash
pytest tests/ -q && ruff check .
```

---

## Phase 1 — the engine · ~3 days

**Deliverables**

* `model/config.py`, `model/transformer.py`, `model/sites.py`,
  `model/hooks.py`, `model/loading.py` per SPEC §1.
* `benchmarks/cross_check_reference.py` per SPEC §1.7.
* `examples/01_hooks_and_patching.py`.

**Tests that must exist**

| test | what would fail it |
|---|---|
| `test_shapes_at_every_site` | a site whose tensor does not match SPEC §1.4 |
| `test_causal_mask_blocks_future` | patching a masked score position changes the output |
| `test_intervention_is_exactly_reversible` | a leaked hook handle (`torch.equal`, not `allclose`) |
| `test_capture_matches_forward` | a captured activation that is not the one actually used |
| `test_fold_ln_preserves_logits` | a folding bug — the most likely single error in this phase |
| `test_center_writing_weights_preserves_logits` | as above |
| `test_center_unembed_preserves_argmax` | as above |
| `test_attn_result_sums_to_attn_out` | per-head decomposition that does not reconstitute |
| `test_resid_decomposition_is_exact` | `resid_post = embed + sum(heads) + sum(mlps)` failing |
| `test_budget_raises_before_oom` | a capture that silently allocates gigabytes |
| `test_glob_with_no_match_raises` | a silently-empty site expansion |

`test_resid_decomposition_is_exact` is the one that catches most engine bugs:
the residual stream at any layer must equal the sum of everything written into
it so far, to float tolerance, and that identity touches nearly every code
path.

**Gate**

```bash
pytest tests/ -q -m "not slow" && python benchmarks/cross_check_reference.py
```

The cross-check must print a maximum absolute deviation under `1e-4` with
folding both off and on, and write `benchmarks/results/cross_check.json`.

---

## Phase 2 — the compiler · ~5 days

The hardest phase and the one the project's value depends on. Expect the
attention-logit scale and the MLP lookup margins to need measurement rather
than derivation.

**Deliverables**

* `symbolic/lang.py`, `bases.py`, `compiler.py`, `truth.py`, `programs.py`
  per SPEC §2, with all nine programs in the suite.
* `experiments/compiler_fidelity.py` (§E1).
* `examples/02_compiled_ground_truth.py`.

**Tests that must exist**

| test | what would fail it |
|---|---|
| `test_program_interpreter_matches_definition` | the reference interpreter itself being wrong — write it first, it is the oracle for the compiler |
| `test_compiled_matches_interpreter_exhaustively` | any compilation bug, over the *entire* input space for `T <= 5` and a small vocabulary |
| `test_basis_blocks_are_disjoint` | two variables sharing a dimension |
| `test_ill_defined_aggregate_raises` | a selector with multiple matches compiling silently |
| `test_unused_components_have_no_influence` | the `unused` true-negative set being contaminated |
| `test_ground_truth_edges_match_program_dependencies` | the emitted graph disagreeing with the program's own structure |
| `test_bos_absorbs_empty_selection` | an empty selector producing a uniform aggregate |

Exhaustive verification is the point. For `T <= 5` over a 4-symbol vocabulary
there are at most 1365 inputs per program; run all of them and assert exact
equality of the decoded output with the interpreter. A compiler that is right
on sampled inputs and wrong on one edge case would silently corrupt every
downstream measurement.

**Gate**

```bash
pytest tests/test_compiler.py -q
PYTHONPATH=experiments python experiments/compiler_fidelity.py
```

All nine programs at 100% exact match at the chosen logit scale, and the
fidelity-vs-scale curve written to `results/compiler_fidelity.json`.

---

## Phase 3 — exact attribution · ~3 days

**Deliverables**

* `attrib/metric.py`, `patching.py`, `paths.py`, `circuit.py` per SPEC §3.1–3.4.

**Tests that must exist**

| test | what would fail it |
|---|---|
| `test_patching_identity_is_noop` | patching a site with its own clean value changing anything |
| `test_patching_everything_recovers_clean` | a normalisation or direction error |
| `test_normalised_metric_endpoints` | the `0`/`1` convention being inverted |
| `test_degenerate_pairs_are_dropped` | a near-zero denominator surviving into a result |
| `test_single_path_edge_equals_node_effect` | path patching's freezing being wrong — run on a compiled one-path program |
| `test_direction_is_required` | a defaulted denoise/noise |
| `test_zero_threshold_discovery_removes_nothing` | the pruning loop's bookkeeping |

**Gate**

```bash
pytest tests/test_patching.py tests/test_paths.py -q
```

Plus one reported number, printed by the test run and recorded in
`docs/EXPERIMENTS.md`: on the two-path program, the discrepancy between the
sum of edge effects and the node effect. This is a real property of the
nonlinearity, not a bug, and it bounds how much any edge decomposition can
mean.

---

## Phase 4 — gradient approximations and the headline measurement · ~3 days

**Deliverables**

* `attrib/gradients.py` per SPEC §3.5: attribution patching, integrated
  gradients, softmax recomputation, gradient dropout.
* `experiments/attribution_error.py` (§E2) — the project's central result.
* README `[[M1a]]`–`[[M1i]]` filled from it.

**Tests**

* `test_attribution_patching_matches_exact_in_linear_model` — construct a
  model with the nonlinearities removed (identity activation, no softmax: a
  hand-built linear config) where the first-order estimate is *exact*, and
  assert agreement to float tolerance. If the estimator is wrong, this test
  catches it; the error measured in §E2 is then attributable to curvature
  rather than to a sign bug.
* `test_integrated_gradients_converges_to_exact` — error decreasing in `n` at
  the expected rate on a smooth case.
* `test_gradient_cost_accounting` — reported forward/backward counts matching
  an independent count from hook call tallies.

**Gate**

```bash
PYTHONPATH=experiments python experiments/attribution_error.py
```

`results/attribution_error.json` exists, contains per-program breakdowns
(never a single pooled number), and the README table has no placeholders left
in it.

---

## Phase 5 — discovery and circuit metrics · ~4 days

**Deliverables**

* `attrib/acdc.py`, `attrib/metrics.py` per SPEC §3.6–3.7.
* `experiments/ablation_choice.py` (§E3), `experiments/discovery_roc.py` (§E4).
* README `[[M2a]]`–`[[M2h]]` filled.

**Tests**

* `test_recovers_planted_circuit_exactly_at_tight_threshold` — on `shift` and
  `hist`, where the circuit is small enough that exact recovery should be
  achievable; if it is not, find out why before moving on.
* `test_completeness_reports_its_own_error_bar`.
* `test_edge_and_node_metrics_differ_on_a_known_case`.

**Gate**

```bash
PYTHONPATH=experiments python experiments/ablation_choice.py
PYTHONPATH=experiments python experiments/discovery_roc.py
```

Both JSONs written, ROC curves per program, and a written answer in
`docs/EXPERIMENTS.md` to: *which counterfactual recovered the planted circuit
best, and by how much?*

---

## Phase 6 — dictionaries on known features · ~4 days

**Deliverables**

* `dictlearn/sae.py`, `train.py`, `evaluate.py`; `toy/superposition.py`.
* `util/store.py` upgraded to the real streaming shuffle buffer.
* `experiments/dictionary_recovery.py` (§E5).
* `examples/03_dictionaries.py`.
* README `[[M3a]]`–`[[M3l]]` filled.

**Tests**

* `test_sae_recovers_orthogonal_features_exactly` — when `m >= n` and features
  are orthogonal there is no superposition and recovery must be essentially
  perfect. A dictionary that fails this is broken, and this test is much more
  informative than a loss-goes-down test.
* `test_topk_l0_is_exactly_k`.
* `test_batchtopk_inference_threshold_matches_training_l0` — the train/eval
  mismatch this architecture invites.
* `test_jumprelu_straight_through_gradient_sign`.
* `test_decoder_norm_constraint_holds_after_step`.
* `test_shuffle_buffer_decorrelates` — autocorrelation of consecutive batch
  items below a stated threshold; a buffer that does not shuffle is a silent
  and severe confound.

**Gate**

```bash
pytest tests/test_sae.py -q
PYTHONPATH=experiments python experiments/dictionary_recovery.py
```

Recovery reported as a function of sparsity and `n/m`, both MMCS directions
and the Hungarian-matched mean, for all four architectures.

---

## Phase 7 — the analytic case · ~3 days

**Deliverables**

* `toy/modadd.py`: training, the 2D-DFT analysis, the closed-form fit.
* `experiments/grokking_circuit.py` (§E6).
* README `[[M4a]]`–`[[M4d]]` filled.

**Tests**

* `test_analytic_logits_are_recovered_from_synthetic_weights` — build weights
  that implement the trig algorithm *by construction* and check the analysis
  recovers the frequencies you put in. The analysis must be validated before
  it is pointed at a trained network, same as everything else here.
* `test_dft_frequency_identification_on_known_signal`.

**Gate**

```bash
PYTHONPATH=experiments python experiments/grokking_circuit.py
```

Explained variance reported, frequencies listed, and the trajectory figure
produced. If the network does not grok under the stated hyperparameters,
report the hyperparameters that were needed and what changed.

---

## Phase 8 — the real model · ~5 days

**Deliverables**

* `toy/tasks.py` IOI generator; `dictlearn/transcoder.py`, `graph.py`.
* `experiments/gpt2_ioi.py` (§E7).
* README `[[M5a]]`–`[[M5d]]` filled.

**Tests**

* `test_ioi_pairs_are_non_degenerate`.
* `test_transcoder_reconstructs_mlp_better_than_identity` — a floor test; the
  informative numbers are in the experiment.
* `test_attribution_graph_edges_match_bruteforce_on_tiny_model` — compute the
  graph's edge weights by the closed form and by brute-force ablation on a
  2-layer model and assert agreement. Without this, the graph is unfalsifiable.
* `test_error_node_fraction_is_reported`.

**Gate**

```bash
PYTHONPATH=experiments python experiments/gpt2_ioi.py
```

The recovered circuit, the published circuit, and every disagreement listed
individually in `docs/EXPERIMENTS.md`. "Broadly consistent" is not an outcome;
name the heads that differ.

---

## Phase 9 — steering · ~2 days

**Deliverables**

* `steer.py`; `experiments/steering_cost.py` (§E8).
* README `[[M6a]]`–`[[M6c]]` filled.

**Tests**

* `test_steer_restores_model_on_exit`.
* `test_random_direction_baseline_is_included` — a guard against the
  experiment being run without its most important control.

**Gate**

```bash
PYTHONPATH=experiments python experiments/steering_cost.py
```

---

## Phase 10 — the write-up · ~3 days

**Deliverables**

* Every `[[Mx]]` placeholder in `README.md` replaced by a measured value.
* `docs/EXPERIMENTS.md` complete: for each experiment, the hypothesis, what
  was measured, what the number was, and whether the hypothesis survived.
* A `## What disagreed` section in the README, listing the places where the
  measurement contradicted the expected answer. If this section is empty, the
  experiments were not sharp enough — go back and find the disagreement,
  because it is there.
* Coverage at or above 93% with the floor enforced in CI.
* All three examples runnable from a clean checkout.

**Gate**

```bash
pytest tests/ -q -m "not gpu" --cov=circuitry --cov-fail-under=93
ruff check .
grep -c '\[\[M' README.md    # must print 0
```

---

## Dependency summary

```
  0 -> 1 -> 2 -> 3 -> 4
            |     |
            |     +-> 5
            +-> 6 -> 7
  1,3,5,6 -> 8 -> 9 -> 10
```

Phases 6 and 7 are independent of 4 and 5 and can be done in either order. The
only ordering that is not negotiable is that 1 precedes everything and 2
precedes 3, 4 and 5, because those three are all scored against the compiler's
ground truth.
