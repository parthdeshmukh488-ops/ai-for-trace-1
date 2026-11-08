Read AGENT_BRIEF.md. Phase 0's gate must be green. This is phase 1 of
docs/BUILD_PLAN.md: the transformer engine.

Implement SPEC §1 in full: `model/config.py`, `model/transformer.py`,
`model/sites.py`, `model/hooks.py`, `model/loading.py`.

Order of work, which matters here:

1. `sites.py` first — the naming scheme, the glob expansion, and
   `index_space`. Everything else is organised around it, and writing it first
   stops the site names from accreting inconsistently.
2. The forward pass, with hook points at every site in SPEC §1.4. Get
   `test_resid_decomposition_is_exact` passing early; it catches more engine
   bugs than the rest of the suite combined.
3. The hook and capture contexts, with the reversibility contract (SPEC §1.5,
   brief R6). Use `torch.equal` in that test, not `allclose` — a leaked hook
   that perturbs the output slightly is exactly the failure mode this is for.
4. The three weight folds (SPEC §1.6), each with its own
   logits-unchanged test.
5. `loading.py` for the external checkpoint. The checkpoint stores the
   attention projections fused and transposed relative to the layout here;
   work out the mapping from shapes rather than from memory, and write the
   mapping down in the module docstring as a table.

Then `benchmarks/cross_check_reference.py` per SPEC §1.7, and
`examples/01_hooks_and_patching.py` — a short walkthrough using only the
public API that captures an attention pattern, ablates a head, and prints the
change in a logit difference.

Write every test in the phase-1 table of docs/BUILD_PLAN.md.

Gate:

```bash
pytest tests/ -q -m "not slow" && python benchmarks/cross_check_reference.py
```

The cross-check must report a maximum absolute deviation under 1e-4 with
folding off and with folding on, and write
`benchmarks/results/cross_check.json`.

Report: the achieved max absolute and relative deviation per site family, the
layer where it is largest, whether it grows with depth as accumulation
predicts, and the weight-mapping table you derived. If the deviation is above
1e-4, do not proceed — it is the activation function or a transposed weight,
and docs/ENGINEERING.md#debugging-recipes has the order to check in.
