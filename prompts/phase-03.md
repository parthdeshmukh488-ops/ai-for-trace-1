Read AGENT_BRIEF.md. Phase 2's gate must be green. This is phase 3 of
docs/BUILD_PLAN.md: exact attribution.

Implement SPEC §3.1 through §3.4: `attrib/metric.py`, `attrib/patching.py`,
`attrib/paths.py`, `attrib/circuit.py`.

Points to be careful about, in the order they bite:

1. The normalised metric convention and the degenerate-pair filter (SPEC §3.1).
   Report the number of pairs dropped; a silently-filtered dataset is how a
   patching plot ends up with effects larger than the metric's own range.
2. `direction` is a required argument with no default. Denoising and noising
   answer different questions and the API must not let them be confused.
3. The counterfactual families (SPEC §3.2), including `MeanAblation` with an
   explicit `over=` argument. Implement the mean-over-positions variant too,
   even though it is the wrong thing — phase 5 measures what it costs.
4. Path patching's freezing (SPEC §3.4). This is the subtle one. "Freeze all
   other inputs to the receiver" includes the receiver's own query and key
   inputs when the edge is into its value port, or the attention pattern moves
   and you have measured something else. The `port` argument exists for this.
5. Implement the **batched** patching path described in
   docs/ENGINEERING.md#performance — replicate the prompt across the batch and
   patch a different node per replica — behind the same API, with a test
   asserting equality against the unbatched path. E2 and E4 are hours without
   it and minutes with it.

Write every test in the phase-3 table of docs/BUILD_PLAN.md.

Gate:

```bash
pytest tests/test_patching.py tests/test_paths.py -q
```

Report: the gate output, and one measured number — on a compiled program with
two parallel paths, the discrepancy between the sum of the two edge effects
and the node effect. State it as a fraction of the node effect. That number
bounds what any edge decomposition can mean, so record it in
docs/EXPERIMENTS.md as well.
