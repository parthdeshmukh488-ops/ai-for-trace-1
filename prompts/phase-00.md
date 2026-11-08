Read AGENT_BRIEF.md and docs/SPEC.md before starting. This is phase 0 of
docs/BUILD_PLAN.md.

Set up the skeleton of the package so that later phases have somewhere to
land, and so that CI is green from the first commit rather than at the end.

Build:

1. `src/circuitry/__init__.py` declaring the public surface from SPEC §0.
   Names that do not exist yet should be stubs that raise
   `NotImplementedError("phase N")` naming the phase that implements them, so
   an accidental early call says what is missing rather than failing at import.
2. `src/circuitry/util/seeding.py` — `generator(seed)`, `seed_everything(seed)`
   per SPEC §7 and docs/ENGINEERING.md#determinism. Library code must never
   read or write global RNG state; this module is the only place allowed to.
3. `src/circuitry/util/flops.py` — analytic forward and backward FLOP counts
   from a `ModelConfig` and a sequence length. Derive the arithmetic in the
   docstring rather than quoting a formula, and include the embedding and
   unembedding terms, which are a third of GPT-2 small's cost and are usually
   left out.
4. `src/circuitry/util/store.py` — for now a simple in-memory
   `ActivationStore` with the `budget_bytes` check and the
   `ActivationBudgetExceeded` error from SPEC §1.5. The streaming shuffle
   buffer is phase 6; leave a `NotImplementedError` stub for it.
5. `tests/conftest.py` with: a `tiny_config` fixture
   (`L=2, H=2, d=16, dm=32, V=20, n_ctx=16`), a seeded `gen` fixture, and a
   `tmp_results` fixture writing under `tmp_path`.
6. `tests/test_seeding.py`, `tests/test_flops.py`, `tests/test_store.py` —
   real tests, not smoke tests. For flops, check against a hand-computed value
   for one small config written out in the test body.

Do not implement the transformer. If you find yourself needing it to write a
test, the test belongs in phase 1.

Gate:

```bash
pytest tests/ -q && ruff check .
```

Report: the gate output, the public names you stubbed and which phase each is
assigned to, and anything in SPEC §0 or §7 that was ambiguous enough that you
had to choose.
