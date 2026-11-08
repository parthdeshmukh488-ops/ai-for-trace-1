Read AGENT_BRIEF.md. Phase 2's gate must be green; phases 4 and 5 are not
prerequisites. This is phase 6 of docs/BUILD_PLAN.md: dictionary learning
against a known feature basis.

Implement SPEC §4.1, §4.3, §4.4 and §5.1:

* `toy/superposition.py` — the known-basis model and its training.
* `dictlearn/sae.py` — all four architectures. The details that matter are the
  pre-encoder decoder-bias subtraction, the decoder-norm factor in the L1
  term, the norm constraint applying to `ReluSae` and not to `TopKSae`, and
  BatchTopK's inference threshold differing from its training behaviour.
* `dictlearn/train.py` — including the real streaming shuffle buffer in
  `util/store.py`, with its generator passed explicitly. Report the buffer
  size in every run: consecutive tokens from one document are correlated, and
  a small buffer makes the dictionary fit document structure instead of
  features.
* `dictlearn/evaluate.py` — MMCS in both directions, the Hungarian-matched
  mean, the L0-MSE frontier, and `downstream_kl`.

Write the phase-6 tests from docs/BUILD_PLAN.md. The important one is
`test_sae_recovers_orthogonal_features_exactly`: with no superposition
(`m >= n`, orthogonal features) recovery must be essentially perfect, and a
dictionary that fails it is broken in a way no loss curve will reveal.

Then run `experiments/dictionary_recovery.py` per docs/EXPERIMENTS.md#e5. The
grid is reported as a grid — recovery depends on sparsity and on `n/m`, so a
single headline number would hide the only interesting structure in the
result. Three seeds per configuration; report the spread.

The second half of the experiment runs on GPT-2 residuals, where no recovery
number exists. Say that in the output, in those words, and report only the
frontier metrics and the downstream KL there.

Fill in `[[M3a]]` through `[[M3l]]` in README.md, and
`examples/03_dictionaries.py`.

Gate:

```bash
pytest tests/test_sae.py -q
PYTHONPATH=experiments python experiments/dictionary_recovery.py
grep -c '\[\[M3' README.md    # must print 0
```

Report: whether the architecture ranking under the frontier metrics matched
the ranking under actual recovery — the experiment's central question; the
sparsity and `n/m` regime where recovery collapses while reconstruction loss
still looks acceptable; the dead-feature fraction under each remedy; and the
seed-to-seed spread, which decides whether any of the differences are real.
