Read AGENT_BRIEF.md. Phase 6's gate must be green. This is phase 7 of
docs/BUILD_PLAN.md: the analytic case.

Implement SPEC §5.2 in `toy/modadd.py`: training, the 2D-DFT analysis, the
closed-form fit, and the per-checkpoint trajectory.

Validate the analysis before pointing it at a trained network. Build weights
by hand that implement the trig algorithm exactly, and check that the DFT
analysis recovers the frequencies you put in and that the closed-form fit
explains essentially all of the variance. The same discipline as everywhere
else: the measurement instrument is checked against a known answer first.

Include the negative control. Run the whole analysis on a network trained on
randomly permuted labels; it must show no frequency structure. Without it, the
Fourier result could be an artefact of applying spectral analysis to any
smooth function on a cyclic group, and the control is cheap.

Then run `experiments/grokking_circuit.py` per docs/EXPERIMENTS.md#e6, three
seeds, checkpoints every 100 steps.

Report per seed, not pooled. The algorithm this network learns is known not to
be unique, so three seeds finding three different frequency sets is a real
possible outcome and averaging them would destroy it.

Fill in `[[M4a]]` through `[[M4d]]` in README.md.

Gate:

```bash
PYTHONPATH=experiments python experiments/grokking_circuit.py
grep -c '\[\[M4' README.md    # must print 0
```

Report: explained variance per seed; the key frequencies per seed and whether
they agree; the fraction of neurons assigned to a single frequency; what the
negative control showed; and what the trajectory says about whether the
circuit's signatures appear before the test loss moves. If the network did not
grok at the stated hyperparameters, report what you changed and why — do not
quietly tune until it works and present the tuned run as the protocol.
