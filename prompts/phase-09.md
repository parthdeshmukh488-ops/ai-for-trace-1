Read AGENT_BRIEF.md. Phase 8's gate must be green. This is phase 9 of
docs/BUILD_PLAN.md: steering and what it costs.

Implement SPEC §6: `steer.py`, with `steer` and `steering_frontier`.

The whole value of this phase is in the controls, so build them first:

* a random direction at matched L2 norm, 20 draws, giving a band;
* the mean-difference direction between two prompt sets;
* a trained linear probe direction.

A steering curve that sits inside the random-direction band is not a result,
and the figure must make that readable at a glance by shading the band.

Then run `experiments/steering_cost.py` per docs/EXPERIMENTS.md#e8: intended
effect against collateral KL on held-out unrelated text, swept over 12
strengths, for each direction type, at `positions="all"` and `"last"`
separately.

Fill in `[[M6a]]` through `[[M6c]]` in README.md.

Gate:

```bash
PYTHONPATH=experiments python experiments/steering_cost.py
grep -c '\[\[M6' README.md    # must print 0
```

Report: the frontier in one sentence per direction type; the strength at which
unrelated-text loss degrades by 1%; how much of the dictionary feature's
apparent effect at typical demonstration strengths is matched by a random
direction of the same norm; and whether `positions="all"` and `"last"` behave
differently enough to matter.
