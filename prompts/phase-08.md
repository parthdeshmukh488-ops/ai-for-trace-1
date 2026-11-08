Read AGENT_BRIEF.md. Phases 1, 3, 5 and 6 must all be green. This is phase 8
of docs/BUILD_PLAN.md: the real model.

Implement:

* `toy/tasks.py` — the IOI generator per SPEC §5.3, with the non-degeneracy
  assertion and the count of discarded pairs.
* `dictlearn/transcoder.py` — `Transcoder` and `CrossLayerTranscoder` per
  SPEC §4.2.
* `dictlearn/graph.py` — attribution graphs per SPEC §4.5: the local
  replacement model with attention patterns and LayerNorm scales frozen, MLPs
  replaced by transcoders, and an **error node per layer**.

The graph is unfalsifiable without `test_attribution_graph_edges_match_
bruteforce_on_tiny_model`: compute edge weights by the closed form and by
brute-force ablation on a 2-layer model and assert they agree. Write that test
before the experiment.

Two numbers are mandatory in every graph report, and the graph is a picture
rather than an explanation without them:

* the fraction of influence on the target flowing through error nodes — the
  part the dictionaries did not explain;
* intervention agreement — the graph predicts an ablation's effect, so perform
  the ablation in the real model and correlate predicted against measured.

Then run `experiments/gpt2_ioi.py` per docs/EXPERIMENTS.md#e7.

This regime has **no ground truth**, and the write-up must say so in those
words. What it has is an independently published circuit, internal consistency
between exact and approximate methods, and the faithfulness family — all
proxies. The error bars carried over from E2 through E4 are what make the
numbers here mean anything, so state them alongside.

Compare head by head with the published circuit and list **every**
disagreement individually. "Broadly consistent with the published circuit" is
not an outcome; name the heads.

Fill in `[[M5a]]` through `[[M5d]]` in README.md.

Gate:

```bash
PYTHONPATH=experiments python experiments/gpt2_ioi.py
grep -c '\[\[M5' README.md    # must print 0
```

Report: the recovered circuit; the enumerated disagreements with the published
one and which are threshold- or counterfactual-dependent; faithfulness,
completeness and minimality under each counterfactual; the error-node
influence fraction; the intervention-agreement correlation; and the tier and
scope you ran at.
