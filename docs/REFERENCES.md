# References

Grouped by the part of the project they bear on. The point of the list is that
each item is something the implementation can be checked against or departs
from deliberately — not a bibliography for its own sake.

## Superposition and dictionary learning

* Elhage et al. (2022), *Toy Models of Superposition.* The construction in
  SPEC §5.1 is from here; the known feature basis it provides is what makes
  E5 a measurement rather than a demonstration.
* Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable
  Features in Language Models.*
* Bricken et al. (2023), *Towards Monosemanticity.* Source of the resampling
  remedy for dead features.
* Gao et al. (2024), *Scaling and Evaluating Sparse Autoencoders.* TopK, the
  AuxK auxiliary loss, and the argument for removing the sparsity coefficient.
* Bussmann et al. (2024), *BatchTopK Sparse Autoencoders.* Including the
  train/inference threshold mismatch that SPEC §4.1 requires be reported.
* Rajamanoharan et al. (2024), *Jumping Ahead: Improving Reconstruction
  Fidelity with JumpReLU Sparse Autoencoders.* The straight-through estimator
  and its bandwidth.
* Dunefsky et al. (2024), *Transcoders Find Interpretable LLM Feature
  Circuits.* Why an input-output sparse map makes cross-layer tracing
  tractable at all.

## Circuits, patching and discovery

* Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits.*
  The residual-stream decomposition that SPEC §1.6 relies on, and the reason
  LayerNorm folding and weight centring are exact.
* Meng et al. (2022), *Locating and Editing Factual Associations in GPT.*
  Activation patching as a localisation method.
* Wang et al. (2022), *Interpretability in the Wild: a Circuit for Indirect
  Object Identification in GPT-2 Small.* The published circuit E7 is measured
  against, and the source of the faithfulness / completeness / minimality
  definitions in SPEC §3.7.
* Goldowsky-Dill et al. (2023), *Localizing Model Behavior with Path Patching.*
  The freezing procedure in SPEC §3.4.
* Chan et al. (2022), *Causal Scrubbing.* The strongest form of the hypothesis
  test this project approximates, and worth stating what is given up by not
  implementing it in full.
* Conmy et al. (2023), *Towards Automated Circuit Discovery for Mechanistic
  Interpretability.* The greedy edge-pruning procedure in SPEC §3.6.
* Nanda (2023), *Attribution Patching.* The first-order estimator whose error
  E2 measures.
* Kramár et al. (2024), *AtP*: An Efficient and Scalable Method for Localizing
  LLM Behaviour to Components.* Softmax recomputation and gradient dropout,
  the two fixes implemented in SPEC §3.5.
* Syed et al. (2023), *Attribution Patching Outperforms Automated Circuit
  Discovery.* A claim this project is in a position to test rather than accept.
* Ameisen et al. (2025), *Circuit Tracing: Revealing Computational Graphs in
  Language Models.* Local replacement models, error nodes, and the
  intervention-agreement check in SPEC §4.5.

## Compiled and planted-circuit models

* Weiss et al. (2021), *Thinking Like Transformers.* The RASP language that
  SPEC §2.2 is a subset of.
* Lindner et al. (2023), *Tracr: Compiled Transformers as a Laboratory for
  Interpretability.* The compilation strategy, the categorical/numerical
  encoding distinction, and the BOS-absorption trick.
* Zhou et al. (2023), *What Algorithms can Transformers Learn?* On the gap
  between what compiles and what trains.

## Planted algorithms and the analytic case

* Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic
  Interpretability.* The closed form in SPEC §5.2 and the progress measures E6
  tracks over training.
* Olsson et al. (2022), *In-context Learning and Induction Heads.*
* Zhong et al. (2023), *The Clock and the Pizza.* Evidence that the modular
  addition algorithm is not unique, which is why SPEC §5.2 reports per-seed
  frequencies rather than pooling them.

## Steering

* Turner et al. (2023), *Activation Addition.*
* Templeton et al. (2024), *Scaling Monosemanticity.* Feature steering at
  scale, and the clamping behaviour that motivates the strength sweep in E8.
* Rimsky et al. (2023), *Steering Llama 2 via Contrastive Activation
  Addition.* The mean-difference baseline E8 measures against.

## Method and measurement

* Gelman and Loken (2013), *The Garden of Forking Paths.* The reason E1-E8
  record their hypotheses before the run.
* Sculley et al. (2018), *Winner's Curse? On Pace, Progress, and Empirical
  Rigor.* On matched-compute comparison and ablation hygiene, both of which
  this project's experiment protocols are written to respect.
