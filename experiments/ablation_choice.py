"""How much of a discovered circuit is an artefact of the counterfactual?

Patching needs something to patch *in*, and zero, mean and resample ablation
are all in common use.  They are not the same experiment.  Zeroing an
activation puts the network in a state it occupies on no input at all, so the
behaviour it destroys may be nothing more than the model objecting to being
off-distribution; a mean taken over positions rather than over the prompt
distribution destroys positional information while looking like it preserved
the distribution.

Because the planted circuit is known, the choices can be *scored* rather than
argued about: run the same discovery procedure under each and compare the
subgraph that comes back.  Five arms, including the mean-over-positions
variant deliberately, so that what the common mistake costs is measured rather
than asserted.

Each counterfactual is shown at *its own* best threshold.  A shared threshold
would favour whichever one happens to produce effects on the same scale, which
is a property of the counterfactual and not of the circuit.

Two numbers make the result readable, and the second is not standard.  Recall
against the planted graph is capped: the planted graph is the *program's*
dependency graph, and a given prompt sample exercises only part of it, so an
edge with no causal signal on these prompts cannot be found by any method.
That ceiling is measured and reported alongside, because without it a correct
method looks worse the more thorough the ground truth is.

    python experiments/ablation_choice.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from circuitry import compile_program
from circuitry.attrib.ablate import AblationEngine
from circuitry.attrib.acdc import discover, noise_floor
from circuitry.attrib.circuit import Circuit, all_edges
from circuitry.attrib.metrics import recoverable, score
from circuitry.attrib.patching import (
    GaussianAblation,
    MeanAblation,
    ResampleAblation,
    ZeroAblation,
)
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs

RESULTS = Path(__file__).resolve().parent / "results"
THRESHOLDS = (0.002, 0.01, 0.03, 0.08, 0.2, 0.4)


def counterfactuals(seed: int):
    return [
        ("zero", ZeroAblation()),
        ("mean[batch]", MeanAblation(over="batch")),
        ("mean[positions]", MeanAblation(over="positions")),
        ("resample", ResampleAblation()),
        ("gaussian", GaussianAblation(sigma=1.0,
                                      generator=torch.Generator().manual_seed(seed))),
    ]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def run_program(name: str, *, n_pairs: int, max_len: int, seed: int, thresholds) -> dict:
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, max_len, verify=False)
    report = compiled_pairs(
        compiled, n_pairs=n_pairs, length=max_len,
        generator=torch.Generator().manual_seed(seed),
    )
    pair = report.pair
    truth = Circuit.from_ground_truth(compiled.graph, compiled.cfg)
    floor = noise_floor(
        compiled.model, pair=pair, n_subsets=3, n_boot=100,
        generator=torch.Generator().manual_seed(seed),
    )
    out: dict[str, object] = {
        "task": report.to_dict(),
        "noise_floor": floor["noise_floor"],
        "edge_space": len(all_edges(compiled.cfg)),
        "n_planted_edges": len(truth.edges),
        "model": {
            "n_layers": compiled.cfg.n_layers, "n_heads": compiled.cfg.n_heads,
            "d_model": compiled.cfg.d_model,
        },
        "arms": {},
    }
    for label, counterfactual in counterfactuals(seed):
        engine = AblationEngine(compiled.model, pair, "noise", counterfactual)
        bound = recoverable(engine, truth, threshold=max(floor["noise_floor"], 1e-9))
        best = None
        per_threshold = {}
        for threshold in thresholds:
            found = discover(
                compiled.model, pair=pair, threshold=threshold,
                direction="noise", counterfactual=counterfactual,
            )
            scored = score(
                engine, found.circuit, truth,
                recall_threshold=max(floor["noise_floor"], 1e-9),
            )
            scored["n_kept"] = found.n_kept
            scored["final_level"] = found.final_level
            per_threshold[str(threshold)] = scored
            if best is None or scored["edge_f1"] > best[1]["edge_f1"]:
                best = (threshold, scored, found.circuit)
        out["arms"][label] = {
            "per_threshold": per_threshold,
            "best_threshold": best[0],
            "best": best[1],
            "ceiling_on_recall": bound["ceiling_on_recall"],
            "n_recoverable": bound["n_recoverable"],
            "circuit": sorted(e.label for e in best[2].edges),
        }
        out["arms"][label]["_circuit_obj"] = best[2]

    # How much does the choice change the answer, independently of accuracy?
    labels = list(out["arms"])
    jaccard = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            jaccard[f"{a} vs {b}"] = out["arms"][a]["_circuit_obj"].jaccard(
                out["arms"][b]["_circuit_obj"], on="edge"
            )
    out["pairwise_jaccard"] = jaccard
    for arm in out["arms"].values():
        arm.pop("_circuit_obj")
    return out


def plot(report: dict, path: Path) -> None:
    from style import BLUE, GRID, MUTED, ORANGE, new_figure, save, style_axes

    names = list(report["programs"])
    arms = list(report["programs"][names[0]]["arms"])
    fig, (left, right) = new_figure(1, 2, figsize=(12.0, 4.6))

    width = 0.8 / len(arms)
    shades = ["#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#6250d6"]
    for i, arm in enumerate(arms):
        f1 = [report["programs"][n]["arms"][arm]["best"]["edge_f1"] for n in names]
        left.bar(
            np.arange(len(names)) + i * width, f1, width=width * 0.92,
            color=shades[i % len(shades)], label=arm,
        )
    left.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
    left.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    style_axes(left, "Best edge F1 against the planted circuit, per counterfactual",
               None, "edge F1")
    left.legend(frameon=False, fontsize=7.5, ncol=2)

    # Recall against the planted graph, against its ceiling.
    for arm, shade in zip(arms, shades, strict=False):
        xs = [report["programs"][n]["arms"][arm]["ceiling_on_recall"] for n in names]
        ys = [report["programs"][n]["arms"][arm]["best"]["edge_recall"] for n in names]
        right.scatter(xs, ys, s=30, color=shade, label=arm, linewidths=0, alpha=0.85)
    right.plot([0, 1], [0, 1], color=GRID, linewidth=1.2)
    right.annotate("the ceiling: no causal method\ncan do better", (0.62, 0.66),
                   fontsize=7.5, color=MUTED)
    style_axes(right, "Recall is capped by what the prompts exercise",
               "fraction of planted edges with any causal signal", "recall achieved")
    right.set_xlim(0, 1.05)
    right.set_ylim(0, 1.05)
    del BLUE, ORANGE
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    names = ("hist", "shift") if args.quick else [
        n for n in programs.names() if n != "pair_balance"
    ]
    n_pairs = 8 if args.quick else 32
    max_len = 3 if args.quick else 4
    thresholds = (0.01, 0.1) if args.quick else THRESHOLDS

    report: dict[str, object] = {"programs": {}}
    for name in names:
        report["programs"][name] = run_program(
            name, n_pairs=n_pairs, max_len=max_len, seed=args.seed, thresholds=thresholds
        )

    report["metadata"] = {
        "seed": args.seed,
        "git_sha": _git_sha(),
        "device": "cpu",
        "n_pairs": n_pairs,
        "max_len": max_len,
        "thresholds": list(thresholds),
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "wall_seconds": round(time.time() - started, 2),
        "scope": "quick" if args.quick else "full",
        "command": "python experiments/ablation_choice.py" + (" --quick" if args.quick else ""),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "ablation_choice_quick" if args.quick else "ablation_choice"
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    arms = list(report["programs"][names[0]]["arms"])
    print("Best edge F1 against the planted circuit, each counterfactual at its own")
    print("best threshold. 'vs ceiling' is recall against the edges that have any")
    print("causal signal on these prompts -- the most any method could find.\n")
    header = f"{'counterfactual':17s} {'edge F1':>8s} {'recall':>7s} {'ceiling':>8s} {'vs ceiling':>11s} {'kept':>6s}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        rows = [report["programs"][n]["arms"][arm] for n in names]
        print(
            f"{arm:17s} "
            f"{np.mean([r['best']['edge_f1'] for r in rows]):8.3f} "
            f"{np.mean([r['best']['edge_recall'] for r in rows]):7.3f} "
            f"{np.mean([r['ceiling_on_recall'] for r in rows]):8.3f} "
            f"{np.mean([r['best']['edge_recall_vs_recoverable'] for r in rows]):11.3f} "
            f"{np.mean([r['best']['n_kept'] for r in rows]):6.1f}"
        )
    print()
    print("Pairwise agreement between the recovered circuits (edge Jaccard), averaged:")
    pairs = report["programs"][names[0]]["pairwise_jaccard"]
    for key in pairs:
        values = [report["programs"][n]["pairwise_jaccard"][key] for n in names]
        print(f"  {key:38s} {np.mean(values):.3f}")

    if not args.quick:
        try:
            plot(report, RESULTS / f"{stem}.png")
        except ImportError:  # pragma: no cover
            print("matplotlib not installed; skipped the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
