"""How well does greedy edge pruning recover a known circuit, and is it worth it?

Greedy pruning judges each edge in the context of the removals already
committed, which is what makes it handle redundancy: two paths that each look
dispensable alone are not both dispensable, and a procedure that scored every
edge independently against the full model would remove both.  That is the
claim, and the planted circuits make it testable.

Four arms, so the claim is put at risk rather than illustrated:

* **greedy** -- the procedure, swept over its threshold.
* **ranking** -- score every edge independently against the full model, keep
  the top ``k``.  Cheaper by a constant factor and expected to lose on
  programs with redundancy.
* **random** -- a random subgraph of matched size, which is what any apparent
  skill has to beat.
* **ceiling** -- the planted edges that have any causal signal on these
  prompts at all.  Recall cannot exceed this, and reporting recall without it
  makes a correct method look worse the more thorough the ground truth is.

Cost is reported alongside accuracy, because "90% as good for 1% of the cost"
is the decision-relevant fact and a plot of accuracy alone cannot show it.

    python experiments/discovery_roc.py [--quick]
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
from circuitry.attrib.metrics import recoverable
from circuitry.attrib.patching import ResampleAblation
from circuitry.symbolic import programs
from circuitry.toy.tasks import compiled_pairs

RESULTS = Path(__file__).resolve().parent / "results"
THRESHOLDS = (0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def individual_ranking(engine: AblationEngine, edges) -> list[tuple[object, float]]:
    """Every edge's effect measured against the full model, independently.

    One pass per edge, like greedy, but the measurements do not interact -- so
    it cannot see that two redundant paths are jointly necessary.
    """
    unpatched = 0.0 if engine.direction == "denoise" else 1.0
    scored = [(edge, abs(engine.level([edge]) - unpatched)) for edge in edges]
    return sorted(scored, key=lambda kv: -kv[1])


def run_program(name: str, *, n_pairs: int, max_len: int, seed: int, thresholds) -> dict:
    program, vocab = programs.build(name)
    compiled = compile_program(program, vocab, max_len, verify=False)
    report = compiled_pairs(
        compiled, n_pairs=n_pairs, length=max_len,
        generator=torch.Generator().manual_seed(seed),
    )
    pair = report.pair
    truth = Circuit.from_ground_truth(compiled.graph, compiled.cfg)
    space = all_edges(compiled.cfg)
    floor = noise_floor(
        compiled.model, pair=pair, n_subsets=3, n_boot=100,
        generator=torch.Generator().manual_seed(seed),
    )
    engine = AblationEngine(compiled.model, pair, "noise", ResampleAblation())
    bound = recoverable(engine, truth, threshold=max(floor["noise_floor"], 1e-9))
    reachable = bound["recoverable"]

    out: dict[str, object] = {
        "task": report.to_dict(),
        "noise_floor": floor["noise_floor"],
        "edge_space": len(space),
        "n_planted_edges": len(truth.edges),
        "n_recoverable": bound["n_recoverable"],
        "ceiling_on_recall": bound["ceiling_on_recall"],
        "depth": programs.PROGRAMS[name].depth,
        "model": {
            "n_layers": compiled.cfg.n_layers, "n_heads": compiled.cfg.n_heads,
            "d_model": compiled.cfg.d_model,
        },
        "greedy": {},
        "ranking": {},
        "random": {},
    }

    # --- greedy, swept over its threshold ----------------------------
    for threshold in thresholds:
        found = discover(
            compiled.model, pair=pair, threshold=threshold,
            direction="noise", counterfactual=ResampleAblation(),
        )
        out["greedy"][str(threshold)] = {
            **found.circuit.score_against(truth),
            "vs_recoverable": found.circuit.score_against(reachable),
            "n_kept": found.n_kept,
            "forward_passes": found.forward_passes,
        }

    # --- independent ranking, at every matched size -------------------
    ranked = individual_ranking(engine, space)
    ranking_passes = len(space) + 4
    for k in sorted({v["n_kept"] for v in out["greedy"].values()} | {len(truth.edges)}):
        if k == 0:
            continue
        picked = Circuit.of(edges=[e for e, _ in ranked[:k]], label=f"top{k}")
        out["ranking"][str(k)] = {
            **picked.score_against(truth),
            "vs_recoverable": picked.score_against(reachable),
            "n_kept": k,
            "forward_passes": ranking_passes,
        }

    # --- random subgraphs at matched size -----------------------------
    gen = torch.Generator().manual_seed(seed + 1)
    for k in sorted(out["ranking"]):
        size = int(k)
        f1s = []
        for _ in range(20):
            picks = torch.randperm(len(space), generator=gen)[:size].tolist()
            random_circuit = Circuit.of(edges=[space[i] for i in picks])
            f1s.append(random_circuit.score_against(truth)["edge_f1"])
        out["random"][k] = {
            "edge_f1_mean": float(np.mean(f1s)),
            "edge_f1_max": float(np.max(f1s)),
            "n_kept": size,
            "forward_passes": 0,
        }
    return out


def _best(block: dict, field: str = "edge_f1") -> dict:
    return max(block.values(), key=lambda v: v[field])


def plot(report: dict, path: Path) -> None:
    from style import AQUA, BLUE, GRID, MUTED, ORANGE, RED, new_figure, save, style_axes

    names = list(report["programs"])
    fig, (left, right) = new_figure(1, 2, figsize=(12.0, 4.6))

    # Precision-recall, greedy, one curve per program, with each program's
    # ceiling marked on the recall axis.
    shades = [BLUE, ORANGE, AQUA, "#6250d6", RED, "#8a6d3b", "#2f6f6f", "#9b3b6a"]
    for i, name in enumerate(names):
        block = report["programs"][name]["greedy"]
        points = sorted(
            ((v["edge_recall"], v["edge_precision"]) for v in block.values()),
            key=lambda kv: kv[0],
        )
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        left.plot(xs, ys, color=shades[i % len(shades)], marker="o", markersize=3,
                  linewidth=1.3, label=name)
        left.axvline(
            report["programs"][name]["ceiling_on_recall"],
            color=shades[i % len(shades)], linewidth=0.7, linestyle=":", alpha=0.6,
        )
    style_axes(left, "Greedy pruning: precision against recall, dotted lines are the ceilings",
               "edge recall", "edge precision")
    left.set_xlim(-0.03, 1.03)
    left.set_ylim(-0.03, 1.08)
    left.legend(frameon=False, fontsize=7, ncol=2)

    # Best F1 against cost, per method.
    for label, colour, key in (
        ("greedy", BLUE, "greedy"),
        ("independent ranking", ORANGE, "ranking"),
    ):
        xs, ys = [], []
        for name in names:
            block = report["programs"][name][key]
            best = _best(block)
            xs.append(best["forward_passes"])
            ys.append(best["edge_f1"])
        right.scatter(xs, ys, s=34, color=colour, label=label, linewidths=0, alpha=0.85)
    xs, ys = [], []
    for name in names:
        block = report["programs"][name]["random"]
        xs.append(1)
        ys.append(max(v["edge_f1_mean"] for v in block.values()))
    right.scatter(xs, ys, s=34, color=GRID, label="random, matched size", linewidths=0)
    style_axes(right, "Best edge F1 against what it cost",
               "forward passes", "best edge F1")
    right.set_xscale("log")
    right.legend(frameon=False, fontsize=8, loc="lower right")
    del MUTED
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
        "command": "python experiments/discovery_roc.py" + (" --quick" if args.quick else ""),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "discovery_roc_quick" if args.quick else "discovery_roc"
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    header = (
        f"{'program':11s} {'depth':>5s} {'space':>6s} {'planted':>8s} {'ceiling':>8s} "
        f"{'greedy F1':>10s} {'rank F1':>8s} {'random F1':>10s} {'passes':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        block = report["programs"][name]
        greedy = _best(block["greedy"])
        ranking = _best(block["ranking"])
        random_best = max(v["edge_f1_mean"] for v in block["random"].values())
        print(
            f"{name:11s} {block['depth']:5d} {block['edge_space']:6d} "
            f"{block['n_planted_edges']:8d} {block['ceiling_on_recall']:8.2f} "
            f"{greedy['edge_f1']:10.3f} {ranking['edge_f1']:8.3f} "
            f"{random_best:10.3f} {greedy['forward_passes']:7d}"
        )
    print()
    g = [
        _best(report["programs"][n]["greedy"])["edge_f1"] for n in names
    ]
    r = [_best(report["programs"][n]["ranking"])["edge_f1"] for n in names]
    gr = [
        _best(report["programs"][n]["greedy"])["vs_recoverable"]["edge_recall"] for n in names
    ]
    print(f"mean best edge F1: greedy {np.mean(g):.3f}, independent ranking {np.mean(r):.3f}")
    print(f"mean recall against the ceiling, greedy: {np.mean(gr):.3f}")
    print(f"programs where greedy beat ranking: {sum(1 for a, b in zip(g, r, strict=True) if a > b)}"
          f" of {len(names)}")

    if not args.quick:
        try:
            plot(report, RESULTS / f"{stem}.png")
        except ImportError:  # pragma: no cover
            print("matplotlib not installed; skipped the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
