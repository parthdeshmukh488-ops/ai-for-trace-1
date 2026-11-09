"""Do sparse autoencoders recover a known feature basis, and do the proxies predict it?

A dictionary is normally judged on reconstruction loss and ``L0``.  Neither is
the claim.  The claim is that the learned directions *are* the network's
features, and in a model whose features were planted by hand that is directly
measurable: for each true direction, how well the best learned direction
aligns with it.

So this experiment measures both -- the proxy and the thing it is a proxy for
-- across a grid of planted models, and asks whether the first predicts the
second.  The pre-registered hypothesis was that the two rankings would
disagree.  They do, and the direction is worse than "disagree".

Two parts of the design matter.

**Comparisons at matched L0.**  The four architectures obtain sparsity in
different ways, so comparing them at their default settings compares the
settings.  The sparsity knob is therefore swept for each, and the headline
comparison is taken at the closest achieved ``L0``.

**Recovery measured only over represented features.**  A planted model at low
sparsity gives up on some of its features -- their columns go to near zero --
and a direction that is not in the hidden space cannot be recovered from it.
Counting those as misses would measure the planted model rather than the
dictionary, so they are excluded and their count reported.

Reported as a grid, never pooled: superposition is a function of sparsity, so
a dictionary's success at undoing it is too, and a single headline number
would hide the only interesting structure in the result.

    python experiments/dictionary_recovery.py [--quick]
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

from circuitry.dictlearn.evaluate import frontier, recovery
from circuitry.dictlearn.sae import SaeConfig
from circuitry.dictlearn.train import train_sae
from circuitry.toy.superposition import SuperpositionSpec, train_superposition

RESULTS = Path(__file__).resolve().parent / "results"

#: The sparsity knob for each architecture.  TopK variants take k directly;
#: the penalty-based ones take a coefficient, swept wide enough to span a
#: comparable range of achieved L0.
KNOBS = {
    "topk": ("k", (2, 4, 8, 16)),
    "batchtopk": ("k", (2, 4, 8, 16)),
    # The penalty range is extended well past what looks reasonable because
    # the first attempt stopped at 2e-2 and the penalty-based architectures
    # bottomed out at an L0 of ten -- so the "matched L0" comparison was not
    # matched, and the comparison it made was between settings rather than
    # between architectures.
    "relu": ("lambda_sparsity", (3e-4, 1e-3, 5e-3, 2e-2, 6e-2, 1.5e-1)),
    "jumprelu": ("lambda_sparsity", (3e-4, 1e-3, 5e-3, 2e-2, 6e-2, 1.5e-1)),
}
TARGET_L0 = 4.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def planted(spec: SuperpositionSpec, *, seed: int, steps: int):
    model, report = train_superposition(spec, steps=steps, batch=512, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed + 1000)
    activations = model.hidden(model.sample(16384, generator=generator)).detach()
    norms = model.feature_norms()
    represented = norms > 0.1 * norms.max()
    return model, activations, represented, report


def run_cell(
    spec: SuperpositionSpec, *, seed: int, steps: int, sae_steps: int, expansion: int
) -> dict:
    model, activations, represented, planted_report = planted(spec, seed=seed, steps=steps)
    cell: dict[str, object] = {
        "planted": planted_report,
        "n_represented": int(represented.sum()),
        "runs": {},
    }
    reports = []
    for kind, (knob, values) in KNOBS.items():
        for value in values:
            cfg = SaeConfig(
                d_in=spec.n_hidden,
                expansion=expansion,
                k=value if knob == "k" else None,
                lambda_sparsity=value if knob == "lambda_sparsity" else 0.0,
                dtype="float64",
            )
            sae, report = train_sae(
                kind, cfg, activations, steps=sae_steps, batch_size=256, lr=3e-3,
                seed=seed, lambda_warmup=max(1, sae_steps // 10),
            )
            found = recovery(sae.directions, model.true_directions, represented=represented)
            cell["runs"][f"{kind}|{knob}={value}"] = {
                "kind": kind,
                knob: value,
                "train": report.to_dict(),
                "recovery": found,
            }
            reports.append(report)
    cell["frontier"] = frontier(reports)
    return cell


def matched_comparison(cell: dict, target: float = TARGET_L0) -> dict:
    """For each architecture, the run whose achieved L0 is closest to the target."""
    out = {}
    for kind in KNOBS:
        candidates = [
            v for v in cell["runs"].values() if v["kind"] == kind
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda v: abs(v["train"]["final_l0"] - target))
        out[kind] = {
            "l0": best["train"]["final_l0"],
            "mse": best["train"]["final_mse"],
            "explained_variance": best["train"]["final_explained_variance"],
            "dead_fraction": best["train"]["dead_fraction"],
            "matched_mean_cosine": best["recovery"]["matched_mean_cosine"],
            "mmcs_true_to_learned": best["recovery"]["mmcs_true_to_learned"],
            "mmcs_learned_to_true": best["recovery"]["mmcs_learned_to_true"],
            "fraction_true_above_0.9": best["recovery"]["fraction_true_above_0.9"],
        }
    return out


def plot(report: dict, path: Path) -> None:
    from style import AQUA, BLUE, GRID, MUTED, ORANGE, VIOLET, new_figure, save, style_axes

    colours = {"topk": BLUE, "batchtopk": AQUA, "relu": ORANGE, "jumprelu": VIOLET}
    cells = report["cells"]
    fig, (left, middle, right) = new_figure(1, 3, figsize=(14.5, 4.5))

    # 1. the headline: recovery against reconstruction loss.  If the proxy
    # predicted recovery these would trend together.
    for key, cell in cells.items():
        for run in cell["runs"].values():
            left.scatter(
                run["train"]["final_mse"], run["recovery"]["matched_mean_cosine"],
                s=16, color=colours[run["kind"]], alpha=0.65, linewidths=0,
            )
        del key
    for kind, colour in colours.items():
        left.scatter([], [], s=24, color=colour, label=kind, linewidths=0)
    style_axes(left, "Recovery against reconstruction loss",
               "reconstruction MSE (lower is better)", "matched mean cosine to true features")
    left.set_xscale("log")
    left.legend(frameon=False, fontsize=8, loc="lower right")
    left.annotate("better reconstruction,\nworse recovery", (0.03, 0.12),
                  xycoords="axes fraction", fontsize=7.5, color=MUTED)

    # 2. recovery against the planted model's sparsity, at matched L0
    sparsities = sorted({cell["planted"]["sparsity"] for cell in cells.values()})
    for kind, colour in colours.items():
        ys = []
        for s in sparsities:
            values = [
                cell["matched"][kind]["matched_mean_cosine"]
                for cell in cells.values()
                if cell["planted"]["sparsity"] == s and kind in cell["matched"]
            ]
            ys.append(float(np.mean(values)) if values else np.nan)
        middle.plot(sparsities, ys, color=colour, marker="o", markersize=4,
                    linewidth=1.5, label=kind)
    style_axes(middle, "Recovery against the planted sparsity, at matched L0",
               "probability a feature is inactive", "matched mean cosine")
    middle.set_ylim(0, 1.05)

    # 3. the L0-MSE frontier, which is what makes the architectures comparable
    for kind, colour in colours.items():
        points = []
        for cell in cells.values():
            points += [(p["l0"], p["mse"]) for p in cell["frontier"].get(kind, [])]
        points.sort()
        if points:
            right.plot([p[0] for p in points], [p[1] for p in points], color=colour,
                       marker="o", markersize=3, linewidth=1.0, alpha=0.8, label=kind)
    style_axes(right, "The frontier: reconstruction against achieved sparsity",
               "measured L0", "reconstruction MSE")
    right.set_yscale("log")
    del GRID
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    if args.quick:
        features = (16,)
        sparsities = (0.9,)
        seeds = (0,)
        planted_steps, sae_steps, expansion = 300, 300, 4
    else:
        features = (16, 32)
        sparsities = (0.7, 0.9, 0.99)
        seeds = (0, 1)
        planted_steps, sae_steps, expansion = 2000, 2500, 8

    report: dict[str, object] = {"cells": {}}
    for n in features:
        for sparsity in sparsities:
            for seed in seeds:
                spec = SuperpositionSpec(n_features=n, n_hidden=8, sparsity=sparsity)
                key = f"n{n}_S{sparsity}_s{seed}"
                cell = run_cell(
                    spec, seed=seed, steps=planted_steps, sae_steps=sae_steps,
                    expansion=expansion,
                )
                cell["matched"] = matched_comparison(cell)
                report["cells"][key] = cell

    report["metadata"] = {
        "seed": args.seed,
        "git_sha": _git_sha(),
        "device": "cpu",
        "n_hidden": 8,
        "expansion": expansion,
        "planted_steps": planted_steps,
        "sae_steps": sae_steps,
        "target_l0": TARGET_L0,
        "knobs": {k: list(v[1]) for k, v in KNOBS.items()},
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "wall_seconds": round(time.time() - started, 2),
        "scope": "quick" if args.quick else "full",
        "note": (
            "The GPT-2 residual arm is not run here: on four CPU cores a dictionary over "
            "768 dimensions needs hours, and no recovery metric exists in that regime "
            "anyway, only the frontier. It is phase 8's business."
        ),
        "command": "python experiments/dictionary_recovery.py"
        + (" --quick" if args.quick else ""),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "dictionary_recovery_quick" if args.quick else "dictionary_recovery"
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    cells = report["cells"]
    print("At matched L0 (closest achieved to 4), averaged over the grid:\n")
    header = (
        f"{'architecture':13s} {'L0':>5s} {'MSE':>9s} {'EV':>6s} "
        f"{'matched cos':>12s} {'t2l':>6s} {'l2t':>6s} {'>0.9':>6s} {'dead':>6s}"
    )
    print(header)
    print("-" * len(header))
    for kind in KNOBS:
        rows = [c["matched"][kind] for c in cells.values() if kind in c["matched"]]
        print(
            f"{kind:13s} {np.mean([r['l0'] for r in rows]):5.1f} "
            f"{np.mean([r['mse'] for r in rows]):9.4f} "
            f"{np.mean([r['explained_variance'] for r in rows]):6.3f} "
            f"{np.mean([r['matched_mean_cosine'] for r in rows]):12.3f} "
            f"{np.mean([r['mmcs_true_to_learned'] for r in rows]):6.3f} "
            f"{np.mean([r['mmcs_learned_to_true'] for r in rows]):6.3f} "
            f"{np.mean([r['fraction_true_above_0.9'] for r in rows]):6.2f} "
            f"{np.mean([r['dead_fraction'] for r in rows]):6.2f}"
        )

    # Does the proxy predict the thing?
    mse, cos = [], []
    for cell in cells.values():
        for run in cell["runs"].values():
            mse.append(run["train"]["final_mse"])
            cos.append(run["recovery"]["matched_mean_cosine"])
    order_mse = np.argsort(np.argsort(mse))
    order_cos = np.argsort(np.argsort(cos))
    rho = float(np.corrcoef(order_mse, order_cos)[0, 1])
    print()
    print(f"Over all {len(mse)} dictionaries, rank correlation between reconstruction MSE")
    print(f"and recovery: {rho:+.3f}.  A positive value means *worse* reconstruction goes")
    print("with *better* recovery, i.e. the proxy is anti-predictive.")
    report["mse_recovery_rank_correlation"] = rho
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print()
    print("Recovery against the planted sparsity, matched L0, matched-cosine:")
    for s in sorted({c["planted"]["sparsity"] for c in cells.values()}):
        line = f"  S={s:<5g}"
        for kind in KNOBS:
            vals = [
                c["matched"][kind]["matched_mean_cosine"]
                for c in cells.values()
                if c["planted"]["sparsity"] == s and kind in c["matched"]
            ]
            line += f"  {kind}={np.mean(vals):.3f}"
        rep = [c["n_represented"] for c in cells.values() if c["planted"]["sparsity"] == s]
        line += f"   (features represented: {np.mean(rep):.1f})"
        print(line)

    if not args.quick:
        try:
            plot(report, RESULTS / f"{stem}.png")
        except ImportError:  # pragma: no cover
            print("matplotlib not installed; skipped the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
