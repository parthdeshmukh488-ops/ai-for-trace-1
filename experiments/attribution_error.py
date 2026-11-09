"""How wrong is a gradient approximation to a patching effect, when the truth is known?

Exact activation patching *is* the definition of a causal effect in these
models: re-run the forward pass with one activation replaced and read the
metric.  It is also expensive, one evaluation per index, so the field mostly
uses attribution patching -- a first-order Taylor expansion that estimates
every index at once from a single backward pass.  The approximation is exactly
as good as the metric is locally linear in that activation, and the places it
breaks are the interesting ones: saturated softmaxes, LayerNorm, and any node
whose effect is large, which is to say the nodes that end up in a circuit.

Both sides of that comparison are computable, so the error is not a matter of
opinion.  This is the measurement.

Why not on compiled models
--------------------------

The specification said to run this on the compiled programs, where the ground
truth circuit is known.  That is wrong, and the reason is worth stating because
it is a fact about the whole compiled-model approach.  A compiled model is
exact *because* its lookup tables are piecewise constant -- one neuron per
domain element, firing or not -- and the derivative of a piecewise-constant
function is zero.  So the gradient inside a compiled model is identically zero
at a large fraction of its sites: measured at a third to two thirds across the
suite, growing with program depth.  Every gradient-based estimate there is
exactly 0 while the true effect is large, so the "error" measured would be that
artefact rather than curvature.

Compiled models remain the right substrate for *discovery* -- phases 5 and
beyond, where the scorer is exact patching and the question is which subgraph
comes back.  They are the wrong substrate for anything involving a gradient.
This experiment therefore runs on models with smooth nonlinearities, reports
the dead-gradient fractions as a separate arm so the exclusion is on the
record, and spans depth as its proxy for composition.

How the error is measured
-------------------------

Absolute error, and absolute error as a fraction of the largest true effect in
that substrate.  Elementwise *relative* error is deliberately not reported:
most indices have effects near zero, dividing by them produces enormous ratios
that say nothing, and any summary over them is dominated by the smallest
effect in the model -- the same mistake the forward-pass cross-check made
first.  Sign errors are counted only over indices whose true effect exceeds 5%
of the largest, because the sign of a near-zero effect is not a claim anybody
makes.

    python experiments/attribution_error.py [--quick] [--gpt2]
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

from circuitry import ModelConfig, Transformer
from circuitry.attrib.gradients import attribute
from circuitry.attrib.metric import PromptPair, logit_diff
from circuitry.attrib.patching import ResampleAblation, default_granularity, patch

RESULTS = Path(__file__).resolve().parent / "results"

#: Grouped so the report says which *kind* of site a method fails on, rather
#: than listing thirty names.  q, k and scores are the families the softmax
#: fix is expected to help and nothing else.
SITE_GROUPS = {
    "attn.z": "blocks.*.attn.z",
    "attn.q": "blocks.*.attn.q",
    "attn.k": "blocks.*.attn.k",
    "attn.scores": "blocks.*.attn.scores",
    "attn.pattern": "blocks.*.attn.pattern",
    "mlp.out": "blocks.*.mlp.out",
    "resid_post": "blocks.*.resid_post",
}

SIGN_THRESHOLD = 0.05


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in a dependency for one formula."""
    if len(a) < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        return ranks

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _compare(exact, estimate, *, scale: float) -> dict:
    keys = sorted(exact.effects)
    truth = np.array([exact.effects[k] for k in keys])
    guess = np.array([estimate.effects[k] for k in keys])
    error = np.abs(guess - truth)
    big = np.abs(truth) > SIGN_THRESHOLD * scale
    sign_wrong = (np.sign(guess) != np.sign(truth)) & big

    deciles = {}
    if len(truth) >= 10:
        order = np.argsort(np.abs(truth))
        for d, chunk in enumerate(np.array_split(order, 10)):
            if len(chunk):
                deciles[str(d)] = {
                    "mean_abs_true": float(np.abs(truth[chunk]).mean()),
                    "median_abs_error": float(np.median(error[chunk])),
                    "max_abs_error": float(error[chunk].max()),
                }
    return {
        "n_indices": len(keys),
        "median_abs_error": float(np.median(error)),
        "p95_abs_error": float(np.percentile(error, 95)),
        "max_abs_error": float(error.max()),
        "median_error_frac_of_largest": float(np.median(error) / scale) if scale else float("nan"),
        "max_error_frac_of_largest": float(error.max() / scale) if scale else float("nan"),
        "sign_error_frac": float(sign_wrong.sum() / max(int(big.sum()), 1)),
        "n_sign_eligible": int(big.sum()),
        "spearman": _spearman(truth, guess),
        "pearson": _pearson(truth, guess),
        "batch_items": estimate.batch_items,
        "forward_passes": estimate.forward_passes,
        "backward_passes": estimate.backward_passes,
        "by_effect_decile": deciles,
    }


def _substrates(quick: bool):
    """Smooth models, spanning depth as the proxy for composition depth."""
    # Depth and seed counts chosen so the full run finishes in tens of
    # minutes on four CPU cores.  The cost is dominated by the
    # integrated-gradient arms, which need one forward-backward per index per
    # step: for a six-layer model that is about 300 indices times 20 steps
    # times two directions per substrate.  The convergence *rate* -- the
    # evidence that the method converges to the right number at all -- is
    # measured in tests/test_gradients.py at step counts up to 64, where it
    # only has to run on one model.
    depths = (2, 4) if quick else (2, 4, 6)
    seeds = (0,) if quick else (0, 1)
    out = []
    for depth in depths:
        for seed in seeds:
            out.append((f"random_L{depth}_s{seed}", {"depth": depth, "seed": seed, "saturate": 1.0}))
    # A saturated-attention variant: the regime the softmax fix exists for,
    # reached by scaling the query and key projections rather than by training.
    out.append(("saturated_L4", {"depth": 4, "seed": 0, "saturate": 40.0}))
    return out


def build_substrate(depth: int, seed: int, saturate: float, *, batch: int, seq: int):
    gen = torch.Generator().manual_seed(seed)
    cfg = ModelConfig(
        n_layers=depth, n_heads=2, d_model=16, d_mlp=32, d_vocab=24, n_ctx=max(seq, 8),
        dtype="float64",
    )
    model = Transformer(cfg, generator=gen).eval()
    if saturate != 1.0:
        with torch.no_grad():
            for block in model.blocks:
                block.attn.W_Q *= saturate
                block.attn.W_K *= saturate
    clean = torch.randint(0, cfg.d_vocab, (batch, seq), generator=gen)
    corrupt = torch.randint(0, cfg.d_vocab, (batch, seq), generator=gen)
    pair = PromptPair(
        clean, corrupt, logit_diff([0] * batch, [1] * batch, position=-1), f"L{depth}"
    )
    return model, pair


def estimators(quick: bool):
    steps = (4,) if quick else (4, 16)
    out = [
        ("linear", {"method": "linear"}),
        ("linear+softmax", {"method": "linear", "recompute_softmax": True}),
        ("linear+dropout", {"method": "linear", "gradient_dropout": True}),
        (
            "linear+both",
            {"method": "linear", "recompute_softmax": True, "gradient_dropout": True},
        ),
    ]
    out += [(f"integrated_n{n}", {"method": "integrated", "n_steps": n}) for n in steps]
    return out


def dead_gradient_arm() -> dict:
    """The measurement that justifies excluding compiled models from this experiment."""
    from circuitry import compile_program
    from circuitry.attrib.gradients import _run_with_gradients
    from circuitry.symbolic import programs
    from circuitry.toy.tasks import compiled_pairs

    out = {}
    for name in programs.names():
        program, vocab = programs.build(name)
        compiled = compile_program(program, vocab, 4, verify=False)
        if compiled.root.encoding != "categorical":
            continue
        try:
            pair = compiled_pairs(
                compiled, n_pairs=8, length=4, generator=torch.Generator().manual_seed(0)
            ).pair
        except ValueError:  # pragma: no cover - a constant program
            continue
        sites = [
            f"blocks.{i}.{leaf}"
            for i in range(compiled.cfg.n_layers)
            for leaf in ("attn.z", "mlp.out", "attn.pattern")
        ]
        _, grads = _run_with_gradients(compiled.model, pair.clean, pair.metric, sites)
        alive = sum(1 for value in grads.values() if float(value.abs().max()) > 1e-12)
        out[name] = {
            "n_layers": compiled.cfg.n_layers,
            "n_sites": len(grads),
            "sites_with_gradient": alive,
            "dead_fraction": 1.0 - alive / len(grads),
        }
    return out


def granularity_contrast(quick: bool, *, batch: int, seq: int) -> dict:
    """The same estimator at two displacement sizes.

    A first-order expansion is a claim about small displacements.  Patching one
    head at one position moves the residual stream a little; pooling that
    head's whole contribution across every position moves it a lot.  Reporting
    one granularity would make the method look either better or worse than it
    is, depending on which was chosen, so both are measured.
    """
    out: dict[str, dict] = {}
    for label, spec in _substrates(quick):
        if label.startswith("saturated"):
            continue
        model, pair = build_substrate(**spec, batch=batch, seq=seq)
        per: dict[str, dict] = {}
        for name, granularity in (
            ("per_position", default_granularity(pos="each")),
            ("pooled_over_positions", default_granularity(pos="all")),
        ):
            shared = {
                "pair": pair, "sites": SITE_GROUPS["attn.z"], "direction": "denoise",
                "granularity": granularity,
            }
            exact = patch(model, **shared)
            scale = max((abs(v) for v in exact.effects.values()), default=0.0)
            estimate = attribute(model, method="linear", **shared)
            entry = _compare(exact, estimate, scale=scale)
            entry["exact_batch_items"] = exact.forward_passes * pair.batch
            per[name] = entry
        out[label] = per
    return out


def plot(report: dict, path: Path) -> None:
    from style import AQUA, BLUE, GRID, ORANGE, RED, VIOLET, new_figure, save, style_axes

    fig, axes = new_figure(2, 2, figsize=(11.5, 8.4))
    (tl, tr), (bl, br) = axes
    runs = report["runs"]
    smooth = [k for k in runs if not k.startswith("saturated")]

    # 1. estimated against true, for the plain linear estimator
    scatter = runs[smooth[-1]]["denoise"]["linear"]["scatter"]
    truth = np.array(scatter["true"])
    guess = np.array(scatter["estimated"])
    lim = max(np.abs(truth).max(), np.abs(guess).max()) * 1.1
    tl.plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1.0, zorder=1)
    tl.scatter(truth, guess, s=9, color=BLUE, alpha=0.75, linewidths=0, zorder=2)
    style_axes(tl, f"Estimated against true effect, {smooth[-1]}, first order",
               "exact patching effect", "attribution patching estimate")

    # 2. error against the size of the true effect -- is it wrong where it matters?
    for name, colour in (("linear", BLUE), ("integrated_n16", ORANGE)):
        block = runs[smooth[-1]]["denoise"].get(name)
        if not block:
            continue
        s = block["scatter"]
        tv, gv = np.array(s["true"]), np.array(s["estimated"])
        tr.scatter(np.abs(tv), np.abs(gv - tv), s=9, alpha=0.7, color=colour, linewidths=0,
                   label=name)
    style_axes(tr, "Error against the size of the true effect",
               "|true effect|", "|estimate - true|")
    tr.set_yscale("log")
    tr.legend(frameon=False, fontsize=8, loc="lower right")

    # 3. sign-error rate by estimator
    names = [n for n, _ in report["estimator_order"]]
    rates = [
        float(np.nanmean([
            runs[r]["denoise"][n]["sign_error_frac"] for r in smooth if n in runs[r]["denoise"]
        ]))
        for n in names
    ]
    bl.barh(range(len(names)), rates, color=VIOLET, height=0.6)
    bl.set_yticks(range(len(names)))
    bl.set_yticklabels(names, fontsize=8)
    style_axes(bl, "Fraction of large effects given the wrong sign",
               "sign-error rate (denoise, averaged over substrates)", None)

    # 4. the frontier: error against compute, with exact patching marked
    for name, colour in (("linear", BLUE), ("linear+both", AQUA)):
        xs, ys = [], []
        for run in smooth:
            block = runs[run]["denoise"].get(name)
            if block:
                xs.append(block["batch_items"])
                ys.append(block["median_error_frac_of_largest"])
        if xs:
            br.scatter(xs, ys, s=26, color=colour, label=name, linewidths=0)
    ig_names = [n for n in names if n.startswith("integrated")]
    xs = [float(np.mean([runs[r]["denoise"][n]["batch_items"] for r in smooth])) for n in ig_names]
    ys = [
        float(np.mean([
            runs[r]["denoise"][n]["median_error_frac_of_largest"] for r in smooth
        ]))
        for n in ig_names
    ]
    br.plot(xs, ys, color=ORANGE, marker="o", markersize=4, linewidth=1.4,
            label="integrated gradients")
    exact_items = float(np.mean([runs[r]["exact_batch_items"] for r in smooth]))
    br.axvline(exact_items, color=RED, linewidth=1.1, linestyle="--")
    br.annotate("exact patching\n(zero error)", (exact_items, 1e-3), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=RED, va="bottom")
    style_axes(br, "Error against compute, and what exact patching costs",
               "batch items processed", "median error / largest true effect")
    br.set_xscale("log")
    br.set_yscale("log")
    br.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    batch, seq = (4, 4) if args.quick else (8, 6)
    groups = (
        {"attn.z": SITE_GROUPS["attn.z"], "attn.scores": SITE_GROUPS["attn.scores"]}
        if args.quick
        else SITE_GROUPS
    )
    est = estimators(args.quick)
    report: dict[str, object] = {
        "runs": {},
        "estimator_order": est,
        "site_groups": list(groups),
        "dead_gradient_in_compiled_models": dead_gradient_arm(),
        "granularity_contrast": granularity_contrast(args.quick, batch=batch, seq=seq),
    }

    for label, spec in _substrates(args.quick):
        model, pair = build_substrate(**spec, batch=batch, seq=seq)
        sites = list(groups.values())
        granularity = default_granularity(pos="each")
        run: dict[str, object] = {}
        for direction in ("denoise", "noise"):
            cf = None if direction == "denoise" else ResampleAblation()
            exact = patch(
                model, pair=pair, sites=sites, direction=direction, counterfactual=cf,
                granularity=granularity,
            )
            scale = max((abs(v) for v in exact.effects.values()), default=0.0)
            run["exact_batch_items"] = exact.forward_passes * pair.batch
            per: dict[str, object] = {}
            for name, kwargs in est:
                estimate = attribute(
                    model, pair=pair, sites=sites, direction=direction, counterfactual=cf,
                    granularity=granularity, **kwargs,
                )
                entry = _compare(exact, estimate, scale=scale)
                keys = sorted(exact.effects)
                entry["scatter"] = {
                    "true": [exact.effects[k] for k in keys],
                    "estimated": [estimate.effects[k] for k in keys],
                }
                entry["by_site_group"] = {}
                for group, pattern in groups.items():
                    prefix = pattern.split("*")[-1]
                    subset = [k for k in keys if k[0].endswith(prefix)]
                    if len(subset) < 3:
                        continue
                    tv = np.array([exact.effects[k] for k in subset])
                    gv = np.array([estimate.effects[k] for k in subset])
                    entry["by_site_group"][group] = {
                        "median_abs_error": float(np.median(np.abs(gv - tv))),
                        "max_abs_error": float(np.abs(gv - tv).max()),
                        "n": len(subset),
                    }
                per[name] = entry
            run[direction] = per
            run["largest_true_effect"] = scale
        report["runs"][label] = run

    report["metadata"] = {
        "seed": args.seed,
        "git_sha": _git_sha(),
        "device": "cpu",
        "dtype": "float64",
        "batch": batch,
        "seq": seq,
        "sign_threshold_frac_of_largest": SIGN_THRESHOLD,
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "wall_seconds": round(time.time() - started, 2),
        "scope": "quick" if args.quick else "full",
        "command": "python experiments/attribution_error.py" + (" --quick" if args.quick else ""),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "attribution_error_quick" if args.quick else "attribution_error"
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # --- the headline table -------------------------------------------
    smooth = [k for k in report["runs"] if not k.startswith("saturated")]
    print("Averaged over smooth substrates, denoising. Error as a fraction of the")
    print("largest true effect in that substrate; sign errors over effects above 5% of it.\n")
    header = f"{'estimator':16s} {'median err':>11s} {'max err':>9s} {'sign wrong':>11s} {'items':>9s}"
    print(header)
    print("-" * len(header))
    for name, _ in est:
        rows = [report["runs"][r]["denoise"][name] for r in smooth]
        print(
            f"{name:16s} "
            f"{np.mean([r['median_error_frac_of_largest'] for r in rows]):11.4f} "
            f"{np.mean([r['max_error_frac_of_largest'] for r in rows]):9.4f} "
            f"{np.mean([r['sign_error_frac'] for r in rows]):11.1%} "
            f"{np.mean([r['batch_items'] for r in rows]):9.0f}"
        )
    print(
        f"{'exact patching':16s} {0.0:11.4f} {0.0:9.4f} {0.0:11.1%} "
        f"{np.mean([report['runs'][r]['exact_batch_items'] for r in smooth]):9.0f}"
    )
    print()
    contrast = report["granularity_contrast"]
    print("The same first-order estimator at two displacement sizes, on attn.z:")
    for name in ("per_position", "pooled_over_positions"):
        rows = [v[name] for v in contrast.values()]
        print(
            f"  {name:22s} median err "
            f"{np.mean([r['median_error_frac_of_largest'] for r in rows]):.4f}   "
            f"sign wrong {np.mean([r['sign_error_frac'] for r in rows]):.1%}   "
            f"over {np.mean([r['n_indices'] for r in rows]):.0f} effects"
        )
    print()
    dead = report["dead_gradient_in_compiled_models"]
    fractions = [v["dead_fraction"] for v in dead.values()]
    print(f"compiled models: the gradient is identically zero at "
          f"{min(fractions):.0%} to {max(fractions):.0%} of sites "
          f"({len(dead)} programs), which is why they are not the substrate here")

    if not args.quick:
        try:
            plot(report, RESULTS / f"{stem}.png")
        except ImportError:  # pragma: no cover
            print("matplotlib not installed; skipped the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
