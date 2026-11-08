"""Does a compiled program compute exactly what its source says, and at what scale?

The compiler claims to turn a program into a transformer that computes it
*exactly*.  That claim is checkable in a way almost nothing else in
interpretability is: the reference interpreter in
:mod:`circuitry.symbolic.lang` was written first, is a few dozen lines of
plain Python over plain lists, and can be run on every input of a small
vocabulary.  So this experiment is not a demonstration -- it is the
measurement that licenses everything downstream, because every ground-truth
circuit in this repository is only as good as the compiled model matching its
source.

The free parameter is the attention logit scale.  A selector is realised as a
score of ``lambda`` where the predicate holds and ``0`` where it does not, so
softmax approaches hard attention as ``lambda`` grows and the residual
attention paid to non-matching positions goes as ``e^(-lambda)``.  The
beginning-of-sequence position, which supplies the default when a selector
matches nothing, sits at *half* score and therefore converges as
``e^(-lambda/2)``.  The prediction is that the empty-selection path is the
first thing to fail as the scale drops, and that programs whose selectors have
many non-matching keys fail before programs whose selectors are tight.

Both precisions are measured, because the specification asserted that float32
would overflow and bound the scale from above.  It does not: ``torch.softmax``
subtracts the row maximum before exponentiating, so there is nothing to
overflow, and float32 is exact for every program at every scale tested up to
100.  Its only cost is on the one program that reads a magnitude back, where
the floor is ``3.3e-8`` instead of float64's ``1.1e-16``.  That is the whole
reason compiled models are float64, and it is a smaller reason than the
specification claimed.

    python experiments/compiler_fidelity.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from circuitry.symbolic import programs
from circuitry.symbolic.compiler import DEFAULT_LOGIT_SCALE, compile_program
from circuitry.symbolic.lang import Encoding, evaluate

RESULTS = Path(__file__).resolve().parent / "results"
SCALES = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 100.0)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _all_inputs(vocab, max_len):
    for length in range(1, max_len + 1):
        yield from (list(s) for s in itertools.product(vocab, repeat=length))


def measure_program(name: str, max_len: int, scales, dtypes) -> dict:
    program, vocab = programs.build(name)
    spec = programs.PROGRAMS[name]
    numerical = program.encoding == Encoding.NUMERICAL
    cases = [(seq, evaluate(program, seq)) for seq in _all_inputs(vocab, max_len)]
    # A case with an empty selection somewhere is the one the half-score
    # beginning-of-sequence path has to handle, and it is reported separately
    # because it is predicted to converge more slowly.
    out: dict[str, dict] = {}
    for dtype in dtypes:
        per_scale = {}
        for scale in scales:
            compiled = compile_program(
                program, vocab, max_len, logit_scale=scale, verify=False
            )
            if dtype == "float32":
                compiled.model.to(torch.float32)
            exact = 0
            worst = 0.0
            first_failure = None
            for sequence, expected in cases:
                got = compiled.run(sequence)
                if numerical:
                    gap = max(
                        (abs(float(a) - float(b)) for a, b in zip(expected, got, strict=True)),
                        default=0.0,
                    )
                    ok = gap < 1e-6
                else:
                    gap = 0.0 if list(got) == list(expected) else 1.0
                    ok = gap == 0.0
                worst = max(worst, gap)
                if ok:
                    exact += 1
                elif first_failure is None:
                    first_failure = {
                        "input": [str(s) for s in sequence],
                        "expected": [str(v) for v in expected],
                        "got": [str(v) for v in got],
                    }
            per_scale[str(scale)] = {
                "exact_match_rate": exact / len(cases),
                "n_inputs": len(cases),
                "max_error": worst,
                "first_failure": first_failure,
            }
        out[dtype] = per_scale
    shape = compile_program(program, vocab, max_len, verify=False).cfg
    graph = compile_program(program, vocab, max_len, verify=False).graph
    return {
        "per_dtype": out,
        "depth": spec.depth,
        "exercises": spec.exercises,
        "vocab_size": len(vocab),
        "n_inputs": len(cases),
        "model": {
            "n_layers": shape.n_layers, "n_heads": shape.n_heads, "d_model": shape.d_model,
            "d_head": shape.d_head, "d_mlp": shape.d_mlp,
        },
        "graph": {
            "nodes": graph.n_nodes, "edges": graph.n_edges, "unused_heads": len(graph.unused),
        },
    }


def smallest_exact_scale(per_scale: dict) -> float | None:
    """The smallest scale at which every input is exact, and stays exact above."""
    items = sorted(((float(k), v) for k, v in per_scale.items()), key=lambda kv: kv[0])
    best = None
    for scale, entry in reversed(items):
        if entry["exact_match_rate"] < 1.0:
            break
        best = scale
    return best


def plot(report: dict, path: Path) -> None:
    from style import AQUA, BLUE, MUTED, ORANGE, RED, VIOLET, new_figure, save, style_axes

    palette = [BLUE, ORANGE, AQUA, VIOLET, RED, "#8a6d3b", "#2f6f6f", "#9b3b6a", "#4a4a9b"]
    names = list(report["programs"])
    fig, (left, right) = new_figure(1, 2, figsize=(11.5, 4.4))

    for i, name in enumerate(names):
        per = report["programs"][name]["per_dtype"]["float64"]
        xs = sorted(float(k) for k in per)
        ys = [per[str(x)]["exact_match_rate"] for x in xs]
        left.plot(xs, ys, color=palette[i % len(palette)], linewidth=1.6, marker="o",
                  markersize=3)
        left.annotate(name, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                      fontsize=7.5, color=palette[i % len(palette)], va="center")
    style_axes(left, "Exact agreement with the interpreter, float64",
               "attention logit scale", "fraction of inputs exactly right")
    left.set_xscale("log", base=2)
    left.set_ylim(-0.05, 1.12)
    left.set_xlim(0.85, max(SCALES) * 2.6)
    chosen = report["chosen_scale"]
    left.axvline(chosen, color=MUTED, linewidth=0.9, linestyle=":")
    left.annotate(f"default = {chosen:g}", (chosen, -0.02), xytext=(4, 0),
                  textcoords="offset points", fontsize=7.5, color=MUTED)

    for i, name in enumerate(names):
        for dtype, style in (("float64", "-"), ("float32", "--")):
            per = report["programs"][name]["per_dtype"].get(dtype)
            if not per:
                continue
            xs = sorted(float(k) for k in per)
            ys = [max(per[str(x)]["max_error"], 1e-17) for x in xs]
            right.plot(xs, ys, color=palette[i % len(palette)], linewidth=1.3,
                       linestyle=style, alpha=0.9)
    style_axes(right, "Worst error per input, solid float64 and dashed float32",
               "attention logit scale", "max error over all inputs")
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    max_len = 3 if args.quick else 5
    scales = (4.0, 16.0, 32.0) if args.quick else SCALES
    dtypes = ("float64",) if args.quick else ("float64", "float32")
    names = programs.names()[:3] if args.quick else programs.names()

    report: dict[str, object] = {"programs": {}}
    for name in names:
        report["programs"][name] = measure_program(name, max_len, scales, dtypes)

    thresholds = {
        name: smallest_exact_scale(block["per_dtype"]["float64"])
        for name, block in report["programs"].items()
    }
    report["smallest_exact_scale"] = thresholds
    needed = [v for v in thresholds.values() if v is not None]
    report["chosen_scale"] = DEFAULT_LOGIT_SCALE
    report["scale_sufficient_for_all"] = bool(
        needed and len(needed) == len(thresholds) and max(needed) <= DEFAULT_LOGIT_SCALE
    )
    report["metadata"] = {
        "seed": args.seed,
        "git_sha": _git_sha(),
        "device": "cpu",
        "max_len": max_len,
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "wall_seconds": round(time.time() - started, 2),
        "scope": "quick" if args.quick else "full",
        "command": "python experiments/compiler_fidelity.py" + (" --quick" if args.quick else ""),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "compiler_fidelity_quick" if args.quick else "compiler_fidelity"
    (RESULTS / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"{'program':14s} {'inputs':>7s} {'exact@default':>14s} {'min scale':>10s}  model")
    for name in names:
        block = report["programs"][name]
        per = block["per_dtype"]["float64"]
        at_default = per.get(str(float(DEFAULT_LOGIT_SCALE)))
        rate = at_default["exact_match_rate"] if at_default else float("nan")
        shape = block["model"]
        print(
            f"{name:14s} {block['n_inputs']:7d} {rate:14.3f} "
            f"{thresholds[name]!s:>10s}  "
            f"L{shape['n_layers']} H{shape['n_heads']} d{shape['d_model']} "
            f"dm{shape['d_mlp']} ({block['graph']['edges']} edges, "
            f"{block['graph']['unused_heads']} unused)"
        )
    print()
    # The decay rate is only meaningful where the error is a magnitude.  For a
    # categorical program the error is 0 or 1 -- the output is either the right
    # symbol or the wrong one -- so a fitted exponent would be an artefact of
    # the indicator, not a measurement of the softmax.  Those programs are
    # characterised by their threshold scale instead.
    import math

    decay_rates: dict[str, float] = {}
    for name in names:
        per = report["programs"][name]["per_dtype"]["float64"]
        smallest = min(per, key=float)
        if per[smallest]["max_error"] in (0.0, 1.0):
            continue
        pairs = sorted(((float(k), v["max_error"]) for k, v in per.items()), key=lambda kv: kv[0])
        ratios = [
            (b[1] / a[1], b[0] - a[0])
            for a, b in itertools.pairwise(pairs)
            if b[1] > 1e-15 and a[1] > 0
        ]
        if ratios:
            decay_rates[name] = sum(math.log(1 / r) / d for r, d in ratios) / len(ratios)
    report["measured_decay_rate"] = decay_rates
    if decay_rates:
        print("error decay, where the error is a magnitude rather than an indicator:")
        for name, decay in decay_rates.items():
            print(
                f"  {name:14s} error ~ exp(-{decay:.2f} * lambda)   "
                "[the half-score BOS path predicts 0.50; a plain match predicts 1.00]"
            )
    print()
    print(f"smallest scale that is exact for every program: {max(needed) if needed else None}")
    print(f"default scale {DEFAULT_LOGIT_SCALE:g} sufficient for all: "
          f"{report['scale_sufficient_for_all']}")

    if not args.quick:
        try:
            plot(report, RESULTS / f"{stem}.png")
        except ImportError:
            print("matplotlib not installed; skipped the figure")
    return 0 if report["scale_sufficient_for_all"] or args.quick else 1


if __name__ == "__main__":
    raise SystemExit(main())
