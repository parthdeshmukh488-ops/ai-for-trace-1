"""Is this engine's forward pass right, and compared to what?

An independently written transformer is the strongest oracle available for a
transformer: if two implementations agree on every activation site to machine
precision, both are very unlikely to be wrong in the same way.  Every
attribution number in this repository is downstream of this agreeing, which is
why it runs in CI rather than once by hand.

Two comparisons, which test different claims and are reported separately:

* **Export.**  A randomly initialised model here is exported to the reference
  implementation and the two are run on the same inputs.  This checks the
  forward pass with the weight mapping out of the way, needs no network, and
  is therefore the arm that can run in CI.
* **Import.**  A real checkpoint is loaded into this engine and compared
  against the reference running the same checkpoint.  This checks the mapping
  in :mod:`circuitry.model.loading`, which is a different claim and the one
  where a transposed matrix would hide.

Both run with folding off and with folding on.  Off shows the forward pass is
right; on shows the three folds are exact, which they are claimed to be.

Bitwise equality is not expected and not asserted: the two implementations
contract their matrices in different orders, so the measurement *is* the
deviation.  What the script asserts is a bound, and what it reports is the
achieved figure, where in the network it is largest, and whether it grows with
depth at the rate accumulation predicts.  A deviation that is flat in depth is
rounding; one that jumps at a single layer is a bug, and the script says which
layer.

Two things the first version of this script got wrong, both worth stating
because both are easy to get wrong in the same way again.

**The bound is relative to the activation scale, not absolute.**  GPT-2
small's residual stream reaches an RMS of 41 and individual entries above
3000 by the last layer.  An absolute bound of 1e-4 there is a *relative* bound
of 2e-6, which is sixteen times float32 epsilon -- tighter than twelve layers
of accumulation can possibly achieve, so the original absolute bound failed on
a correct implementation.  Measured: the deviation at layer 11 is 4.9e-4
absolute, which is 1.19e-5 relative to the RMS, almost exactly 100x epsilon.
The bound below is set from that measurement with headroom, and is checked
against the per-site RMS.

**Exactness and conditioning are different claims and get different
precisions.**  "The three folds leave the model's function unchanged" is an
algebraic identity, and testing it in float32 measures the identity and the
arithmetic noise together: at full sequence length the folded GPT-2 comparison
lands at 7.7e-5 relative, above a bound that the unfolded comparison clears at
2.3e-5, purely because folding ``gamma`` into ``W_Q`` changes the order the
products are accumulated in.  So the folded arm is run twice -- in float32,
reported but not bounded, because it measures conditioning; and in float64,
bounded tightly, because that is where the identity itself can be seen.

**Folding is behaviour-preserving, not activation-preserving.**  Only the
model's *function* is claimed to be unchanged, and two of the three folds
visibly move the intermediates: ``center_writing_weights`` shifts the residual
stream by a multiple of the all-ones vector, which the next LayerNorm removes
(measured: 4.69 raw at layer 6, 7.3e-4 once the mean is taken out), and
``fold_layernorm`` moves the final gain and bias into the unembedding, so
``ln_final.normalized`` legitimately differs afterwards.  Comparing raw
intermediates across a fold therefore compares two different tensors.  The
folded arm compares the quantities that *are* invariant: the log-softmax of
the logits (raw logits move by a per-position constant under
``center_unembed``), the attention patterns, and the mean-centred residual
stream.

    python benchmarks/cross_check_reference.py [--quick] [--no-import]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from circuitry import ModelConfig
from circuitry.model.hooks import capture
from circuitry.model.loading import (
    center_unembed,
    center_writing_weights,
    fold_layernorm,
    from_reference_state_dict,
    reference_config,
    to_reference,
)
from circuitry.model.transformer import Transformer

RESULTS = Path(__file__).resolve().parent / "results"

# The bound, expressed relative to each site's own RMS -- see the module
# docstring for why an absolute bound is the wrong unit here.  Set from the
# measured worst case on GPT-2 small (1.19e-5 at layer 11, about 100x float32
# epsilon) with a factor of four of headroom.  A real bug -- a transposed
# weight, the wrong activation function, a position off by one -- gives an
# O(1) relative deviation and cannot pass this; accumulation cannot fail it.
TOLERANCE = 5e-5

# Sites whose absolute deviation is still worth bounding directly, because
# their natural scale is O(1) and a relative bound would be slack: attention
# patterns are probabilities and the log-softmax is a log-probability.
ABS_TOLERANCE = 1e-4

# In float64 the folds are an algebraic identity and nothing else, so the bound
# is near the precision floor rather than near an accumulation budget.
TOLERANCE_F64 = 1e-10


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - a tarball checkout has no git
        return "unknown"


def _deviation(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Max absolute deviation, the reference RMS, and their ratio.

    Elementwise relative error is deliberately *not* reported: most entries of
    a residual stream are near zero, dividing by them produces enormous ratios
    that say nothing, and a summary taken over them is dominated by the
    smallest element in the tensor.  Normalising one absolute deviation by the
    tensor's own scale is the quantity that means something.
    """
    diff = (a.double() - b.double()).abs()
    rms = float(b.double().pow(2).mean().sqrt())
    max_abs = float(diff.max())
    return {
        "max_abs": max_abs,
        "rms": rms,
        "rel_rms": max_abs / rms if rms > 0 else 0.0,
    }


def compare(
    model: Transformer,
    reference,
    tokens: torch.Tensor,
    *,
    invariant_only: bool = False,
    rel_bound: float = TOLERANCE,
    abs_bound: float = ABS_TOLERANCE,
) -> dict:
    """Compare every comparable site between the two implementations.

    The alignment is worth stating because it is not the obvious one and
    comparing against the wrong element would look like a deep-layer bug: the
    reference's ``hidden_states`` are the block *inputs*, so element ``i`` is
    this engine's ``resid_pre`` of layer ``i``, and the final element is the
    output of the last LayerNorm rather than the last residual stream.  The
    last block's ``resid_post`` is therefore not in the list at all, and is
    checked indirectly through the logits.

    ``invariant_only`` restricts the comparison to quantities that survive the
    three weight folds -- see the module docstring.  It is not a weaker test of
    the same claim; it is the right test of a different one.
    """
    n_layers = model.cfg.n_layers
    sites = (
        ["embed", "pos_embed", "ln_final.normalized", "logits"]
        + [f"blocks.{i}.resid_pre" for i in range(n_layers)]
        + [f"blocks.{i}.attn.pattern" for i in range(n_layers)]
    )
    with torch.inference_mode():
        with capture(model, sites) as store:
            model(tokens)
        out = reference(tokens, output_hidden_states=True, output_attentions=True)

    if len(out.hidden_states) != n_layers + 1:
        raise AssertionError(
            f"expected {n_layers + 1} reference hidden states, got {len(out.hidden_states)}; "
            "the alignment assumed by this script no longer holds"
        )
    if not out.attentions:
        raise AssertionError(
            "the reference returned no attention patterns; it needs an eager attention "
            "implementation, which circuitry.model.loading.to_reference sets"
        )

    per_site: dict[str, dict[str, float]] = {}

    def record(name: str, mine: torch.Tensor, theirs: torch.Tensor, *, absolute=False) -> None:
        entry = _deviation(mine, theirs)
        entry["bound"] = "absolute" if absolute else "relative_to_rms"
        entry["within_bound"] = (
            entry["max_abs"] < abs_bound if absolute else entry["rel_rms"] < rel_bound
        )
        per_site[name] = entry

    def centred(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)

    if invariant_only:
        # The log-softmax rather than the logits: center_unembed shifts every
        # logit at a position by the same constant, which the softmax ignores.
        record(
            "log_softmax(logits)",
            store["logits"].log_softmax(dim=-1),
            out.logits.log_softmax(dim=-1),
            absolute=True,
        )
        for i in range(n_layers):
            record(
                f"blocks.{i}.resid_pre (centred)",
                centred(store[f"blocks.{i}.resid_pre"]),
                centred(out.hidden_states[i]),
            )
    else:
        record("logits", store["logits"], out.logits)
        record("ln_final.normalized", store["ln_final.normalized"], out.hidden_states[-1])
        record("embed+pos_embed", store["embed"] + store["pos_embed"], out.hidden_states[0])
        for i in range(n_layers):
            record(f"blocks.{i}.resid_pre", store[f"blocks.{i}.resid_pre"], out.hidden_states[i])

    # Attention patterns are invariant under all three folds, so they are
    # compared in both modes -- and on an absolute bound, since they are
    # probabilities and already O(1).
    for i in range(n_layers):
        record(
            f"blocks.{i}.attn.pattern", store[f"blocks.{i}.attn.pattern"], out.attentions[i],
            absolute=True,
        )

    key = "resid_pre (centred)" if invariant_only else "resid_pre"
    resid = [per_site[f"blocks.{i}.{key}"]["rel_rms"] for i in range(n_layers)]
    worst = max(per_site.items(), key=lambda kv: kv[1]["rel_rms"])
    return {
        "per_site": per_site,
        "max_abs": max(v["max_abs"] for v in per_site.values()),
        "max_rel_rms": max(v["rel_rms"] for v in per_site.values()),
        "worst_site": worst[0],
        "within_bounds": all(v["within_bound"] for v in per_site.values()),
        "offenders": sorted(name for name, v in per_site.items() if not v["within_bound"]),
        "resid_rel_rms_by_depth": resid,
        "grows_with_depth": bool(len(resid) > 2 and resid[-1] >= resid[1]),
        "n_sites_compared": len(per_site),
        "mode": "fold-invariant" if invariant_only else "exact",
        "rel_bound": rel_bound,
        "abs_bound": abs_bound,
        "dtype": str(model.dtype).replace("torch.", ""),
    }


def export_arm(*, quick: bool, seed: int = 0) -> dict:
    """Random weights here, exported to the reference.  No network needed."""
    cfg = ModelConfig(
        n_layers=2 if quick else 6,
        n_heads=2 if quick else 4,
        d_model=16 if quick else 64,
        d_mlp=32 if quick else 256,
        d_vocab=20 if quick else 128,
        n_ctx=16 if quick else 64,
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch, seq = (2, 6) if quick else (4, 32)
    tokens = torch.randint(0, cfg.d_vocab, (batch, seq), generator=gen)

    out: dict[str, dict] = {}
    model = Transformer(cfg, generator=gen).eval()
    out["unfolded"] = compare(model, to_reference(model), tokens)

    # Folded here, unfolded there: the folds are claimed to be exact, so the
    # two must still agree.  Comparing a folded model against a folded
    # reference would be the weaker test -- it would pass even if both were
    # wrong in the same way.
    for dtype, label, bounded in (
        (torch.float32, "folded_f32", False),
        (torch.float64, "folded_f64", True),
    ):
        fresh = Transformer(cfg, generator=torch.Generator().manual_seed(seed)).eval()
        reference = to_reference(fresh).to(dtype)
        fresh.to(dtype)
        center_unembed(center_writing_weights(fold_layernorm(fresh)))
        out[label] = compare(
            fresh, reference, tokens, invariant_only=True,
            rel_bound=(TOLERANCE_F64 if dtype is torch.float64 else float("inf")),
            abs_bound=(TOLERANCE_F64 if dtype is torch.float64 else float("inf")),
        )
        out[label]["bounded"] = bounded
    out["config"] = {
        "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp, "d_vocab": cfg.d_vocab, "batch": batch, "seq": seq,
    }
    return out


def import_arm(*, quick: bool, name: str = "gpt2", seed: int = 0) -> dict:
    """A real checkpoint loaded here, compared against the reference running it."""
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(name, attn_implementation="eager").eval()
    cfg = reference_config(hf.config)
    model = from_reference_state_dict(cfg, hf.state_dict()).eval()

    gen = torch.Generator()
    gen.manual_seed(seed)
    batch, seq = (1, 8) if quick else (2, 24)
    tokens = torch.randint(0, cfg.d_vocab, (batch, seq), generator=gen)

    out: dict[str, dict] = {"unfolded": compare(model, hf, tokens)}
    for dtype, label, bounded in (
        (torch.float32, "folded_f32", False),
        (torch.float64, "folded_f64", True),
    ):
        fresh = from_reference_state_dict(cfg, hf.state_dict()).eval().to(dtype)
        center_unembed(center_writing_weights(fold_layernorm(fresh)))
        out[label] = compare(
            fresh, hf.to(dtype), tokens, invariant_only=True,
            rel_bound=(TOLERANCE_F64 if dtype is torch.float64 else float("inf")),
            abs_bound=(TOLERANCE_F64 if dtype is torch.float64 else float("inf")),
        )
        out[label]["bounded"] = bounded
    hf.to(torch.float32)
    out["checkpoint"] = {"name": name, "n_layers": cfg.n_layers, "d_model": cfg.d_model}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small model, short sequences")
    parser.add_argument("--no-import", action="store_true", help="skip the checkpoint arm")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    report: dict[str, object] = {}
    report["export"] = export_arm(quick=args.quick, seed=args.seed)

    if args.no_import:
        report["import"] = {"skipped": "disabled with --no-import"}
    else:
        try:
            report["import"] = import_arm(quick=args.quick, seed=args.seed)
        except Exception as exc:  # network, or no local cache
            report["import"] = {"skipped": f"{type(exc).__name__}: {exc}"[:300]}

    report["metadata"] = {
        "seed": args.seed,
        "git_sha": _git_sha(),
        "device": "cpu",
        "dtype": "float32",
        "tolerance_rel_rms": TOLERANCE,
        "tolerance_abs": ABS_TOLERANCE,
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "wall_seconds": round(time.time() - started, 2),
        "scope": "quick" if args.quick else "full",
        "command": "python benchmarks/cross_check_reference.py" + (" --quick" if args.quick else ""),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / ("cross_check_quick.json" if args.quick else "cross_check.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    failures = []
    for arm in ("export", "import"):
        block = report[arm]
        if "skipped" in block:
            print(f"{arm:7s} skipped: {block['skipped']}")
            continue
        for folding in ("unfolded", "folded_f32", "folded_f64"):
            res = block[folding]
            bounded = res.get("bounded", True)
            flag = "ok " if res["within_bounds"] else "FAIL"
            if not bounded:
                flag = "-- "
            print(
                f"{flag} {arm:7s} {folding:10s} max_abs={res['max_abs']:.3e} "
                f"rel_to_rms={res['max_rel_rms']:.3e} worst={res['worst_site']} "
                f"({res['n_sites_compared']} sites, {res['mode']}, {res['dtype']}"
                + ("" if bounded else ", reported only") + ")"
            )
            if bounded and not res["within_bounds"]:
                failures.append(f"{arm}/{folding}")
                print(f"     over bound: {', '.join(res['offenders'][:6])}")
                depths = res["resid_rel_rms_by_depth"]
                if len(depths) > 2:
                    jump = max(range(1, len(depths)), key=lambda i: depths[i] - depths[i - 1])
                    print(f"     largest increase at layer {jump}; relative deviation by depth: "
                          + ", ".join(f"{d:.1e}" for d in depths))

    print(f"wrote {path}")
    if failures:
        print(f"FAILED: {', '.join(failures)} exceeded the bound")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
