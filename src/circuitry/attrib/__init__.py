"""Attribution: metrics, exact patching, edge patching, and circuits."""

from __future__ import annotations

from .ablate import AblationEngine
from .acdc import discover, noise_floor
from .circuit import Circuit, Component, Edge, all_components, all_edges
from .gradients import attribute
from .metric import PromptPair, effect_of, kl_from, logit_diff, logprob, measure_baselines
from .metrics import completeness, faithfulness, minimality, recoverable
from .patching import (
    GaussianAblation,
    MeanAblation,
    ResampleAblation,
    ZeroAblation,
    patch,
    patch_sites_together,
)
from .paths import path_decomposition_error, path_patch, path_patch_by_freezing

__all__ = [
    "AblationEngine",
    "Circuit",
    "Component",
    "Edge",
    "GaussianAblation",
    "MeanAblation",
    "PromptPair",
    "ResampleAblation",
    "ZeroAblation",
    "all_components",
    "all_edges",
    "attribute",
    "completeness",
    "discover",
    "effect_of",
    "faithfulness",
    "kl_from",
    "logit_diff",
    "logprob",
    "measure_baselines",
    "minimality",
    "noise_floor",
    "patch",
    "patch_sites_together",
    "path_decomposition_error",
    "path_patch",
    "path_patch_by_freezing",
    "recoverable",
]
