"""Shared figure styling.

Colours come from a palette validated for colour-vision deficiency and for
contrast against the chart surface, so the figures stay readable in print and
for colourblind readers.  Every series is also directly labelled rather than
identified by colour alone, which is the other half of the same requirement.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d8d4"
SURFACE = "#fcfcfb"

# Categorical slots 1-5 of the validated palette; these clear the all-pairs
# separation floors in both light and dark modes.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
RED = "#e34948"


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Recessive grid, no chart junk, left-aligned title."""
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)
    return ax


def new_figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
