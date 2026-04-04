"""Publication-quality plot styling for NeuroCGMD analysis.

Provides consistent, visually striking defaults and helper functions
for creating presentation-ready figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

CHAIN_COLORS = [
    "#E63946",  # red
    "#457B9D",  # steel blue
    "#2A9D8F",  # teal
    "#E9C46A",  # gold
    "#F4A261",  # sandy orange
    "#264653",  # dark blue
    "#8338EC",  # purple
    "#06D6A0",  # mint
]

ENERGY_COLORS = {
    "potential": "#1D3557",
    "kinetic": "#E63946",
    "total": "#457B9D",
    "bonded": "#2A9D8F",
    "nonbonded": "#E9C46A",
}

EVENT_COLORS = {
    "bond_forming": "#2A9D8F",
    "bond_breaking": "#E63946",
    "conformational_shift": "#457B9D",
    "interface_rearrangement": "#F4A261",
}

# Custom colormaps
_CONTACT_CMAP_DATA = {
    "red": [(0, 1, 1), (0.5, 0.97, 0.97), (1, 0.13, 0.13)],
    "green": [(0, 1, 1), (0.5, 0.78, 0.78), (1, 0.24, 0.24)],
    "blue": [(0, 1, 1), (0.5, 0.44, 0.44), (1, 0.36, 0.36)],
}
CONTACT_CMAP = LinearSegmentedColormap("NeuroCGMD_Contact", _CONTACT_CMAP_DATA)

_ENERGY_CMAP_DATA = {
    "red": [(0, 0.08, 0.08), (0.5, 0.97, 0.97), (1, 0.9, 0.9)],
    "green": [(0, 0.24, 0.24), (0.5, 0.95, 0.95), (1, 0.22, 0.22)],
    "blue": [(0, 0.36, 0.36), (0.5, 0.85, 0.85), (1, 0.17, 0.17)],
}
ENERGY_LANDSCAPE_CMAP = LinearSegmentedColormap("NeuroCGMD_FEL", _ENERGY_CMAP_DATA)


def setup_style():
    """Apply publication-quality matplotlib style globally."""
    mpl.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": True,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def styled_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create a figure with consistent styling."""
    if figsize is None:
        w = 5.5 * ncols
        h = 3.8 * nrows
        figsize = (w, h)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def add_watermark(fig, text="NeuroCGMD"):
    """Add a subtle watermark to the figure."""
    fig.text(0.99, 0.01, text, fontsize=7, color="#CCCCCC",
             ha="right", va="bottom", style="italic", alpha=0.6)


def smooth(values, window=5):
    """Simple moving average for smoother plot lines."""
    if len(values) < window:
        return values
    kernel = [1.0 / window] * window
    padded = [values[0]] * (window // 2) + list(values) + [values[-1]] * (window // 2)
    return [sum(padded[i:i + window][j] * kernel[j] for j in range(window))
            for i in range(len(values))]
