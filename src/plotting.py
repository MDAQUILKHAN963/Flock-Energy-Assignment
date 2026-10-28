"""Shared matplotlib style so every figure in the report reads as one set."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config

# Categorical slots, used in this fixed order (validated colour-blind-safe set).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#a3a29c"
SURFACE, GRID, SHADE = "#fcfcfb", "#e6e5e0", "#ecebe7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})


def shade_spans(ax, spans, label=None):
    """Shade (start, end) intervals, e.g. outages, behind the data."""
    for i, (start, end) in enumerate(spans):
        ax.axvspan(start, end, color=SHADE, lw=0, zorder=0, label=label if i == 0 else None)


def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path
