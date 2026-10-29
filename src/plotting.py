"""Shared matplotlib style so every figure in the report reads as one set."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# Categorical colours, used in this fixed order (validated colour-blind-safe set).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#a3a29c"
SURFACE, GRID, SHADE = "#fcfcfb", "#e6e5e0", "#ecebe7"
LIGHT_BLUE = "#cde2fb"

# Sequential (magnitude) ramp: one hue, light -> dark.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

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
