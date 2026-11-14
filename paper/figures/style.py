"""Shared look of the paper figures: Latin Modern like the LaTeX text, sizes in final print inches."""
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

TEXTWIDTH = 6.378  # inches: A4 with 2.4 cm margins, as in main.tex
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _latin_modern():
    """Register Latin Modern Roman from the TeX installation; False if it is not there."""
    try:
        path = subprocess.run(["kpsewhich", "lmroman10-regular.otf"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return False
    if not path:
        return False
    folder = Path(path).parent
    for name in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic", "lmroman10-bolditalic"):
        if (folder / f"{name}.otf").exists():
            font_manager.fontManager.addfont(str(folder / f"{name}.otf"))
    return True


def apply():
    plt.rcParams.update({
        "font.family": "Latin Modern Roman" if _latin_modern() else "serif",
        "mathtext.fontset": "cm",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
