"""Brightness profiles the ruler decoder reads: Siemens (column 1150) and Lumify (columns 6 and 9)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/raw/test_images_v2/test_set_v2"


def gray(name):
    a = np.asarray(Image.open(TEST / name)).astype(float)
    return a.mean(axis=-1) if a.ndim == 3 else a


fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 2.4]})

# Siemens: one column through the ruler on the right
g = gray("IMG_00001.tif")
col = g[:, 1150]
axes[0, 0].imshow(g[:, 1100:1200], cmap="gray", aspect="auto")
axes[0, 0].axvline(50, color="cyan", lw=1)
axes[0, 0].set_title("IMG_00001 (Siemens): полоса x=1100..1200,\nголубая линия — колонка x=1150", fontsize=10)
axes[0, 0].set_xticks([])
ax = axes[0, 1]
ax.plot(col, color="black", lw=1)
ax.axhline(50, color="red", ls="--", lw=1)
ticks = np.flatnonzero(np.diff((col > 50).astype(int)) == 1) + 1
for y in ticks:
    ax.axvline(y, color="red", alpha=0.3)
ax.set_xlim(0, len(col))
ax.set_xlabel("y, пиксели сверху вниз")
ax.set_ylabel("яркость")
ax.set_title(f"яркость вдоль колонки: каждая риска — всплеск выше порога 50. Найдено {len(ticks)} рисок, "
             f"первая y={ticks[0]}, последняя y={ticks[-1]}", fontsize=10)

# Lumify: short and long ticks, columns 6 and 9
g = gray("IMG_00252.png")
axes[1, 0].imshow(g[:, :40], cmap="gray", aspect="auto")
axes[1, 0].axvline(6, color="cyan", lw=1)
axes[1, 0].axvline(9, color="orange", lw=1)
axes[1, 0].set_title("IMG_00252 (Lumify): левый край x=0..40,\nколонки x=6 (голубая) и x=9 (оранжевая)", fontsize=10)
axes[1, 0].set_xticks([])
ax = axes[1, 1]
ax.plot(g[:, 6], color="tab:cyan", lw=1, label="x=6: короткие и длинные риски")
ax.plot(g[:, 9] + 260, color="tab:orange", lw=1, label="x=9 (сдвинуто вверх): только длинные риски")
ax.axhline(50, color="red", ls="--", lw=1)
ax.set_xlim(0, len(g))
ax.set_xlabel("y, пиксели сверху вниз")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("короткая риска не дотягивается до x=9, длинная дотягивается: так отличаем 5 мм от 1 см", fontsize=10)
plt.tight_layout()
plt.savefig(ROOT / "eda/fig_tick_profiles.png", dpi=90)
print(len(ticks), ticks.tolist())
