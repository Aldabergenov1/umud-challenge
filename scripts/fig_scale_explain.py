"""Figure explaining px->mm calibration on one test frame (IMG_00001, Siemens, ruler on the right)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from umud_scale import decode_scale

a = np.asarray(Image.open(ROOT / "data/raw/test_images_v2/test_set_v2/IMG_00001.tif").convert("L")).astype(float)
px_cm, (l, t, r, b) = decode_scale(np.asarray(Image.open(ROOT / "data/raw/test_images_v2/test_set_v2/IMG_00001.tif")), "tif")
col = a[:, 1150]
ticks = np.flatnonzero(np.diff((col > 50).astype(int)) == 1) + 1
print("ticks y:", ticks.tolist(), "px/cm:", px_cm)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
ax.imshow(a, cmap="gray")
ax.add_patch(plt.Rectangle((l, t), r - l, b - t, fill=False, ec="yellow", lw=1.5))
ax.add_patch(plt.Rectangle((1120, 60), 60, 670, fill=False, ec="cyan", lw=2))
ax.text(1100, 50, "шкала глубины", color="cyan", ha="right", fontsize=11)
ax.text(l + 5, t - 10, "УЗ-область", color="yellow", fontsize=11)
ax.set_title("IMG_00001.tif (Siemens), кадр 1200×800 пикселей", fontsize=11)
ax.axis("off")

ax = axes[1]
y0, y1 = 60, 730
ax.imshow(a[y0:y1, 1100:1200], cmap="gray", extent=(1100, 1200, y1, y0), aspect=0.25)
for i, y in enumerate(ticks):
    ax.axhline(y, xmin=0.55, xmax=0.75, color="red", lw=1.5)
    ax.text(1182, y + 4, f"{i * 5} мм  (y={y})", color="red", fontsize=8.5)
ax.annotate("", xy=(1125, ticks[-1]), xytext=(1125, ticks[0]),
            arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.5))
ax.text(1103, (ticks[0] + ticks[-1]) / 2,
        f"{ticks[-1] - ticks[0]} px\n= {len(ticks) - 1}×5 мм\n= {(len(ticks) - 1) * 5} мм",
        color="cyan", fontsize=10, va="center")
ax.set_xlim(1100, 1260)
ax.set_title(f"риски через 5 мм → {px_cm:.1f} px/см = {px_cm / 10:.2f} px/мм", fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.savefig(ROOT / "eda/fig_scale_explain.png", dpi=100)
