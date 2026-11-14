"""Which fascicle fragments drive PA on given test frames: drawn coloured by their angle, plus the histogram."""
import io
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from core import load_pipeline

P = load_pipeline()
meta = P["test_meta"]().set_index("image_id")
rois, probs = pickle.load(open(ROOT / "work/test_probs.pkl", "rb"))
ids = sys.argv[1:] or ["IMG_00062.tif", "IMG_00099.tif", "IMG_00126.tif", "IMG_00138.tif"]

fig, axes = plt.subplots(2, len(ids), figsize=(6 * len(ids), 9), gridspec_kw={"height_ratios": [2, 1]})
cmap = plt.get_cmap("coolwarm")
for col, i in enumerate(ids):
    r = meta.loc[i]
    res, dbg = P["measure"](rois[i], probs[i].astype(np.float32), r.px_per_cm / 10, return_debug=True)
    ax = axes[0, col]
    ax.imshow(dbg["gm"], cmap="gray")
    W = dbg["gm"].shape[1]
    for a, b in (dbg["sup"], dbg["deep"]):
        ax.plot([0, W], [b, a * W + b], color="white", lw=1)
    segs = dbg["segs"]
    for pa, length, x, y, t in segs:
        dx, dy = np.cos(t) * length / 2, np.sin(t) * length / 2
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=cmap(np.clip((pa - 5) / 25, 0, 1)), lw=2)
    ax.set_title(f"{i} ({r.layout})\nPA сегменты {res['pa_seg']:.1f}°, tensor {res['pa_tensor']:.1f}°, n={len(segs)}", fontsize=10)
    ax.axis("off")
    pas, lens = np.array([s[0] for s in segs]), np.array([s[1] for s in segs]) / P["CFG"]["px_per_mm"]
    axes[1, col].hist(pas, bins=np.arange(0, 40, 1.5), weights=lens, color="gray")
    axes[1, col].axvline(res["pa_seg"], color="red")
    axes[1, col].set_xlabel("угол фрагмента к глубокому апоневрозу, ° (вес = длина, мм)")
    print(i, "fragments (PA°, length mm):", sorted(((round(p, 1), round(l, 1)) for p, l in zip(pas, lens)), key=lambda z: -z[1])[:12])
plt.tight_layout()
plt.savefig(ROOT / "work/diag_segments.png", dpi=70)
