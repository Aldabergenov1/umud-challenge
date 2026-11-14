"""Figure 1 of the paper: one UMUD benchmark image through the pipeline. CPU only.

    python paper/figures/fig_pipeline.py [im_16_arch]

The default image has the median leave-one-out score of the 35 benchmark images. The benchmark is CC BY 4.0;
competition images are not shown because the competition rules forbid publishing them.
Uses the network maps of the final run (work/v9/b_mix/bench_probs.pkl).
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(OUT))
import style
from core import load_pipeline

style.apply()
plt.rcParams.update({"axes.grid": False})
RED, BLUE, GREEN, GOLD, PINK, CYAN = "#e5484d", "#3e8ef7", "#30c27a", "#f5b700", "#e05bb0", "#19c3d6"
ARROW = "#444444"

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "im_16_arch"
P = load_pipeline()
CFG, TOL = P["CFG"], P["TOL"]
CFG["pa_offset"] = 0.0          # the PA offset belongs to the test raters, not to the benchmark
k = CFG["px_per_mm"]
bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
i = bench.index[bench.ImageID == IMAGE][0]
raters = [f"R{j}" for j in range(1, 8)]
truth = pd.DataFrame({t: bench[[f"{r}_{t.upper()}" for r in raters]].mean(axis=1) for t in TOL})
per_rater = bench.loc[i, [f"{r}_{t.upper()}" for r in raters for t in TOL]].astype(float)
spread = {t: (per_rater.filter(like=f"_{t.upper()}").min(), per_rater.filter(like=f"_{t.upper()}").max()) for t in TOL}

rois, probs = pickle.load(open(ROOT / "work/v9/b_mix/bench_probs.pkl", "rb"))
raw = pd.DataFrame([P["measure"](r, p.astype(np.float32), s / 10)
                    for r, p, s in zip(rois, probs, bench.Scale_pixel_per_cm)])
keep = np.arange(len(raw)) != i
calib, _ = P["fit_calibration"](raw[keep].reset_index(drop=True), truth[keep].reset_index(drop=True))
final = P["apply_calibration"](raw.iloc[[i]], calib).iloc[0]

full = P["read_gray"](P["BENCH"] / f"{IMAGE}.tif")
l, t, rr, b = P["find_roi"](full)
roi, prob = rois[i], probs[i].astype(np.float32)
res, d = P["measure"](roi, prob, bench.Scale_pixel_per_cm[i] / 10, return_debug=True)

# layout: images keep their aspect ratio and share a height; narrow columns hold the arrows
asp_full = full.shape[1] / full.shape[0]
asp_roi = roi.shape[1] / roi.shape[0]
gap = 0.36
W = style.TEXTWIDTH
h_img = W / (asp_full + 2 * asp_roi + 2 * gap)
fig = plt.figure(figsize=(W, 2 * h_img + 0.62))
gs = fig.add_gridspec(2, 5, width_ratios=[asp_full, gap, asp_roi, gap, asp_roi], height_ratios=[1, 1],
                      left=0.008, right=0.992, bottom=0.0, top=0.93, hspace=0.30, wspace=0)
ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])
ax_d, ax_e = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2:])
for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=9, fontweight="bold", pad=3)


# (a) input image and ultrasound box
ax_a.imshow(full, cmap="gray")
ax_a.add_patch(Rectangle((l, t), rr - l, b - t, fill=False, ec=GOLD, lw=1.2))
title(ax_a, "(a) Input image")

# (b) network outputs on the ultrasound box
ax_b.imshow(roi, cmap="gray")
for p, color, thr in ((prob[2], GREEN, 0.2), (prob[0], RED, 0.3), (prob[1], BLUE, 0.3)):
    rgba = np.zeros((*p.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.clip((p - thr) / (1 - thr), 0, 1) * 0.8
    ax_b.imshow(rgba)
title(ax_b, "(b) U-Net outputs")

# (c) geometry on the metric grid
gm = d["gm"]
H, Wg = gm.shape
ax_c.imshow(gm, cmap="gray")
xs = np.array([0, Wg])
(a1, b1), (a2, b2) = d["sup"], d["deep"]
ax_c.plot(xs, a1 * xs + b1, color=RED, lw=1.2)
ax_c.plot(xs, a2 * xs + b2, color=BLUE, lw=1.2)
for x in d["xs_mt"]:
    ax_c.plot([x, x], [a1 * x + b1, a2 * x + b2], color=PINK, lw=1.0)
for _, length, x, y, th in d["segs"]:
    dx, dy = np.cos(th) * length / 2, np.sin(th) * length / 2
    ax_c.plot([x - dx, x + dx], [y - dy, y + dy], color=GOLD, lw=1.0)
ax_c.plot([0.04 * Wg, 0.04 * Wg + 10 * k], [0.93 * H] * 2, color="white", lw=1.8)
ax_c.text(0.04 * Wg, 0.90 * H, "10 mm", color="white", fontsize=7, va="bottom")
ax_c.set_xlim(0, Wg); ax_c.set_ylim(H, 0)
title(ax_c, "(c) Geometry")

# (d) the output as a straight fascicle at the final PA, centred between the two lines
ax_d.imshow(gm, cmap="gray")
ax_d.plot(xs, a1 * xs + b1, color=RED, lw=1.2)
ax_d.plot(xs, a2 * xs + b2, color=BLUE, lw=1.2)
t_deep = np.arctan(a2)
side = 1.0 if ((d["t_band"] - t_deep + np.pi / 2) % np.pi - np.pi / 2) > 0 else -1.0
th = t_deep + side * np.radians(final.pa)
ux, uy = np.cos(th), np.sin(th)
ux, uy = (-ux, -uy) if uy > 0 else (ux, uy)


def to_superficial(x0):
    """From the point of the deep line at x0, up to the superficial line."""
    y0 = a2 * x0 + b2
    s_ = (a1 * x0 + b1 - y0) / (uy - a1 * ux)
    return y0, x0 + s_ * ux, y0 + s_ * uy


_, x1, _ = to_superficial(Wg / 2)
x0 = Wg / 2 - (x1 - Wg / 2) / 2
y0, x1, y1 = to_superficial(x0)
ax_d.plot([x0, x1], [y0, y1], color=CYAN, lw=1.6)
ax_d.set_xlim(0, Wg); ax_d.set_ylim(H, 0)
title(ax_d, "(d) Output fascicle")

# (e) the numbers at each stage
rows = [("raw geometry", f"{res['pa']:.1f}", f"{res['mt'] / np.sin(np.radians(res['pa'])):.1f}", f"{res['mt']:.1f}", "normal", style.INK),
        ("+ calibration (output)", f"{final.pa:.1f}", f"{final.fl:.1f}", f"{final.mt:.1f}", "bold", style.INK),
        ("experts, mean", f"{truth.pa[i]:.1f}", f"{truth.fl[i]:.1f}", f"{truth.mt[i]:.1f}", "normal", style.MUTED),
        ("experts, range", *(f"{spread[t][0]:.1f}–{spread[t][1]:.1f}" for t in ("pa", "fl", "mt")), "normal", style.MUTED)]
cols_x = [0.0, 0.52, 0.73, 0.94]
y_top, dy = 0.86, 0.19
for x, h in zip(cols_x, ["", "PA (°)", "FL (mm)", "MT (mm)"]):
    ax_e.text(x, y_top, h, transform=ax_e.transAxes, fontsize=8, ha="left" if x == 0 else "right", color=style.MUTED)
ax_e.plot([0, 0.94], [y_top - 0.07] * 2, color="#bbbbbb", lw=0.6, transform=ax_e.transAxes)
for j, (name, pa, fl, mt, weight, color) in enumerate(rows):
    y = y_top - dy * (j + 1) - (0.06 if j >= 2 else 0)
    ax_e.text(0, y, name, transform=ax_e.transAxes, fontsize=8, fontweight=weight, color=color)
    for x, v in zip(cols_x[1:], (pa, fl, mt)):
        ax_e.text(x, y, v, transform=ax_e.transAxes, fontsize=8, ha="right", fontweight=weight, color=color)
ax_e.plot([0, 0.94], [y_top - dy * 2.55] * 2, color="#bbbbbb", lw=0.6, transform=ax_e.transAxes)
ax_e.set_xlim(0, 1); ax_e.set_ylim(0, 1)
title(ax_e, "(e) Numbers for this image")


def arrow(ax1, xy1, ax2, xy2, label=None):
    fig.add_artist(ConnectionPatch(xy1, xy2, coordsA="axes fraction", coordsB="axes fraction", axesA=ax1, axesB=ax2,
                                   arrowstyle="-|>", mutation_scale=9, lw=0.9, color=ARROW))
    if label:
        x = (ax1.transAxes.transform(xy1)[0] + ax2.transAxes.transform(xy2)[0]) / 2
        xf, yf = fig.transFigure.inverted().transform((x, ax1.transAxes.transform(xy1)[1]))
        fig.text(xf, yf + 0.012, label, ha="center", va="bottom", fontsize=7, color=ARROW, linespacing=1.0)


arrow(ax_a, (1.02, 0.5), ax_b, (-0.03, 0.5), "crop,\nsegment")
arrow(ax_b, (1.03, 0.5), ax_c, (-0.03, 0.5), "lines,\nfragments")
x_c = (asp_roi + gap + asp_roi / 2) / (2 * asp_roi + gap)   # centre of panel (c) in table coordinates
arrow(ax_c, (0.5, -0.03), ax_e, (x_c, 1.0))
arrow(ax_d, (1.03, 0.5), ax_e, (-0.03, 0.5))

fig.savefig(OUT / "fig1_pipeline.pdf", dpi=300)
print([r[:4] for r in rows])
