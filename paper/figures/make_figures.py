"""Figures 2-4 of the paper, from cached network outputs and the manual test labels. No GPU needed.

    python paper/figures/make_figures.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]

sys.path.insert(0, str(OUT))
import style
from style import BLUE, INK, MUTED, ORANGE, TEXTWIDTH

style.apply()
MAPS = ROOT / "work/v9/b_mix"  # network maps of the final submission


def alpha_hat(a_deg, r):
    """Orientation estimate when speckle adds horizontal tensor energy r relative to the fascicles."""
    a = np.radians(a_deg)
    return np.degrees(0.5 * np.arctan2(np.sin(2 * a), np.cos(2 * a) + r))


def fig_speckle():
    from core import load_pipeline
    P = load_pipeline()
    bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
    truth = bench[[f"R{i}_PA" for i in range(1, 8)]].mean(axis=1).values
    rois, probs = pickle.load(open(MAPS / "bench_probs.pkl", "rb"))
    rows = []
    for roi, p, s, y in zip(rois, probs, bench.Scale_pixel_per_cm, truth):
        gm, pm = P["to_metric"](roi, p.astype(np.float32), s / 10)
        sup, deep = P["fit_edge_line"](pm[0], "lower"), P["fit_edge_line"](pm[1], "upper")
        if sup is None or deep is None:
            continue
        H, W = gm.shape
        yy, xx = np.mgrid[0:H, 0:W]
        depth = (yy - (sup[0] * xx + sup[1])) / np.maximum((deep[0] - sup[0]) * xx + deep[1] - sup[1], 1)
        region = (depth > 0.12) & (depth < 0.88) & (xx > 0.1 * W) & (xx < 0.9 * W)
        t_deep = np.arctan(deep[0])
        row = dict(truth=y)
        for name, img, weight in (("raw", gm.astype(np.float32), lambda c: c),
                                  ("fmap", pm[2], lambda c: c * pm[2] ** 2)):
            theta, coh = P["orientation_field"](img)
            row[name] = P["angle_between"](P["mean_orientation"](theta, weight(coh) * region), t_deep)
        rows.append(row)
    d = pd.DataFrame(rows)
    r = minimize_scalar(lambda r: np.abs(alpha_hat(d.truth.values, r) - d.raw.values).mean(),
                        bounds=(0, 3), method="bounded").x

    fig, ax = plt.subplots(figsize=(0.55 * TEXTWIDTH, 0.55 * TEXTWIDTH * 0.9))
    lim = (5, 35)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":", zorder=1)
    a = np.linspace(*lim, 100)
    ax.plot(a, alpha_hat(a, r), color=ORANGE, lw=1.2, ls="--", zorder=2)
    ax.scatter(d.truth, d.raw, s=16, color=ORANGE, edgecolor="white", linewidth=0.6, zorder=3,
               label=f"raw image (bias {np.mean(d.raw - d.truth):+.1f}°)".replace("-", "−"))
    ax.scatter(d.truth, d.fmap, s=16, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3,
               label=f"fascicle map (bias {np.mean(d.fmap - d.truth):+.1f}°)".replace("-", "−"))
    ax.annotate(f"model, $r={r:.2f}$", xy=(31, alpha_hat(31, r)), xytext=(26.5, 13.5), fontsize=8,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(29.5, 31.5, "$y=x$", fontsize=8, color=MUTED)
    ax.set(xlim=lim, ylim=(min(lim[0], np.floor(d.raw.min()) - 1), lim[1]), xlabel="Expert PA, mean of 7 raters (°)", ylabel="Structure-tensor PA (°)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "fig4_speckle.pdf")
    for name in ("raw", "fmap"):
        e = d[name] - d.truth
        print(f"fig4 speckle, {name}: n={len(d)}, bias {e.mean():.2f}, MAE {e.abs().mean():.2f}, "
              f"underestimated on {(e < 0).sum()}, corr(bias, expert PA) {np.corrcoef(e, d.truth)[0, 1]:.2f}")
    print(f"fig4 speckle: model r={r:.3f}, model MAE {np.abs(alpha_hat(d.truth.values, r) - d.raw.values).mean():.2f}")


def test_errors():
    import manual_labels as M
    meta = M.test_boxes()
    lab = M.label_table(M.from_annotator_dir(ROOT / "annotations"), meta)
    lab["pa"] += 0.75
    lab["fl"] *= 1.04
    lab["mt"] *= 0.97
    sub = pd.read_csv(ROOT / "work/v9/sub_b_mix_segments.csv")
    m = lab.merge(sub, on="image_id")
    for t in M.TOL:
        m[f"e_{t}"] = m[M.COLS[t]] - m[t]
    m["score"] = sum(m[f"e_{t}"].abs() / M.TOL[t] for t in M.TOL) / 3
    return m


def fig_sensitivity(m):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(0.6 * TEXTWIDTH, 0.6 * TEXTWIDTH * 0.9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
    a = np.linspace(6, 40, 300)
    ar = np.radians(a)
    for mt, shade in ((25, "#174f91"), (20, BLUE), (15, "#8ab8ee")):
        ax.plot(a, np.pi / 360 * (mt / np.sin(ar)) / np.tan(ar), color=shade, label=f"MT {mt} mm")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(39.5, 1.15, "FL term = PA term", fontsize=8, color=MUTED, ha="right")
    ax.legend(loc="upper right")
    ax.set(ylim=(0, 10), ylabel=r"$\rho(\alpha)$, FL cost / PA cost")
    ax2.hist(m.pa_deg, bins=np.arange(6, 41, 1), color=BLUE, edgecolor="white", linewidth=0.6)
    ax2.set(xlabel="PA (°); histogram: our predictions on the test set", ylabel="Test frames", xlim=(6, 40))
    ax2.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig2_sensitivity.pdf")
    print(f"fig2: share of test frames with predicted PA < 20°: {(m.pa_deg < 20).mean():.2f}")


def fig_errors(m):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
    kw = dict(s=9, color=BLUE, alpha=0.55, edgecolor="none")
    ax.scatter(m.pa, m.score, **kw)
    bins = np.arange(6, 40, 3)
    c = pd.cut(m.pa, bins)
    g = m.groupby(c, observed=True).score.median()
    mids = np.array([iv.mid for iv in g.index])
    n = m.groupby(c, observed=True).size().values
    ok = n >= 5
    ax.plot(mids[ok], g.values[ok], color=ORANGE, marker="o", ms=4, mec="white", mew=0.6,
            label="median per 3° bin")
    ax.set(xlabel="Labelled PA (°)", ylabel="Per-frame score", ylim=(0, 1.6))
    ax.legend(loc="upper right")
    ax.set_title("(a)", loc="left")

    x = m.e_pa.abs()
    b, a0 = np.polyfit(x, m.score, 1)
    ax2.scatter(x, m.score, **kw)
    xs = np.linspace(0, x.max(), 50)
    ax2.plot(xs, a0 + b * xs, color=ORANGE, label=f"$S_i = {a0:.2f} + {b:.2f}\\,|e_{{\\mathrm{{PA}}}}|$")
    ax2.set(xlabel="|PA error| (°)", ylabel="Per-frame score", ylim=(0, 1.6))
    ax2.legend(loc="upper left")
    ax2.set_title("(b)", loc="left")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig3_errors.pdf")
    print(f"fig3: n={len(m)}, fit {a0:.3f} + {b:.3f}|e|, corr(e_pa, e_fl) {np.corrcoef(m.e_pa, m.e_fl)[0, 1]:.2f}")


if __name__ == "__main__":
    fig_speckle()
    m = test_errors()
    fig_sensitivity(m)
    fig_errors(m)
