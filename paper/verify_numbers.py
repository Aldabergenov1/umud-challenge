"""Numbers quoted in the paper that are not produced by figures/make_figures.py. CPU only.

Uses the network maps of the final submission (work/v9/b_mix/*_probs.pkl) and the manual labels.

    python paper/verify_numbers.py
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
from core import load_pipeline
import manual_labels as M

P = load_pipeline()
CFG, TOL = P["CFG"], P["TOL"]
MAPS = ROOT / "work/v9/b_mix"
SUBS = {"v5": "sub_v5_learned.csv", "v7": "sub_v7_tta.csv", "v9a": "v9/sub_a_cnx_segments.csv",
        "v9 two nets": "v9/sub_a_mix_segments.csv", "v9 three nets": "v9/sub_b_mix_segments.csv"}
PUBLIC_LB = {"v5": 0.385, "v7": 0.395, "v9a": 0.391, "v9 two nets": 0.394, "v9 three nets": 0.383}


def score(pred, true):
    return np.mean([np.nanmean(np.abs(pred[t].values - true[t].values)) / TOL[t] for t in TOL])


# --- A. Final pipeline on the benchmark ------------------------------------------------------------
print("A. Benchmark, final maps")
bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
raters = [f"R{i}" for i in range(1, 8)]
rater = {r: pd.DataFrame({t: bench[f"{r}_{t.upper()}"] for t in TOL}) for r in raters}
consensus = pd.DataFrame({t: bench[[f"{r}_{t.upper()}" for r in raters]].mean(axis=1) for t in TOL})
rois, probs = pickle.load(open(MAPS / "bench_probs.pkl", "rb"))
raw = pd.DataFrame([P["measure"](r, p.astype(np.float32), s / 10) for r, p, s in zip(rois, probs, bench.Scale_pixel_per_cm)])
def loo_calibrated(pa_offset):
    CFG["pa_offset"] = pa_offset
    out = []
    for i in range(len(raw)):
        keep = np.arange(len(raw)) != i
        calib, _ = P["fit_calibration"](raw[keep].reset_index(drop=True), consensus[keep].reset_index(drop=True))
        out.append(P["apply_calibration"](raw.iloc[[i]], calib).iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


pairs = list(itertools.combinations(raters, 2))
pair_ref = [pd.DataFrame({t: (rater[a][t] + rater[b][t]) / 2 for t in TOL}) for a, b in pairs]
for name, pred in (("raw, FL = MT/sin PA", raw.assign(fl=raw.fl_trig)),
                   ("calibrated, leave-one-out", loo_calibrated(0.0)),
                   ("calibrated LOO + test PA offset 1.68", loo_calibrated(1.68))):
    print(f"  {name:38s} vs 7-rater mean {score(pred, consensus):.3f}   vs rater pairs (mean of 21) "
          f"{np.mean([score(pred, ref) for ref in pair_ref]):.3f}   MAE PA {np.abs(pred.pa - consensus.pa).mean():.2f}"
          f"  FL {np.abs(pred.fl - consensus.fl).mean():.2f}  MT {np.abs(pred.mt - consensus.mt).mean():.2f}")

# --- Test labels and features -----------------------------------------------------------------------
meta = P["test_meta"]()
lab = M.label_table(M.from_annotator_dir(ROOT / "annotations"), M.test_boxes())
lab["pa"] += 0.75
lab["fl"] *= 1.04
lab["mt"] *= 0.97
rois_t, probs_t = pickle.load(open(MAPS / "test_probs.pkl", "rb"))
feats = P["frame_features"](rois_t, probs_t, meta)
d = lab.merge(feats, on="image_id").merge(pd.read_csv(ROOT / "work" / SUBS["v9 three nets"]), on="image_id")

# --- B. FL as a centred straight fascicle between our lines ----------------------------------------
print("\nB. FL as a centred straight fascicle between our aponeurosis lines")
k = CFG["px_per_mm"]
fl_ours, fl_lab = [], []
for r in meta.itertuples():
    row = d[d.image_id == r.image_id]
    if row.empty:
        continue
    row = row.iloc[0]
    res, dbg = P["measure"](rois_t[r.image_id], probs_t[r.image_id].astype(np.float32), r.px_per_cm / 10,
                            CFG["h_over_v"].get(r.layout, 1.0), return_debug=True)
    if dbg is None:
        continue
    t_deep = np.arctan(dbg["deep"][0])
    side = 1.0 if ((dbg["t_band"] - t_deep + np.pi / 2) % np.pi - np.pi / 2) > 0 else -1.0
    W = dbg["gm"].shape[1]
    ray = lambda pa: P["fascicle_ray_length"](dbg["sup"], dbg["deep"], t_deep + side * np.radians(pa), W) / k
    fl_ours.append((row.fl, ray(row.pa_deg)))
    fl_lab.append((row.fl, ray(row.pa)))
for name, v in (("our final PA", fl_ours), ("labelled PA", fl_lab)):
    v = np.array(v, float)
    ok = np.isfinite(v).all(axis=1)
    y, x = v[ok, 0], v[ok, 1]
    s = np.median(y / x)
    print(f"  {name:13s}: n {ok.sum()}, global scale {s:.3f}, FL MAE {np.abs(s * x - y).mean():.2f} mm "
          f"(without scale {np.abs(x - y).mean():.2f})")

# --- C. Which percentile of fragment angles matches the labels --------------------------------------
print("\nC. Percentiles of fragment angles (raw geometry) against labelled PA")
for q in ("pa_q50", "pa_q75", "pa_q90"):
    e = (d.pa - d[q]).dropna()
    off = e.median()
    print(f"  {q}: best offset {off:+.2f}°, MAE after offset {np.abs(e - off).mean():.2f}°, n {len(e)}")

print("\nC2. Other PA estimates against labelled PA, MAE after the best constant offset")
for c in ("raw_pa_seg", "raw_pa_orient", "pa_radon", "pa_gabor", "raw_pa_tensor"):
    e = (d.pa - d[c]).dropna()
    print(f"  {c}: n {len(e)}, offset {e.median():+.2f}°, MAE {np.abs(e - e.median()).mean():.2f}°")

# --- E. Random public thirds ------------------------------------------------------------------------
print("\nE. Random public thirds (102 of 309 frames), scored against the labels")
subs = {n: lab[["image_id"]].merge(pd.read_csv(ROOT / "work" / f), on="image_id") for n, f in SUBS.items()}
err = {n: sum(np.abs(s[M.COLS[t]].values - lab[t].values) / TOL[t] for t in TOL) / 3 for n, s in subs.items()}
rng = np.random.default_rng(0)
idx = np.array([rng.choice(len(lab), 102, replace=False) for _ in range(20000)])
for a, b in (("v5", "v7"), ("v5", "v9 three nets"), ("v9a", "v9 three nets"), ("v9 two nets", "v9 three nets")):
    diff = err[b][idx].mean(axis=1) - err[a][idx].mean(axis=1)
    lb = PUBLIC_LB[b] - PUBLIC_LB[a]
    print(f"  {b} - {a}: all frames {err[b].mean() - err[a].mean():+.4f}, sd over thirds {diff.std():.4f}, "
          f"LB {lb:+.3f}, P(third >= LB diff) {(diff >= lb - 1e-9).mean():.4f}")
print("  local vs LB, five later submissions: Pearson",
      round(np.corrcoef([err[n].mean() for n in SUBS], [PUBLIC_LB[n] for n in SUBS])[0, 1], 2))
