"""PA at the deep aponeurosis: fit fragment angle vs depth and read it off near the insertion.

Raters measure PA where a fascicle meets the deep aponeurosis, while our fragments often sit in the upper
half of the muscle where curved fascicles are flatter. Compares variants on the benchmark and shows how
much each one moves test PA per layout.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from core import load_pipeline

P = load_pipeline()
TOL = P["TOL"]


def fragment_table(roi, prob, px_per_mm):
    res, dbg = P["measure"](roi, prob.astype(np.float32), px_per_mm, return_debug=True)
    if dbg is None:
        return res, pd.DataFrame(columns=["pa", "w", "depth"])
    (a1, b1), (a2, b2) = dbg["sup"], dbg["deep"]
    rows = [dict(pa=pa, w=length, depth=(y - (a1 * x + b1)) / max(1, (a2 - a1) * x + b2 - b1))
            for pa, length, x, y, t in dbg["segs"]]
    return res, pd.DataFrame(rows)


def pa_at_depth(seg: pd.DataFrame, fallback: float, target=0.85, max_slope=15.0, shrink=1.0):
    """Weighted linear fit PA = a + b*depth, evaluated at `target` depth; slope clipped and shrunk."""
    if len(seg) < 4 or seg.depth.max() - seg.depth.min() < 0.2:
        return fallback
    b, a = np.polyfit(seg.depth, seg.pa, 1, w=np.sqrt(seg.w))
    b = float(np.clip(b, -max_slope, max_slope)) * shrink
    centre = np.average(seg.depth, weights=seg.w)
    level = P["weighted_median"](seg.pa.values, seg.w.values)
    return level + b * (target - centre)


def variants(res, seg):
    out = {"median": res["pa_seg"] if np.isfinite(res["pa_seg"]) else res["pa_tensor"]}
    for t in (0.7, 0.85):
        for s in (0.5, 1.0):
            out[f"depth{t}_s{s}"] = pa_at_depth(seg, out["median"], target=t, shrink=s)
    low = seg[seg.depth > 0.5]
    out["lower_half_median"] = P["weighted_median"](low.pa.values, low.w.values) if len(low) >= 2 else out["median"]
    return out


# benchmark
bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
truth = pd.DataFrame({t: bench[[f"R{i}_{t.upper()}" for i in range(1, 8)]].mean(axis=1) for t in TOL})
rois, probs = pickle.load(open(ROOT / "work/bench_probs.pkl", "rb"))
rows, mts = [], []
for roi, p, s in zip(rois, probs, bench.Scale_pixel_per_cm):
    res, seg = fragment_table(roi, p, s / 10)
    rows.append(variants(res, seg))
    mts.append(res["mt"])
vb = pd.DataFrame(rows)
mt = pd.Series(mts)
print("benchmark (vs 7-rater mean):")
for c in vb:
    fl = mt / np.sin(np.radians(vb[c]))
    s = np.mean([(vb[c] - truth.pa).abs().mean() / 6, (fl - truth.fl).abs().mean() / 12, (mt - truth.mt).abs().mean() / 3])
    print(f"  {c:20s} PA MAE {(vb[c] - truth.pa).abs().mean():.2f} bias {(vb[c] - truth.pa).mean():+.2f} | "
          f"FL MAE {(fl - truth.fl).abs().mean():.2f} bias {(fl - truth.fl).mean():+.2f} | score {s:.3f}")

# test: how much each variant moves PA
meta = P["test_meta"]()
trois, tprobs = pickle.load(open(ROOT / "work/test_probs.pkl", "rb"))
rows = []
for r in meta.itertuples():
    res, seg = fragment_table(trois[r.image_id], tprobs[r.image_id], r.px_per_cm / 10)
    rows.append(dict(layout=r.layout, **variants(res, seg)))
vt = pd.DataFrame(rows)
cols = [c for c in vt if c not in ("layout", "median")]
shift = vt[cols].sub(vt["median"], axis=0)
shift["layout"] = vt.layout
print("\ntest: median change of PA vs the current method, by layout (°):")
print(shift.groupby("layout").median().round(2).to_string())
print("all:", shift[cols].median().round(2).to_dict())
vt.to_csv(ROOT / "work/test_pa_variants.csv", index=False)
