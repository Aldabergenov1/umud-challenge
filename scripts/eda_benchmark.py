"""UMUD expert benchmark (35 images, 7 raters): agreement in UMUD-score units.

Test labels are the mean of 2 raters, so we also simulate "mean of 2 raters"
targets and score single raters / algorithms / other rater pairs against them.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = {"MT": 3.0, "FL": 12.0, "PA": 6.0}
xlsx = Path(sys.argv[1])
df = pd.read_excel(xlsx, sheet_name="Manual_architecture")
raters = [f"R{i}" for i in range(1, 8)]
methods = raters + ["DLTrack", "SMA"]


def comp(pred: str, ref: pd.DataFrame) -> dict:
    out = {}
    for t in TOL:
        out[t] = (df[f"{pred}_{t}"] - ref[t]).abs().mean()
    out["score"] = np.mean([out[t] / TOL[t] for t in TOL])
    for t in TOL:
        out[f"bias_{t}"] = (df[f"{pred}_{t}"] - ref[t]).mean()
    return out


def mean_of(cols):
    return pd.DataFrame({t: df[[f"{c}_{t}" for c in cols]].mean(axis=1) for t in TOL})


print("Images:", len(df), "| px/cm values:", sorted(df.Scale_pixel_per_cm.unique()))
print("\nTarget distribution (mean of 7 raters):")
print(mean_of(raters).describe().round(2).to_string())

print("\n1) Each method vs mean of the 7 raters (rater itself excluded from its reference):")
rows = []
for m in methods:
    ref = mean_of([r for r in raters if r != m])
    rows.append({"method": m, **comp(m, ref)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n2) Expected score vs a 2-rater-mean target (all pairs of the remaining raters):")
rows = []
for m in methods:
    others = [r for r in raters if r != m]
    sc = [comp(m, mean_of(p))["score"] for p in itertools.combinations(others, 2)]
    rows.append({"method": m, "mean": np.mean(sc), "min": np.min(sc), "max": np.max(sc)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n3) Noise floor: mean of 2 raters vs mean of 2 *other* raters, and mean-of-5 vs pair:")
s22, s52 = [], []
for p in itertools.combinations(raters, 2):
    rest = [r for r in raters if r not in p]
    tgt = mean_of(p)
    pred5 = mean_of(rest)
    s52.append(np.mean([(pred5[t] - tgt[t]).abs().mean() / TOL[t] for t in TOL]))
    for q in itertools.combinations(rest, 2):
        pq = mean_of(q)
        s22.append(np.mean([(pq[t] - tgt[t]).abs().mean() / TOL[t] for t in TOL]))
print(f"pair vs other pair: mean {np.mean(s22):.3f} (min {np.min(s22):.3f}, max {np.max(s22):.3f})")
print(f"mean of other 5 vs pair (oracle 'consensus' predictor): mean {np.mean(s52):.3f} (min {np.min(s52):.3f})")

print("\n4) FL vs MT/sin(PA) inside the rater consensus:")
c = mean_of(raters)
geo = c.MT / np.sin(np.radians(c.PA))
print(pd.DataFrame({"FL": c.FL, "MT/sinPA": geo, "ratio": c.FL / geo}).describe().round(3).to_string())
print("MAE(FL, MT/sinPA) =", round((c.FL - geo).abs().mean(), 2), "mm")

print("\n5) Per-image rater spread (std across 7 raters), median:")
for t in TOL:
    print(t, round(df[[f"{r}_{t}" for r in raters]].std(axis=1).median(), 2))
