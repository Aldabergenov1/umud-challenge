"""Fold disagreement as label-free uncertainty, and uncertainty-weighted FL shrinkage.

Needs work/per_fold_{test,bench}.csv (scripts/per_fold_measure.py) and the 5-fold maps in work/*_probs.pkl.
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
bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
raters = [f"R{i}" for i in range(1, 8)]
truth = pd.DataFrame({t: bench[[f"{r}_{t.upper()}" for r in raters]].mean(axis=1) for t in TOL})

pf_t = pd.read_csv(ROOT / "work/per_fold_test.csv")
pf_b = pd.read_csv(ROOT / "work/per_fold_bench.csv")
for d in (pf_t, pf_b):
    d["fl"] = d.mt / np.sin(np.radians(d.pa))

rois, probs = pickle.load(open(ROOT / "work/bench_probs.pkl", "rb"))
ens_b = pd.DataFrame([P["measure"](r, p.astype(np.float32), s / 10) for r, p, s in
                      zip(rois, probs, bench.Scale_pixel_per_cm)])
ens_b["fl"] = ens_b.mt / np.sin(np.radians(ens_b.pa))
ens_t = pd.read_csv(ROOT / "work/test_raw_5fold.csv")
ens_t["fl"] = ens_t.mt / np.sin(np.radians(ens_t.pa))

sd_b = pf_b.groupby("image", sort=False)[["pa", "mt", "fl"]].std().reset_index(drop=True)
sd_t = pf_t.groupby("image_id", sort=False)[["pa", "mt", "fl"]].std().reset_index(drop=True)
sd_t["layout"] = pf_t.groupby("image_id", sort=False).layout.first().values

print("fold std, median per image (PA °, MT mm, FL mm):")
print("  benchmark:", sd_b.median().round(2).to_dict())
print(sd_t.groupby("layout")[["pa", "mt", "fl"]].median().round(2).to_string())

# does fold spread predict the error of the ensemble on the benchmark?
err = (ens_b.fl - truth.fl).abs()
print(f"\nbenchmark: corr(|FL error|, fold std FL) = {np.corrcoef(err, sd_b.fl)[0, 1]:.2f}; "
      f"median |err| / std = {np.median(err / sd_b.fl):.1f}")

# uncertainty-weighted shrinkage of FL towards a + b*MT (prior fitted leave-one-out on the benchmark)
def prior_fl(mt_train, fl_train, mt):
    return np.polyval(np.polyfit(mt_train, fl_train, 1), mt)


def shrink(fl, prior, sd, k, s_prior):
    w = (k * sd) ** 2 / ((k * sd) ** 2 + s_prior ** 2)
    return (1 - w) * fl + w * prior, w


n = len(truth)
prior_b = pd.Series([prior_fl(ens_b.mt[np.arange(n) != i], truth.fl[np.arange(n) != i], ens_b.mt[i]) for i in range(n)])
s_prior = (prior_b - truth.fl).abs().mean() * 1.25          # MAE -> sigma for a normal error
print(f"prior a+b*MT: MAE {(prior_b - truth.fl).abs().mean():.1f} mm, sigma ~{s_prior:.1f}")
for k in [0, 1, 2, 3, 5, 8]:
    fl, w = shrink(ens_b.fl, prior_b, sd_b.fl, k, s_prior)
    print(f"  k={k}: FL MAE {(fl - truth.fl).abs().mean():.2f}, mean prior weight {w.mean():.2f}")

prior_t = prior_fl(ens_b.mt, truth.fl, ens_t.mt)
for k in [2, 3, 5]:
    _, w = shrink(ens_t.fl, prior_t, sd_t.fl, k, s_prior)
    print(f"test, k={k}: mean prior weight {w.mean():.2f}; by layout",
          pd.Series(w).groupby(sd_t.layout).mean().round(2).to_dict())
