"""Benchmark check of FL / PA estimator ideas on cached 5-fold maps (leave-one-out where something is fit)."""
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
rois, probs = pickle.load(open(ROOT / "work/bench_probs.pkl", "rb"))
raw = pd.DataFrame([P["measure"](r, p.astype(np.float32), s / 10) for r, p, s in zip(rois, probs, bench.Scale_pixel_per_cm)])


def mae(x, t):
    return (x - truth[t]).abs().mean()


def loo(fit, predict, n=len(truth)):
    """Leave-one-out predictions from fit(train_idx) -> params, predict(params, i) -> value."""
    return pd.Series([predict(fit(np.arange(n) != i), i) for i in range(n)])


print("experts' FL vs MT:  corr", round(np.corrcoef(truth.mt, truth.fl)[0, 1], 2),
      "| FL vs MT/sinPA (experts' own PA): corr",
      round(np.corrcoef(truth.mt / np.sin(np.radians(truth.pa)), truth.fl)[0, 1], 2))

# FL from MT alone: FL = a + b*MT
fl_mt = loo(lambda m: np.polyfit(raw.mt[m], truth.fl[m], 1), lambda c, i: np.polyval(c, raw.mt[i]))
fl_const = loo(lambda m: truth.fl[m].median(), lambda c, i: c)
fl_trig = raw.mt / np.sin(np.radians(raw.pa))
print(f"\nFL MAE: constant {mae(fl_const, 'fl'):.1f} | a+b*MT {mae(fl_mt, 'fl'):.1f} | "
      f"MT/sin(PA) {mae(fl_trig, 'fl'):.1f} | fl_line {mae(raw.fl_line, 'fl'):.1f}")

for w in [0.0, 0.2, 0.4, 0.6]:
    print(f"  blend MT/sinPA with a+b*MT, weight of prior {w:.1f}: FL MAE {mae((1 - w) * fl_trig + w * fl_mt, 'fl'):.2f}")

# PA variants
pa_mix = (raw.pa_seg + raw.pa_tensor) / 2
print(f"\nPA MAE: segments {mae(raw.pa_seg, 'pa'):.2f} | tensor {mae(raw.pa_tensor, 'pa'):.2f} | mean of both {mae(pa_mix, 'pa'):.2f}")
fl_mix = raw.mt / np.sin(np.radians(pa_mix))
print(f"FL MAE with PA = mean of both: {mae(fl_mix, 'fl'):.2f}")

# the sensitivity: d FL / d PA at each image, mm per degree
sens = fl_trig / np.tan(np.radians(raw.pa)) * np.pi / 180
print(f"\nFL sensitivity to PA on the benchmark: median {sens.median():.1f} mm per degree (PA median {raw.pa.median():.1f}°)")
