"""Bias of the pipeline (5-fold, no calibration) against each benchmark rater: which shifts are plausible."""
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
rois, probs = pickle.load(open(ROOT / "work/bench_probs.pkl", "rb"))
ens = pd.DataFrame([P["measure"](r, p.astype(np.float32), s / 10) for r, p, s in zip(rois, probs, bench.Scale_pixel_per_cm)])
ens["fl"] = ens.mt / np.sin(np.radians(ens.pa))
rows = []
for r in [f"R{i}" for i in range(1, 8)] + ["mean"]:
    ref = (pd.DataFrame({t: bench[[f"R{i}_{t.upper()}" for i in range(1, 8)]].mean(axis=1) for t in TOL}) if r == "mean"
           else pd.DataFrame({t: bench[f"{r}_{t.upper()}"] for t in TOL}))
    rows.append(dict(rater=r, **{f"{t}_bias": (ref[t] - ens[t]).median() for t in TOL},
                     fl_ratio=(ref.fl / ens.fl).median(),
                     score=np.mean([(ens[t] - ref[t]).abs().mean() / TOL[t] for t in TOL])))
print("rater value minus ours (median over 35 images):")
print(pd.DataFrame(rows).round(2).to_string(index=False))
