"""Depth-gain normalisation before the networks (inference only).

Ultrasound echoes fade with depth and every device compensates differently, so the lower half of the
muscle is often darker than anything in the training set. Here each row is rescaled to a common mean
brightness before segmentation. Compares fragment depth coverage and PA with and without it.
Usage: python scripts/depth_gain.py [n_test_frames]
"""
import io
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import available_folds, load_pipeline

P = load_pipeline()
torch.set_grad_enabled(False)
TOL, WORK = P["TOL"], P["WORK"]


def depth_gain(roi: np.ndarray, smooth_frac=0.08, max_gain=3.0) -> np.ndarray:
    f = roi.astype(np.float32)
    w = f.shape[1]
    rows = f[:, w // 10: -w // 10].mean(axis=1)
    rows = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=max(1.0, smooth_frac * len(rows))).ravel()
    gain = np.clip(np.median(rows) / np.maximum(rows, 1.0), 1 / max_gain, max_gain)
    return np.clip(f * gain[:, None], 0, 255).astype(np.uint8)


models = {}
for task, n in (("apo", 2), ("fasc", 1)):
    models[task] = []
    for k in available_folds(WORK):
        m = P["make_model"](n, pretrained=False)
        m.load_state_dict(torch.load(WORK / f"{task}_fold{k}.pt", map_location="cpu"))
        models[task].append(m.to(P["DEVICE"]).eval())


def probs(roi):
    return np.concatenate([P["predict_probs"](models["apo"], roi), P["predict_probs"](models["fasc"], roi)])


def summary(roi, pr, px_mm):
    res, dbg = P["measure"](roi, pr, px_mm, return_debug=True)
    if dbg is None:
        return dict(res, deep_share=np.nan)
    (a1, b1), (a2, b2) = dbg["sup"], dbg["deep"]
    d = [((y - (a1 * x + b1)) / max(1, (a2 - a1) * x + b2 - b1), L) for _, L, x, y, _ in dbg["segs"]]
    total = sum(L for _, L in d) or 1
    return dict(res, deep_share=sum(L for dd, L in d if dd > 0.5) / total)


bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
truth = pd.DataFrame({t: bench[[f"R{i}_{t.upper()}" for i in range(1, 8)]].mean(axis=1) for t in TOL})
b_rois, b_probs = pickle.load(open(WORK / "bench_probs.pkl", "rb"))
base, gain = [], []
for roi, pr, s in zip(b_rois, b_probs, bench.Scale_pixel_per_cm):
    base.append(summary(roi, pr.astype(np.float32), s / 10))
    gain.append(summary(roi, probs(depth_gain(roi)), s / 10))
for name, rows in (("baseline", base), ("depth gain", gain)):
    d = pd.DataFrame(rows)
    fl = d.mt / np.sin(np.radians(d.pa))
    sc = np.mean([(d.pa - truth.pa).abs().mean() / 6, (fl - truth.fl).abs().mean() / 12, (d.mt - truth.mt).abs().mean() / 3])
    print(f"benchmark {name:10s}: score {sc:.3f} | PA MAE {(d.pa - truth.pa).abs().mean():.2f} bias {(d.pa - truth.pa).mean():+.2f}"
          f" | MT MAE {(d.mt - truth.mt).abs().mean():.2f} | share of fragment length in lower half {d.deep_share.median():.2f}")

meta = P["test_meta"]()
t_rois, t_probs = pickle.load(open(WORK / "test_probs.pkl", "rb"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
sel = pd.concat([g.iloc[:: max(1, len(g) // max(1, n // 6))] for _, g in meta.groupby("layout")]).head(n)
rows = []
for r in sel.itertuples():
    roi = t_rois[r.image_id]
    b0 = summary(roi, t_probs[r.image_id].astype(np.float32), r.px_per_cm / 10)
    g1 = summary(roi, probs(depth_gain(roi)), r.px_per_cm / 10)
    rows.append(dict(layout=r.layout, pa0=b0["pa"], pa1=g1["pa"], mt0=b0["mt"], mt1=g1["mt"],
                     deep0=b0["deep_share"], deep1=g1["deep_share"]))
t = pd.DataFrame(rows)
t["d_pa"], t["d_mt"] = t.pa1 - t.pa0, t.mt1 - t.mt0
print(f"\ntest frames: {len(t)}")
print(t.groupby("layout")[["d_pa", "d_mt", "deep0", "deep1"]].median().round(2).to_string())
t.to_csv(WORK / "test_depth_gain.csv", index=False)
