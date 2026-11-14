"""Does a lower probability threshold for fascicle fragments recover fibres near the deep aponeurosis?"""
import io
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from core import load_pipeline

P = load_pipeline()
TOL, WORK = P["TOL"], P["WORK"]
original = P["fascicle_segments"]


def with_threshold(thr):
    def segs(p_fasc, min_len_mm=3.0):
        return original(np.where(p_fasc > thr, 1.0, 0.0).astype(np.float32), min_len_mm)
    return segs


def run(rois, probs, scales):
    rows = []
    for roi, pr, s in zip(rois, probs, scales):
        res, dbg = P["measure"](roi, pr.astype(np.float32), s / 10, return_debug=True)
        (a1, b1), (a2, b2) = dbg["sup"], dbg["deep"]
        d = [((y - (a1 * x + b1)) / max(1, (a2 - a1) * x + b2 - b1), L) for _, L, x, y, _ in dbg["segs"]]
        total = sum(L for _, L in d) or 1
        rows.append(dict(pa=res["pa"], mt=res["mt"], n=len(d), deep_share=sum(L for dd, L in d if dd > 0.5) / total))
    return pd.DataFrame(rows)


bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
truth = pd.DataFrame({t: bench[[f"R{i}_{t.upper()}" for i in range(1, 8)]].mean(axis=1) for t in TOL})
b_rois, b_probs = pickle.load(open(WORK / "bench_probs.pkl", "rb"))
meta = P["test_meta"]()
t_rois, t_probs = pickle.load(open(WORK / "test_probs.pkl", "rb"))

base_t = None
for thr in (0.5, 0.4, 0.3):
    P["fascicle_segments"] = with_threshold(thr)
    b = run(b_rois, b_probs, bench.Scale_pixel_per_cm)
    fl = b.mt / np.sin(np.radians(b.pa))
    sc = np.mean([(b.pa - truth.pa).abs().mean() / 6, (fl - truth.fl).abs().mean() / 12, (b.mt - truth.mt).abs().mean() / 3])
    t = run([t_rois[i] for i in meta.image_id], [t_probs[i] for i in meta.image_id], meta.px_per_cm)
    t["layout"] = meta.layout.values
    if base_t is None:
        base_t = t
    print(f"thr {thr}: benchmark score {sc:.3f}, PA bias {(b.pa - truth.pa).mean():+.2f}, fragments {b.n.median():.0f} | "
          f"test: fragments {t.n.median():.0f}, lower-half share {t.deep_share.median():.2f}, "
          f"PA change vs 0.5 by layout {(t.pa - base_t.pa).groupby(t.layout).median().round(2).to_dict()}")
