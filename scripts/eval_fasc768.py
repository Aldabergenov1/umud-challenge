"""Compare the 768 px fascicle model (work/exp768/fasc_fold0.pt) with the 512 px one of the same fold.

Aponeuroses come from the cached 5-fold ensemble, so only the fascicle map differs.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from core import load_pipeline

P = load_pipeline()
torch.set_grad_enabled(False)
TOL, WORK, CFG = P["TOL"], P["WORK"], P["CFG"]


def fasc_model(path):
    m = P["make_model"](1, pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(P["DEVICE"]).eval()


MODELS = {512: fasc_model(WORK / "fasc_fold0.pt"), 768: fasc_model(WORK / "exp768/fasc_fold0.pt")}


def fasc_prob(size, roi):
    CFG["img_size"] = size
    return P["predict_probs"]([MODELS[size]], roi)


def evaluate(rois, apo_probs, scales, size):
    rows = []
    for roi, apo, s in zip(rois, apo_probs, scales):
        pr = np.concatenate([apo.astype(np.float32)[:2], fasc_prob(size, roi)])
        res, dbg = P["measure"](roi, pr, s / 10, return_debug=True)
        if dbg is None:
            rows.append(dict(res, deep_share=np.nan))
            continue
        (a1, b1), (a2, b2) = dbg["sup"], dbg["deep"]
        d = [((y - (a1 * x + b1)) / max(1, (a2 - a1) * x + b2 - b1), L) for _, L, x, y, _ in dbg["segs"]]
        total = sum(L for _, L in d) or 1
        rows.append(dict(res, deep_share=sum(L for dd, L in d if dd > 0.5) / total))
    return pd.DataFrame(rows)


bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
truth = pd.DataFrame({t: bench[[f"R{i}_{t.upper()}" for i in range(1, 8)]].mean(axis=1) for t in TOL})
b_rois, b_probs = pickle.load(open(WORK / "bench_probs.pkl", "rb"))
meta = P["test_meta"]()
t_rois, t_probs = pickle.load(open(WORK / "test_probs.pkl", "rb"))

out = {}
for size in (512, 768):
    b = evaluate(b_rois, b_probs, bench.Scale_pixel_per_cm, size)
    fl = b.mt / np.sin(np.radians(b.pa))
    sc = np.mean([(b.pa - truth.pa).abs().mean() / 6, (fl - truth.fl).abs().mean() / 12, (b.mt - truth.mt).abs().mean() / 3])
    t = evaluate([t_rois[i] for i in meta.image_id], [t_probs[i] for i in meta.image_id], meta.px_per_cm, size)
    t["layout"] = meta.layout.values
    out[size] = t
    print(f"{size}: benchmark score {sc:.3f}, PA MAE {(b.pa - truth.pa).abs().mean():.2f} bias {(b.pa - truth.pa).mean():+.2f}, "
          f"lower-half share {b.deep_share.median():.2f} | test lower-half share by layout "
          f"{t.groupby('layout').deep_share.median().round(2).to_dict()}")
d = out[768].pa - out[512].pa
print("test PA change 768 - 512, median by layout:", d.groupby(out[768].layout).median().round(2).to_dict())
out[768].to_csv(WORK / "exp768/test_measure.csv", index=False)
