"""Measurements of every single fold on test and benchmark frames -> work/per_fold_{test,bench}.csv.

Lets us see where the folds disagree (a label-free uncertainty on the test domain) and simulate
ensembles. Runs on CPU in ~30 min.
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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import available_folds, load_pipeline

P = load_pipeline()
torch.set_grad_enabled(False)
WORK = P["WORK"]


def fold_models(k):
    out = {}
    for task, n in (("apo", 2), ("fasc", 1)):
        m = P["make_model"](n, pretrained=False)
        m.load_state_dict(torch.load(WORK / f"{task}_fold{k}.pt", map_location="cpu"))
        out[task] = [m.to(P["DEVICE"]).eval()]
    return out


test_rois, _ = pickle.load(open(WORK / "test_probs.pkl", "rb"))
bench_rois, _ = pickle.load(open(WORK / "bench_probs.pkl", "rb"))
meta = P["test_meta"]()
bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")

rows_t, rows_b = [], []
for k in available_folds(WORK):
    M = fold_models(k)
    probs = lambda roi: np.concatenate([P["predict_probs"](M["apo"], roi), P["predict_probs"](M["fasc"], roi)])
    for r in meta.itertuples():
        roi = test_rois[r.image_id]
        rows_t.append(dict(fold=k, image_id=r.image_id, layout=r.layout,
                           **P["measure"](roi, probs(roi), r.px_per_cm / 10)))
    for i, r in enumerate(bench.itertuples()):
        rows_b.append(dict(fold=k, image=r.ImageID, **P["measure"](bench_rois[i], probs(bench_rois[i]),
                                                                   r.Scale_pixel_per_cm / 10)))
    print(f"fold {k} done", flush=True)
    pd.DataFrame(rows_t).to_csv(WORK / "per_fold_test.csv", index=False)
    pd.DataFrame(rows_b).to_csv(WORK / "per_fold_bench.csv", index=False)
print("ALL DONE")
