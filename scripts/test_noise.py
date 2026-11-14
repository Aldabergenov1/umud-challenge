"""Label-free noise estimate on the test set.

Frames inside a 5-frame sequence come from one video and are nearly identical, so their true PA/FL/MT are
nearly equal. The spread of our predictions inside a sequence is therefore model noise on test-domain
images. Reported in metric units: mean |x - sequence median| / tolerance.

Importable: `sequence_noise(df, groups)`; run as a script for the current notebook settings.
"""
import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}


def sequence_noise(df: pd.DataFrame, groups) -> pd.Series:
    d = df.copy()
    d["group"] = list(groups)
    size = d.groupby("group").pa.transform("size")
    d = d[size >= 3]
    out = {t: (d[t] - d.groupby("group")[t].transform("median")).abs().mean() for t in TOL}
    out["score"] = np.mean([out[t] / TOL[t] for t in TOL])
    out["n_frames"] = len(d)
    return pd.Series(out)


def load_notebook_namespace():
    """Notebook functions without heavy cells (same trick as app/core.py)."""
    sys.path.insert(0, str(ROOT / "app"))
    from core import load_pipeline
    return load_pipeline()


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    P = load_pipeline_ns = load_notebook_namespace()
    meta = P["test_meta"]()
    rois, probs = pickle.load(open(ROOT / "work/test_probs.pkl", "rb"))
    groups = P["frame_groups"]([rois[i] for i in meta.image_id])
    rows = [P["measure"](rois[r.image_id], probs[r.image_id].astype(np.float32), r.px_per_cm / 10)
            for r in meta.itertuples()]
    raw = pd.DataFrame(rows)
    raw["fl"] = raw.fl_trig
    raw["layout"] = meta.layout.values
    raw.to_csv(ROOT / "work/test_raw_5fold.csv", index=False)
    print("all layouts:", sequence_noise(raw, groups).round(3).to_dict())
    g = pd.Series(groups, index=raw.index)
    for layout, part in raw.groupby("layout"):
        if (part.groupby(g[part.index]).size() >= 3).any():
            print(f"  {layout:20s}", sequence_noise(part, g[part.index]).round(3).to_dict())
