"""Run the whole notebook source on cached probability maps with CFG overrides; write a submission.

Usage: python scripts/make_variant.py <out.csv> [key=json_value ...]
  e.g. pa_method='"tensor"'  fl_scale=0.95  h_over_v='{"telemed_left_ruler": 0.905}'
Needs work/bench_probs.pkl and work/test_probs.pkl, which a full notebook run writes.
UMUD_TEST_PROBS=<path> and UMUD_BENCH_PROBS=<path> take the maps from other files (e.g. work/v9/ens/).
"""
import io
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: plt.close("all")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT / "notebooks")

from _notebook import notebook_code

out_csv, *pairs = sys.argv[1:]
overrides = {k: json.loads(v) for k, v in (p.split("=", 1) for p in pairs)}
src = notebook_code()


def patch(text, old, new):
    assert old in text, old
    return text.replace(old, new)


src = patch(src, 'DEVICE = "cuda"', f"CFG.update({overrides!r})\nDEVICE = \"cuda\"")
src = patch(src, 'MODELS = {task: load_models(task) for task in ["apo", "fasc"]}', "MODELS = {}")
src = patch(src, "V9 = load_v9()", "V9 = {}")
src = patch(src, "    bench_probs.append(all_probs(bench_rois[-1]))",
            "    bench_probs.append(CACHED_BENCH[len(bench_probs)].astype(np.float32))")
src = patch(src, "test_probs = {k: all_probs(v) for k, v in test_rois.items()}",
            "test_probs = {k: CACHED_TEST[k].astype(np.float32) for k in test_rois}")
src = patch(src, 'submission.to_csv(WORK / "submission.csv", index=False)', f"submission.to_csv({out_csv!r}, index=False)")
src = patch(src, 'json.dump(calib, open(WORK / "calibration.json", "w"), indent=1)', "")
# the cached maps are inputs here: never overwrite them, also not when UMUD_TEST_PROBS points elsewhere
src = patch(src, 'pickle.dump((test_rois, {k: v.astype(np.float16) for k, v in test_probs.items()}), '
                 'open(WORK / "test_probs.pkl", "wb"))', "")
src = patch(src, 'pickle.dump((bench_rois, [p.astype(np.float16) for p in bench_probs]), '
                 'open(WORK / "bench_probs.pkl", "wb"))', "")
src = patch(src, 'submission.to_csv(WORK / "submission_base.csv", index=False)',
            f"submission.to_csv({out_csv.replace('.csv', '_base.csv')!r}, index=False)")

g = {"CACHED_BENCH": pickle.load(open(ROOT / os.environ.get("UMUD_BENCH_PROBS", "work/bench_probs.pkl"), "rb"))[1],
     "CACHED_TEST": pickle.load(open(ROOT / os.environ.get("UMUD_TEST_PROBS", "work/test_probs.pkl"), "rb"))[1]}
exec(compile(src, "notebook", "exec"), g)
