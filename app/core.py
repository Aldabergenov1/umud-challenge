"""Pipeline functions for the viewer, taken from notebooks/umud_pipeline.ipynb.

The notebook stays the single source of truth. From its code cells we execute only imports,
function definitions and a few constants (paths, CFG, DEVICE, TOL); the heavy cells
(training-set cache, training, benchmark, submission) are skipped.
"""
import ast
import json
import os
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/umud_pipeline.ipynb"
CONSTANTS = {"ROOT", "DATA", "BENCH", "WORK", "SMOKE", "CFG", "DEVICE", "TOL", "V9_RUNS"}
NO_CALIBRATION = {"fl_weight": 0.2, "pa": ["none", 0.0], "mt": ["none", 0.0], "fl": ["none", 0.0]}


def load_pipeline() -> dict:
    """Namespace with the notebook's functions: find_roi, decode_scale, predict_probs, measure, ..."""
    os.chdir(ROOT)                      # the notebook derives ROOT from the working directory
    sys.path.insert(0, str(ROOT / "notebooks"))
    code = "\n\n".join(c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code")
    keep = []
    for node in ast.parse(code).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in CONSTANTS for t in node.targets):
            keep.append(node)
    ns = {"__name__": "umud_pipeline"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return ns


def available_folds(work: Path) -> list:
    """Folds that have weights for both networks."""
    folds = []
    for k in range(10):
        if (work / f"apo_fold{k}.pt").exists() and (work / f"fasc_fold{k}.pt").exists():
            folds.append(k)
    return folds


def load_calibration(work: Path) -> dict:
    """The last notebook run's calibration, else the copy committed with the app, else none."""
    for path in (work / "calibration.json", Path(__file__).parent / "calibration.json"):
        if path.exists():
            return json.loads(path.read_text())
    return NO_CALIBRATION
