"""Shared helper: code of the pipeline notebook as one Python string."""
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/umud_pipeline.ipynb"


def notebook_code(path=NOTEBOOK) -> str:
    nb = nbformat.read(path, as_version=4)
    return "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
