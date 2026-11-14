"""Manual labels -> table, and local scores of submissions against them.

Usage: python scripts/manual_labels.py [submission.csv ...]
  Without arguments scores every work/sub_*.csv and work/probes/*.csv.
Reads labels/<annotator>.json (written by the annotation page of the app) and annotations/*.json (one file
per frame, written by umud-annotator/, counted as annotator "arman"). Each label is spread over its series.
Values are recomputed from the clicked points with our ruler scale, so a wrong depth typed in umud-annotator
does not matter. Writes labels/manual_<annotator>.csv and labels/manual_mean.csv (mean over annotators).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from umud_manual import measure_manual

IMAGE_DIR = ROOT / "data/raw/test_images_v2/test_set_v2"
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
COLS = {"pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"}


def test_boxes() -> pd.DataFrame:
    """Scale and ultrasound box per test frame, from the app's cache or the notebook."""
    sys.path.insert(0, str(ROOT / "app"))
    from core import load_pipeline
    return load_pipeline()["test_meta"]().set_index("image_id")


def from_annotator_dir(folder: Path) -> dict:
    """umud-annotator files -> the labels format of the app (sup, deep, fasc in full-frame pixels)."""
    out = {}
    for f in sorted(folder.glob("IMG_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d.get("annotation_complete"):
            continue
        pts = {k: [[p["x_px"], p["y_px"]] for p in v] for k, v in d["points"].items()}
        fasc = pts.get("fascicle", [])
        image_id = next((p.name for p in IMAGE_DIR.glob(d["image_id"] + ".*")), d["image_id"] + ".tif")
        out[image_id] = dict(sup=pts.get("upper_aponeurosis", []), deep=pts.get("lower_aponeurosis", []),
                             fasc=[fasc[i:i + 2] for i in range(0, len(fasc) - 1, 2)], series=[image_id],
                             their=dict(pa=d.get("pa_deg"), fl=d.get("fl_mm"), mt=d.get("mt_mm"),
                                        depth_cm=d.get("depth_cm"), no_depth=bool(d.get("no_depth"))))
    return out


def label_table(source, meta: pd.DataFrame) -> pd.DataFrame:
    labels = source if isinstance(source, dict) else json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for image_id, ann in labels.items():
        if ann.get("skip"):
            continue
        r = meta.loc[image_id]
        m = measure_manual(ann, r.px_per_cm, (r.l, r.t, r.r, r.b))
        if not np.isfinite(m["pa"]):
            continue
        for frame in ann.get("series", [image_id]):
            rows.append(dict(image_id=frame, labelled=image_id, layout=meta.loc[frame, "layout"],
                             pa=m["pa"], fl=m["fl"], mt=m["mt"], n_fasc=m["n_fasc"]))
    return pd.DataFrame(rows)


def score(sub: pd.DataFrame, truth: pd.DataFrame) -> dict:
    m = truth.merge(sub, on="image_id")
    mae = {t: (m[COLS[t]] - m[t]).abs().mean() for t in TOL}
    return dict(n=len(m), score=np.mean([mae[t] / TOL[t] for t in TOL]), **{f"mae_{t}": v for t, v in mae.items()})


if __name__ == "__main__":
    sources = {f.stem: f for f in sorted((ROOT / "labels").glob("*.json"))}
    if (ROOT / "annotations").exists():
        sources["arman"] = from_annotator_dir(ROOT / "annotations")
    if not sources:
        sys.exit("Нет разметки ни в labels/, ни в annotations/.")
    meta = test_boxes()
    (ROOT / "labels").mkdir(exist_ok=True)
    tables = []
    for name, src in sources.items():
        t = label_table(src, meta).assign(annotator=name)
        t.to_csv(ROOT / f"labels/manual_{name}.csv", index=False)
        print(f"{name}: размечено {t.labelled.nunique()} кадров, с сериями {len(t)}")
        tables.append(t)
    allt = pd.concat(tables)
    if allt.annotator.nunique() > 1:
        both = allt.groupby("image_id").filter(lambda g: g.annotator.nunique() > 1)
        pair = both.pivot_table(index="image_id", columns="annotator", values=list(TOL))
        a, b = pair.columns.levels[1][:2]
        print("согласие двух разметчиков:", {t: round((pair[(t, a)] - pair[(t, b)]).abs().mean(), 2) for t in TOL})
    allt = allt.groupby(["annotator", "image_id", "layout"], as_index=False)[list(TOL)].mean()
    truth = allt.groupby(["image_id", "layout"], as_index=False)[list(TOL)].mean()
    truth.to_csv(ROOT / "labels/manual_mean.csv", index=False)

    subs = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "work").glob("sub_*.csv")) + \
        sorted((ROOT / "work/probes").glob("*.csv"))
    rows = [dict(file=p.name, **score(pd.read_csv(p), truth)) for p in subs]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    best = pd.read_csv(subs[0])
    m = truth.merge(best, on="image_id")
    m["err"] = sum((m[COLS[t]] - m[t]).abs() / TOL[t] for t in TOL) / 3
    print(f"\n{subs[0].name} по раскладкам:")
    print(m.groupby("layout").agg(n=("err", "size"), score=("err", "mean"),
                                  bias_pa=("pa", lambda s: (m.loc[s.index, "pa_deg"] - s).mean()),
                                  bias_fl=("fl", lambda s: (m.loc[s.index, "fl_mm"] - s).mean()),
                                  bias_mt=("mt", lambda s: (m.loc[s.index, "mt_mm"] - s).mean())).round(2))
