"""How much can the manual labels improve the numbers? Cross-validated by series (GroupKFold).

Targets are the annotator labels corrected to the hosts' convention (analyze_labels.py: PA +0.75,
FL x1.04, MT x0.97). Features are the raw geometry of the current pipeline (work/test_geom_v4.csv,
written by geom_v4.py) plus layout. Models are median regressions, the MAE-optimal choice.
"""
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}

lab = pd.read_csv(ROOT / "work/label_errors_v4.csv")[["image_id", "layout", "series", "pa", "fl", "mt",
                                                      "pa_deg", "fl_mm", "mt_mm"]]
geo = pd.read_csv(ROOT / "work/test_geom_v4.csv").drop(columns=["layout"])
d = lab.merge(geo, on="image_id", suffixes=("", "_raw"))   # labels (already corrected) pa/fl/mt; raw pa_raw, mt_raw
d["pa_cur"], d["fl_cur"], d["mt_cur"] = d.pa_deg, d.fl_mm, d.mt_mm    # current submission v4
d["trig"] = d.mt_raw / np.sin(np.radians(d.pa_raw + 2.0))
d["inv_sin"] = 1 / np.sin(np.radians(d.pa_raw + 2.0))
for c in ["fl_frag", "fl_line", "pa_tensor", "pa_seg"]:
    d[c] = d[c].fillna(d[c].median())
lay = pd.get_dummies(d.layout, prefix="L", dtype=float)
d = pd.concat([d, lay], axis=1)
L = list(lay.columns)


def series_median(df, col):
    return df.groupby("series")[col].transform("median")


def cv(target, feats, per_layout=True, alpha=0.0):
    X = d[feats + (L if per_layout else [])].values
    y = d[target].values
    pred = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, d.series):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs").fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    d["_p"] = pred
    return series_median(d, "_p")


def mae(p, t):
    return np.abs(p - d[t]).mean()


rows = []
def add(t, name, p):
    rows.append(dict(target=t, variant=name, mae=mae(p, t), score_part=mae(p, t) / TOL[t] / 3))


for t in TOL:
    add(t, "текущий v4", d[f"{t}_cur"])
add("pa", "сдвиг по раскладкам", cv("pa", ["pa_cur"]))
add("pa", "pa_seg + pa_tensor по раскладкам", cv("pa", ["pa_seg", "pa_tensor"]))
add("mt", "сдвиг по раскладкам", cv("mt", ["mt_cur"]))
add("fl", "масштаб по раскладкам", cv("fl", ["fl_cur"]))
add("fl", "fl_frag по раскладкам", cv("fl", ["fl_frag"]))
add("fl", "fl_line по раскладкам", cv("fl", ["fl_line"]))
add("fl", "trig + frag + line", cv("fl", ["fl_cur", "fl_frag", "fl_line"]))
add("fl", "trig + frag + line + mt", cv("fl", ["fl_cur", "fl_frag", "fl_line", "mt_raw"]))
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
best = r.loc[r.groupby("target").mae.idxmin()]
print("\nлучшее по каждому таргету:", best[["target", "variant"]].values.tolist())
print(f"скор: текущий {r[r.variant == 'текущий v4'].score_part.sum():.3f}, лучшие варианты {best.score_part.sum():.3f}")
print("\nFL по раскладкам, MAE: текущий / fl_frag*k / trig+frag+line")
d["p1"], d["p2"] = cv("fl", ["fl_frag"]), cv("fl", ["fl_cur", "fl_frag", "fl_line"])
print(d.groupby("layout").apply(lambda g: pd.Series({"n": len(g), "cur": np.abs(g.fl_cur - g.fl).mean(),
                                                     "frag": np.abs(g.p1 - g.fl).mean(),
                                                     "stack": np.abs(g.p2 - g.fl).mean()})).round(2))
