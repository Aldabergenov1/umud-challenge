"""Learned geometry: per-frame features from the network output -> correction of PA, FL, MT.

The network and the hand-written geometry stay; a gradient-boosting model learns the residual between
the current submission (v4) and the manual labels, from features of the fascicle fragments and the
aponeurosis lines. Cross-fitted by series: a frame is always predicted by a model that has not seen any
frame of its series. Labels are corrected to the hosts' convention as in analyze_labels.py.
"""
import io
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))
from core import load_pipeline

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
FEATS = ROOT / "work/frame_features.csv"


def wq(v, w, q):
    """Weighted quantile."""
    o = np.argsort(v)
    c = np.cumsum(np.asarray(w)[o])
    return float(np.asarray(v)[o][min(len(v) - 1, np.searchsorted(c, q * c[-1]))])


def frame_features(P, meta, rois, probs):
    k = P["CFG"]["px_per_mm"]
    rows = []
    for r in meta.itertuples():
        res, d = P["measure"](rois[r.image_id], probs[r.image_id].astype(np.float32), r.px_per_cm / 10,
                              P["CFG"]["h_over_v"].get(r.layout, 1.0), return_debug=True)
        f = dict(image_id=r.image_id, **{f"raw_{a}": b for a, b in res.items() if np.isscalar(b)})
        if d is not None:
            (a1, b1), (a2, b2) = d["sup"], d["deep"]
            W = d["gm"].shape[1]
            f.update(sup_deg=np.degrees(np.arctan(a1)), deep_deg=np.degrees(np.arctan(a2)),
                     converge_deg=np.degrees(np.arctan(a2) - np.arctan(a1)), width_mm=W / k,
                     mt_left=P["perpendicular_mt"](d["sup"], d["deep"], 0.1 * W) / k,
                     mt_right=P["perpendicular_mt"](d["sup"], d["deep"], 0.9 * W) / k)
            segs = d["segs"]
            if len(segs) >= 3:
                pa = np.array([s[0] for s in segs])
                L = np.array([s[1] for s in segs]) / k
                x = np.array([s[2] for s in segs]) / W
                depth = np.array([(s[3] - (a1 * s[2] + b1)) / max(1, (a2 - a1) * s[2] + b2 - b1) for s in segs])
                f.update({f"pa_q{int(q * 100)}": wq(pa, L, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
                f.update(seg_n=len(segs), seg_len_sum=L.sum(), seg_len_max=L.max(), pa_std=np.sqrt(np.cov(pa, aweights=L)),
                         pa_long=wq(pa, L ** 3, 0.5))
                for name, m in (("deep", depth > 0.5), ("shallow", depth <= 0.5),
                                ("centre", (x > 0.3) & (x < 0.7)), ("edge", (x <= 0.3) | (x >= 0.7))):
                    f[f"pa_{name}"] = wq(pa[m], L[m], 0.5) if m.sum() >= 2 else np.nan
                    f[f"len_{name}"] = L[m].sum()
        rows.append(f)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = load_pipeline()
    meta = P["test_meta"]()
    if not FEATS.exists():
        rois, probs = pickle.load(open(ROOT / "work/test_probs.pkl", "rb"))
        feats = frame_features(P, meta, rois, probs)
        spread = pd.read_csv(ROOT / "work/per_fold_test.csv")
        sp = spread.groupby("image_id")[["pa", "mt", "fl_trig"]].std().add_prefix("fold_std_") \
            if "image_id" in spread else None
        if sp is not None:
            feats = feats.merge(sp, left_on="image_id", right_index=True, how="left")
        feats.to_csv(FEATS, index=False)
    feats = pd.read_csv(FEATS)

    lab = pd.read_csv(ROOT / "work/label_errors_v4.csv")      # corrected labels + v4 predictions + series
    d = lab.merge(feats, on="image_id")
    d = d.merge(meta[["image_id", "px_per_cm", "depth_cm"]], on="image_id")
    d["layout_code"] = d.layout.astype("category").cat.codes
    X_cols = [c for c in feats.columns if c != "image_id"] + ["px_per_cm", "depth_cm", "layout_code",
                                                               "pa_deg", "fl_mm", "mt_mm"]
    # residual targets; FL in log space
    targets = {"pa": d.pa - d.pa_deg, "fl": np.log(d.fl) - np.log(d.fl_mm), "mt": d.mt - d.mt_mm}
    # series-mean targets: the annotator labelled every frame of a series, averaging cuts his noise
    targets = {t: v.groupby(d.series).transform("mean") for t, v in targets.items()}

    out = {}
    for t, y in targets.items():
        pred = np.zeros(len(d))
        for seed in range(5):
            ids = d.series.unique()
            perm = dict(zip(ids, np.random.default_rng(seed).permutation(len(ids))))
            for tr, te in GroupKFold(5).split(d, y, d.series.map(perm)):
                m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05,
                                                  max_iter=200, min_samples_leaf=15, l2_regularization=1.0,
                                                  categorical_features=[X_cols.index("layout_code")],
                                                  random_state=seed)
                m.fit(d.loc[tr, X_cols], y.iloc[tr])
                pred[te] += m.predict(d.loc[te, X_cols]) / 5
        out[t] = pd.Series(pred).groupby(d.series).transform("median")

    new = pd.DataFrame({"pa": d.pa_deg + out["pa"], "fl": d.fl_mm * np.exp(out["fl"]), "mt": d.mt_mm + out["mt"]})
    rows = []
    for name, p in (("v4", d[["pa_deg", "fl_mm", "mt_mm"]].set_axis(["pa", "fl", "mt"], axis=1)), ("обучаемая", new)):
        mae = {t: (p[t] - d[t]).abs().mean() for t in TOL}
        rows.append(dict(variant=name, score=np.mean([mae[t] / TOL[t] for t in TOL]), **mae))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    d["err_v4"] = sum((d[c] - d[t]).abs() / TOL[t] for t, c in zip(TOL, ["pa_deg", "fl_mm", "mt_mm"])) / 3
    d["err_new"] = sum((new[t] - d[t]).abs() / TOL[t] for t in TOL) / 3
    print(d.groupby("layout")[["err_v4", "err_new"]].mean().round(3))
