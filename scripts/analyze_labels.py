"""What the full manual labels (annotations/) say about the pipeline.

1. Labels vs the public LB: per-target convention offset of the annotator relative to the hosts,
   fitted so that local score changes of the LB probes match the LB changes.
2. Annotator noise: spread of labels inside 5-frame series (near-identical frames labelled separately).
3. Scale: annotator's scan-box scale vs our ruler decoder, per layout.
4. Errors of the current best submission per layout and the worst series.
"""
import io
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import manual_labels as M

TOL, COLS = M.TOL, M.COLS
LB = {"sub_v2_segments.csv": 0.52876, "sub_v2_pa+2.csv": 0.47986, "sub_v2_pa+4.csv": 0.60986,
      "sub_v3_5fold_pa2.csv": 0.44112, "probe_fl_x095.csv": 0.49587, "probe_mt_minus075.csv": 0.48419,
      "probe_pa_plus1_fl_fixed.csv": 0.44725, "probe_mt_fl_x104.csv": 0.50474}


def score(sub, lab):
    m = lab.merge(sub, on="image_id")
    return np.mean([(m[COLS[t]] - m[t]).abs().mean() / TOL[t] for t in TOL])


meta = M.test_boxes()
src = M.from_annotator_dir(ROOT / "annotations")
lab = M.label_table(src, meta)
subs = {n: pd.read_csv(ROOT / ("work/probes/" if n.startswith("probe") else "work/") / n) for n in LB}

# 1. convention offset: label' = label*k (MT, FL), label + o (PA)
print("1. Согласие с LB")
base = {n: score(s, lab) for n, s in subs.items()}
lb = pd.Series(LB)
print("   корреляция локальный/LB как есть:", round(np.corrcoef(lb, pd.Series(base))[0, 1], 3))
aligned = {n: lab[["image_id"]].merge(s, on="image_id") for n, s in subs.items()}


def fast_scores(o, kf, km):
    return pd.Series({n: np.mean([np.abs(a.pa_deg.values - lab.pa.values - o).mean() / 6,
                                  np.abs(a.fl_mm.values - lab.fl.values * kf).mean() / 12,
                                  np.abs(a.mt_mm.values - lab.mt.values * km).mean() / 3]) for n, a in aligned.items()})


best = None
for o, kf, km in product(np.arange(-2, 2.01, 0.25), np.arange(0.85, 1.051, 0.01), np.arange(0.85, 1.051, 0.01)):
    loc = fast_scores(o, kf, km)
    err = ((loc - loc.mean()) - (lb - lb.mean())).abs().mean()   # match differences, not the level
    if best is None or err < best[0]:
        best = (err, o, kf, km, loc)
err, o, kf, km, loc = best
print(f"   смещение разметчика к хостам: PA {o:+.2f}°, FL ×{kf:.2f}, MT ×{km:.2f}; "
      f"средняя невязка разностей {err:.3f}; корреляция {np.corrcoef(lb, loc)[0, 1]:.3f}")
print(pd.DataFrame({"LB": lb, "лок. как есть": pd.Series(base), "лок. с поправкой": loc}).round(3).to_string())

# 2. annotator noise inside series
print("\n2. Шум разметки внутри 5-кадровых серий (кадры почти одинаковые)")
P = __import__("core").load_pipeline()
rois = [P["read_gray"](r.path)[r.t:r.b, r.l:r.r] for r in meta.reset_index().itertuples()]
meta["series"] = P["frame_groups"](rois)
lab = lab.join(meta.series, on="image_id")
lab["n_series"] = lab.groupby("series").image_id.transform("size")
multi = lab[lab.n_series > 1]
noise = {t: (multi[t] - multi.groupby("series")[t].transform("median")).abs().mean() for t in TOL}
print("   среднее отклонение от медианы серии:", {t: round(v, 2) for t, v in noise.items()},
      "в единицах метрики:", round(np.mean([noise[t] / TOL[t] for t in TOL]), 3))

# 3. scale
print("\n3. Масштаб друга (рамка скана) против нашего декодера шкалы, медиана отношения мм/px по раскладкам")
their = pd.DataFrame({k: dict(mmpp=json.loads((ROOT / f"annotations/{Path(k).stem}.json").read_text())["mm_per_pixel"])
                      for k in src}).T
their = their.join(meta[["layout", "px_per_cm"]])
their["ratio"] = their.mmpp.astype(float) / (10 / their.px_per_cm)
print(their.groupby("layout").ratio.describe()[["count", "50%", "min", "max"]].round(3))

# 4. errors of the best local submission
name = "sub_v4_geom.csv"
sub = pd.read_csv(ROOT / "work" / name)
adj = lab.assign(pa=lab.pa + o, fl=lab.fl * kf, mt=lab.mt * km)
m = adj.merge(sub, on="image_id")
for t in TOL:
    m[f"e_{t}"] = m[COLS[t]] - m[t]
m["err"] = sum(m[f"e_{t}"].abs() / TOL[t] for t in TOL) / 3
print(f"\n4. {name} против меток с поправкой: {score(sub, adj):.3f}")
print(m.groupby("layout").agg(n=("err", "size"), score=("err", "mean"),
                              **{f"bias_{t}": (f"e_{t}", "mean") for t in TOL},
                              **{f"mae_{t}": (f"e_{t}", lambda s: s.abs().mean()) for t in TOL}).round(2))
share = m.groupby("layout").err.sum() / m.err.sum()
print("   доля общей ошибки по раскладкам:", share.round(2).to_dict())
print("   доля ошибки по таргетам:", {t: round((m[f"e_{t}"].abs() / TOL[t]).sum() / 3 / m.err.sum(), 2) for t in TOL})
w = m.groupby("series").agg(layout=("layout", "first"), first=("image_id", "first"), n=("err", "size"),
                            err=("err", "mean"), e_pa=("e_pa", "mean"), e_fl=("e_fl", "mean"), e_mt=("e_mt", "mean"))
w["share"] = w.err * w.n / m.err.sum()
print("   худшие серии/кадры:")
print(w.sort_values("share", ascending=False).head(15).round(2).to_string())
m.to_csv(ROOT / "work/label_errors_v4.csv", index=False)
