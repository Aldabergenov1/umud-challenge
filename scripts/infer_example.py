"""Inference example: benchmark frames (prediction vs expert consensus) + one test frame.

Runs the notebook code (no training), applies work/calibration.json, saves eda/fig_inference.png
and prints the per-image table.
"""
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: plt.close("all")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
os.chdir(ROOT / "notebooks")

import nbformat

nb = nbformat.read(ROOT / "notebooks/umud_pipeline.ipynb", as_version=4)
code = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
g = {}
exec(compile(code[: code.index("bench_rows = []")], "notebook", "exec"), g)
np, pd = g["np"], g["pd"]
calib = json.load(open(ROOT / "work/calibration.json"))


def infer(roi, px_per_mm, h_over_v=1.0):
    """Full inference for one ultrasound region: masks -> geometry -> calibration."""
    probs = np.concatenate([g["predict_probs"](g["MODELS"]["apo"], roi), g["predict_probs"](g["MODELS"]["fasc"], roi)])
    raw = g["measure"](roi, probs, px_per_mm, h_over_v)
    w = calib["fl_weight"]
    fl = w * raw["fl_line"] + (1 - w) * raw["fl_trig"]
    out = {"pa": raw["pa"], "fl": fl, "mt": raw["mt"]}
    for t in out:
        kind, p = calib[t]
        out[t] = out[t] + p if kind == "offset" else out[t] * p if kind == "scale" else out[t]
    return out, probs


bench, truth = g["bench"], g["truth"]
rows = []
for i, r in enumerate(bench.itertuples()):
    im = g["read_gray"](g["BENCH"] / f"{r.ImageID}.tif")
    l, t, rr, b = g["find_roi"](im)
    pred, _ = infer(im[t:b, l:rr], r.Scale_pixel_per_cm / 10)
    err = {k: abs(pred[k] - truth[k][i]) for k in pred}
    rows.append(dict(image=r.ImageID, **{f"pred_{k}": pred[k] for k in pred},
                     **{f"true_{k}": truth[k][i] for k in pred},
                     score=np.mean([err[k] / g["TOL"][k] for k in err])))
res = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
picks = [res.iloc[len(res) // 4], res.iloc[len(res) // 2], res.iloc[-1]]   # good, typical, worst

fig, axes = plt.subplots(2, 4, figsize=(22, 10), gridspec_kw={"height_ratios": [1, 1]})
for col, row in enumerate(picks):
    r = bench[bench.ImageID == row.image].iloc[0]
    im = g["read_gray"](g["BENCH"] / f"{r.ImageID}.tif")
    l, t, rr, b = g["find_roi"](im)
    roi = im[t:b, l:rr]
    _, probs = infer(roi, r.Scale_pixel_per_cm / 10)
    axes[0, col].imshow(im, cmap="gray")
    axes[0, col].add_patch(plt.Rectangle((l, t), rr - l, b - t, fill=False, ec="yellow", lw=1.5))
    axes[0, col].set_title(f"ВХОД: {r.ImageID}.tif, {r.Scale_pixel_per_cm} px/см", fontsize=10)
    axes[0, col].axis("off")
    g["show_measure"](roi, probs, r.Scale_pixel_per_cm / 10, ax=axes[1, col], title="")
    axes[1, col].set_title("")
    label = ["хороший", "типичный", "худший из 35"][col]
    axes[1, col].set_title(
        f"{label} (score {row.score:.2f})\n"
        f"модель:   PA {row.pred_pa:5.1f}°  FL {row.pred_fl:5.1f} мм  MT {row.pred_mt:4.1f} мм\n"
        f"эксперты: PA {row.true_pa:5.1f}°  FL {row.true_fl:5.1f} мм  MT {row.true_mt:4.1f} мм",
        fontsize=10, family="monospace", loc="left")

# one test frame: no ground truth, just the submission row
meta = g["meta"]
r = meta[meta.image_id == "IMG_00001.tif"].iloc[0]
im = g["read_gray"](r.path)
roi = im[r.t:r.b, r.l:r.r]
pred, probs = infer(roi, r.px_per_cm / 10, g["CFG"]["h_over_v"].get(r.layout, 1.0))
axes[0, 3].imshow(im, cmap="gray")
axes[0, 3].add_patch(plt.Rectangle((r.l, r.t), r.r - r.l, r.b - r.t, fill=False, ec="yellow", lw=1.5))
axes[0, 3].set_title(f"ВХОД: тест {r.image_id}, {r.px_per_cm:.1f} px/см ({r.layout})", fontsize=10)
axes[0, 3].axis("off")
g["show_measure"](roi, probs, r.px_per_cm / 10, ax=axes[1, 3], title="")
axes[1, 3].set_title("")
axes[1, 3].set_title(f"тест: ответа нет, строка сабмита\n{r.image_id},{pred['pa']:.2f},{pred['fl']:.2f},{pred['mt']:.2f}",
                     fontsize=10, family="monospace", loc="left")
plt.tight_layout()
plt.savefig(ROOT / "eda/fig_inference.png", dpi=90)

pd.set_option("display.width", 200)
print(res.round(2).to_string())
print("\nсредний score по 35:", round(res.score.mean(), 3))
print("тест", r.image_id, {k: round(v, 2) for k, v in pred.items()})
