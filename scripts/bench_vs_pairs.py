"""How would our benchmark predictions score against a 2-rater mean (the test-label format)?

Also measures sensitivity to prediction averaging (hflip TTA on/off) as a proxy for
what extra folds could give. Calibration is leave-one-out, so numbers are honest.
"""
import io
import itertools
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
np, pd, torch, cv2 = g["np"], g["pd"], g["torch"], g["cv2"]
bench, TOL = g["bench"], g["TOL"]
raters = [f"R{i}" for i in range(1, 8)]
calib = json.load(open(ROOT / "work/calibration.json"))


@torch.no_grad()
def probs_no_tta(models, img):
    S = g["CFG"]["img_size"]
    x = torch.from_numpy(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)[None, None].to(g["DEVICE"])
    with torch.autocast(g["DEVICE"], dtype=torch.bfloat16):
        p = torch.sigmoid(models[0](x).float())[0].cpu().numpy()
    return np.stack([cv2.resize(c, img.shape[::-1], interpolation=cv2.INTER_LINEAR) for c in p])


def raw_predictions(tta: bool) -> pd.DataFrame:
    rows = []
    for r in bench.itertuples():
        im = g["read_gray"](g["BENCH"] / f"{r.ImageID}.tif")
        l, t, rr, b = g["find_roi"](im)
        roi = im[t:b, l:rr]
        f = g["predict_probs"] if tta else probs_no_tta
        probs = np.concatenate([f(g["MODELS"]["apo"], roi), f(g["MODELS"]["fasc"], roi)])
        rows.append(g["measure"](roi, probs, r.Scale_pixel_per_cm / 10))
    df = pd.DataFrame(rows)
    w = calib["fl_weight"]
    df["fl"] = w * df.fl_line + (1 - w) * df.fl_trig
    return df[["pa", "fl", "mt"]]


def loo_calibrated(raw: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Same correction kinds as the notebook, but each image corrected with params fit on the other 34."""
    out = raw.copy()
    for t in TOL:
        kind = calib[t][0]
        for i in range(len(raw)):
            m = np.arange(len(raw)) != i
            x, y = raw[t][m], truth[t][m]
            p = np.median(y - x) if kind == "offset" else np.median(y / x) if kind == "scale" else 0
            out.loc[i, t] = raw[t][i] + p if kind == "offset" else raw[t][i] * p if kind == "scale" else raw[t][i]
    return out


def score(pred, ref):
    return np.mean([(pred[t] - ref[t]).abs().mean() / TOL[t] for t in TOL])


def mean_of(cols):
    return pd.DataFrame({t: bench[[f"{c}_{t.upper()}" for c in cols]].mean(axis=1) for t in TOL})


consensus = mean_of(raters)
pairs = list(itertools.combinations(raters, 2))
for tta in (True, False):
    raw = raw_predictions(tta)
    pred = loo_calibrated(raw, consensus)
    vs_pairs = [score(pred, mean_of(p)) for p in pairs]
    print(f"TTA={tta}: против консенсуса 7 = {score(pred, consensus):.3f} | "
          f"против пары экспертов: среднее {np.mean(vs_pairs):.3f}, мин {np.min(vs_pairs):.3f}, макс {np.max(vs_pairs):.3f}")
    if tta:
        no19 = [i for i in range(len(bench)) if bench.ImageID[i] != "im_19_arch"]
        print(f"   без провала im_19: против пары {np.mean([score(pred.iloc[no19], mean_of(p).iloc[no19]) for p in pairs]):.3f}")

print("\nОриентиры против пары экспертов (пары из остальных шести):")
for who in raters + ["DLTrack"]:
    single = pd.DataFrame({t: bench[f"{who}_{t.upper()}"] for t in TOL})
    others = [r for r in raters if r != who]
    print(f"  {who:8s} {np.mean([score(single, mean_of(p)) for p in itertools.combinations(others, 2)]):.3f}")
