"""Prototype of the v4 geometry on cached probability maps.

1. Deep aponeurosis: every band of the deep channel is a separate candidate; take the uppermost
   one below which almost no fascicles lie (two stacked muscles -> measure the upper one).
2. FL per fragment: extend every fascicle fragment to both aponeuroses and take a weighted median
   of these lengths, favouring fragments that need little extrapolation (how the hosts measure).

Prints benchmark scores for the FL variants and lists test frames where the deep line changed.
"""
import io
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from core import load_pipeline

P = load_pipeline()
CFG, TOL, WORK = P["CFG"], P["TOL"], P["WORK"]
k = CFG["px_per_mm"]


def edge_line(m, edge):
    xs = np.flatnonzero(m.any(axis=0))
    ys = np.array([np.flatnonzero(m[:, x])[-1 if edge == "lower" else 0] for x in xs])
    a, b = np.polyfit(xs, ys, 1)
    res = np.abs(ys - (a * xs + b))
    keep = res <= 3 * np.median(res) + 1
    return tuple(np.polyfit(xs[keep], ys[keep], 1))


def deep_candidates(prob, thr=0.25, min_span=0.3):
    """Upper-edge line of every band of the deep channel that spans enough of the width.

    A weak band often breaks into pieces, so pieces are joined along the horizontal first.
    """
    W = prob.shape[1]
    m0 = (prob > thr).astype(np.uint8)
    joined = cv2.morphologyEx(m0, cv2.MORPH_CLOSE, np.ones((3, int(10 * k)), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined)
    out = []
    for i in range(1, n):
        if st[i, 2] >= min_span * W:
            m = (lab == i) & (m0 > 0)
            out.append((edge_line(m, "upper"), float(np.nonzero(m)[0].mean())))
    return [c for c, _ in sorted(out, key=lambda c: c[1])]


def fasc_mass_between(p_fasc, top, bottom, W):
    xx = np.arange(W)
    yy = np.arange(p_fasc.shape[0])[:, None]
    region = (yy > top[0] * xx + top[1]) & (yy < bottom[0] * xx + bottom[1])
    region[:, : int(0.1 * W)] = False
    region[:, int(0.9 * W):] = False
    return float((p_fasc * (p_fasc > 0.5) * region).sum())


def choose_deep(pm, sup, deep, below_share=0.15):
    """Replace the default deep line by a higher band if hardly any fascicles lie below that band."""
    W = pm.shape[2]
    cands = [c for c in deep_candidates(pm[1]) if np.mean(c[0] * np.arange(W) + c[1] - (sup[0] * np.arange(W) + sup[1])) > 5 * k]
    if len(cands) < 2:
        return deep, False
    bottom = cands[-1]
    total = fasc_mass_between(pm[2], sup, bottom, W)
    if total <= 0:
        return deep, False
    for c in cands[:-1]:
        if fasc_mass_between(pm[2], c, bottom, W) < below_share * total:
            return c, True
    return deep, False


def line_hit(x, y, t, line):
    a, b = line
    den = np.sin(t) - a * np.cos(t)
    return np.nan if abs(den) < 1e-6 else (a * x + b - y) / den


def fl_fragments(segs, sup, deep):
    """Weighted median over fragments of the fragment line length between the two aponeuroses."""
    L, w = [], []
    for pa, length, x, y, t in segs:
        s1, s2 = line_hit(x, y, t, sup), line_hit(x, y, t, deep)
        if np.isnan(s1) or np.isnan(s2):
            continue
        full = abs(s1 - s2)
        L.append(full / k)
        w.append(length * min(1.0, length / full))   # long fragments that need little extrapolation
    return P["weighted_median"](L, w) if len(L) >= 3 else np.nan


orig_measure = P["measure"]


def measure_v4(g, probs, s, h_over_v=1.0):
    out, d = orig_measure(g, probs, s, h_over_v, return_debug=True)
    out = dict(out, fl_frag=np.nan, deep_moved=False)
    if d is None:
        return out
    deep, moved = choose_deep(d["pm"], d["sup"], d["deep"])
    if moved:
        # re-run the whole geometry with the new deep line by masking the channel below it
        pm = d["pm"].copy()
        H, W = pm.shape[1:]
        yy = np.arange(H)[:, None]
        line_y = deep[0] * np.arange(W) + deep[1]
        pm[1] = ((yy >= line_y) & (yy <= line_y + k)).astype(np.float32)   # the chosen band, 1 mm thick
        pm[2][yy > line_y] = 0
        g2 = d["gm"]
        CFG_px = CFG["px_per_mm"]
        out, d = orig_measure(g2, pm, CFG_px, 1.0, return_debug=True)   # already metric: 1 px/mm scale = k
        out = dict(out, fl_frag=np.nan, deep_moved=True)
        if d is None:
            return out
    out["fl_frag"] = fl_fragments(d["segs"], d["sup"], d["deep"])
    return out


if __name__ == "__main__":
    bench = pd.read_excel(P["BENCH"] / "Results_benchmark_architecture_v0.1.0.xlsx")
    truth = pd.DataFrame({t: bench[[f"R{i}_{t.upper()}" for i in range(1, 8)]].mean(axis=1) for t in TOL})
    b_rois, b_probs = pickle.load(open(WORK / "bench_probs.pkl", "rb"))
    b = pd.DataFrame([measure_v4(r, p.astype(np.float32), s / 10) for r, p, s in zip(b_rois, b_probs, bench.Scale_pixel_per_cm)])
    print("benchmark: deep line moved on", int(b.deep_moved.sum()), "of", len(b))
    pa = b.pa + 0.318
    trig = b.mt / np.sin(np.radians(pa))
    for name, fl in [("fl_trig", trig), ("0.2 line + 0.8 trig", 0.2 * b.fl_line * 0.982 + 0.8 * trig * 0.982),
                     ("fl_frag", b.fl_frag), ("0.5 frag + 0.5 trig", 0.5 * b.fl_frag + 0.5 * trig)]:
        e = fl - truth.fl
        ratio = np.nanmedian(truth.fl / fl)
        print(f"  {name:22s} MAE {e.abs().mean():5.2f}  bias {e.mean():+5.2f}  corr {np.corrcoef(fl.fillna(fl.median()), truth.fl)[0, 1]:.2f}"
              f"  MAE after median scale {(fl * ratio - truth.fl).abs().mean():5.2f}")

    meta = P["test_meta"]()
    t_rois, t_probs = pickle.load(open(WORK / "test_probs.pkl", "rb"))
    rows = []
    for r in meta.itertuples():
        res = measure_v4(t_rois[r.image_id], t_probs[r.image_id].astype(np.float32), r.px_per_cm / 10,
                         CFG["h_over_v"].get(r.layout, 1.0))
        rows.append(dict(image_id=r.image_id, layout=r.layout, **res))
    t = pd.DataFrame(rows)
    t.to_csv(WORK / "test_geom_v4.csv", index=False)
    print("test: deep line moved on", int(t.deep_moved.sum()), "frames:", t[t.deep_moved].image_id.tolist())
    print(t[t.deep_moved][["image_id", "layout", "pa", "mt", "fl_trig", "fl_frag"]].round(1).to_string(index=False))
    print("test median by layout:")
    print(t.groupby("layout")[["fl_trig", "fl_frag", "fl_line"]].median().round(1))
