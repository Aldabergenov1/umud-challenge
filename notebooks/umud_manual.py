"""PA, FL and MT from manual clicks, the way the hosts measure.

An annotation holds points in full-frame pixels:
  sup   - points along the lower (inner) edge of the superficial aponeurosis
  deep  - points along the upper (inner) edge of the deep aponeurosis
  fasc  - list of fascicles, each two points along a clearly visible fascicle

Each aponeurosis becomes a straight line. MT is the mean perpendicular distance at 25, 50 and 75 %
of the ultrasound region width. Every fascicle is extended to both aponeuroses: FL is the length of
that line, PA its angle to the deep aponeurosis. The frame value is the mean over fascicles.
Pixels are taken as square, with the vertical scale, as the hosts do.
"""
import numpy as np


def fit_line(points):
    """y = a*x + b through two or more points."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2 or np.ptp(p[:, 0]) < 1:
        return None
    a, b = np.polyfit(p[:, 0], p[:, 1], 1)
    return float(a), float(b)


def angle_between(t1, t2):
    """Unsigned angle between two line orientations, degrees in [0, 90]."""
    d = (t1 - t2 + np.pi / 2) % np.pi - np.pi / 2
    return abs(np.degrees(d))


def line_hit(x, y, t, line):
    """Signed distance along direction t from (x, y) to the line y = a*x + b."""
    a, b = line
    den = np.sin(t) - a * np.cos(t)
    return np.nan if abs(den) < 1e-9 else (a * x + b - y) / den


def measure_manual(ann: dict, px_per_cm: float, box) -> dict:
    """PA (deg), FL (mm), MT (mm) and per-fascicle values; NaN where clicks are missing."""
    out = dict(pa=np.nan, fl=np.nan, mt=np.nan, n_fasc=0, fascicles=[])
    px_mm = px_per_cm / 10
    sup, deep = fit_line(ann.get("sup", [])), fit_line(ann.get("deep", []))
    if sup is None or deep is None:
        return out
    l, _, r, _ = box
    mts = [abs(deep[0] * x + deep[1] - (sup[0] * x + sup[1])) / np.hypot(deep[0], 1)
           for x in (l + f * (r - l) for f in (0.25, 0.5, 0.75))]
    out["mt"] = float(np.mean(mts)) / px_mm
    t_deep = np.arctan(deep[0])
    for (x1, y1), (x2, y2) in ann.get("fasc", []):
        t = np.arctan2(y2 - y1, x2 - x1)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        fl = abs(line_hit(xm, ym, t, sup) - line_hit(xm, ym, t, deep)) / px_mm
        if np.isfinite(fl):
            out["fascicles"].append(dict(pa=float(angle_between(t, t_deep)), fl=float(fl)))
    if out["fascicles"]:
        out["n_fasc"] = len(out["fascicles"])
        out["pa"] = float(np.mean([f["pa"] for f in out["fascicles"]]))
        out["fl"] = float(np.mean([f["fl"] for f in out["fascicles"]]))
    return out
