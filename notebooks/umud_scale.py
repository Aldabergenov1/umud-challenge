"""Pixel scale (px/cm) and ultrasound-region box for UMUD test frames.

Tick-mark decoder adapted from AmbrosM's public Kaggle notebook "UMUD Quick and Dirty".
Every test frame belongs to one of a few device layouts; each layout has its own
depth ruler. decode_scale returns (px_per_cm, (left, top, right, bottom)); the scale
is the VERTICAL one (depth ruler), as the organisers instruct. layout_name tells
which layout a frame has, e.g. to apply a horizontal correction to one of them.
"""
import numpy as np

# Ultrasound box for the right-ruler 1200x800 layout, by number of 5 mm ticks (depth).
_RIGHT_RULER_BOX = {
    7: (142, 1058), 8: (163, 1037), 9: (211, 989), 10: (249, 951),
    11: (282, 918), 12: (308, 892), 13: (331, 869), 14: (349, 851), 15: (142, 1058),
}


def _first_bright(col, start=0, thr=50):
    return start + int(np.argmax(col[start:] > thr))


def _last_bright(col, thr=50):
    return len(col) - 1 - int(np.argmax(col[::-1] > thr))


def decode_scale(a: np.ndarray, filetype: str):
    """a: full test frame (H, W) or (H, W, 3) uint8; filetype: 'png' or 'tif'."""
    g = a.mean(axis=-1) if a.ndim == 3 else a.astype(float)
    h, w = g.shape

    if filetype == "png":  # Philips Lumify: ruler at the left border, 2.85 cm wide sector
        first = _first_bright(g[:, 6])
        minor = _first_bright(g[:, 6], 150)
        major = _first_bright(g[:, 9], 150)
        last = _last_bright(g[:, 6])
        px_cm = major - first if (major - first) < 3 * (minor - first) else minor - first
        half = int(np.argmin(g[:, w // 2:].sum(axis=0)))
        return float(px_cm), (w // 2 - half, first, w // 2 + half, last)

    if (h, w) == (800, 1200):
        if g[87, 1147:1157].mean() > 150:  # ticks at the right border, depth varies
            col = g[:, 1150]
            first = _first_bright(col)
            major = _first_bright(col, first + 20)
            last = _last_bright(col)
            n_ticks = int(round((last - first) / max(1, major - first)))
            px_cm = (last - first) / n_ticks * 2 if n_ticks <= 14 else (last - first) / 3
            l, r = _RIGHT_RULER_BOX.get(n_ticks, (142, 1058))
            return float(px_cm), (l, 91, r, last)
        return (783 - 42) / 5, (171, 42, 1029, 798)  # ticks at the left border, 5 cm deep

    if (h, w) == (644, 1088):
        return 630.5 / 5, (140, 0, 947, 643)
    if h in (512, 513):  # bottom ruler, 1 cm ticks
        return (442 - 53) / 5, (0, 0, w, h - 10)
    if h == 853:
        return (934 - 100) / 5, (0, 0, w, h)
    raise ValueError(f"unknown frame layout {a.shape}")


def layout_name(a: np.ndarray, filetype: str) -> str:
    g = a.mean(axis=-1) if a.ndim == 3 else a
    h, w = g.shape
    if filetype == "png":
        return "lumify"
    if (h, w) == (800, 1200):
        return "siemens_right_ruler" if g[87, 1147:1157].mean() > 150 else "telemed_left_ruler"
    if (h, w) == (644, 1088):
        return "telemed_644"
    if h in (512, 513):
        return "bottom_ruler_513"
    return "bottom_ruler_853"
