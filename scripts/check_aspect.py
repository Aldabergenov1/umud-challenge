"""Compare horizontal (bottom ticks) vs vertical (decoded ruler) scale per test-frame family."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

m = pd.read_csv("eda/test_meta.csv")
m["name"] = m.image_id.map(lambda s: Path(s.replace("\\", "/")).name)


def tick_positions(row_strip):
    """x positions of short bright tick marks in a thin horizontal strip (max over rows)."""
    v = row_strip.max(axis=0)
    on = v > 90
    xs = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    return xs


rows = []
sel = m[m.px_per_cm.round(1).isin([148.2, 152.2, 135.3, 121.8])].groupby(m.px_per_cm.round(1)).head(3) if len(sys.argv) > 1 else m.groupby("family").head(4)
for r in sel.itertuples():
    a = np.asarray(Image.open(f"data/raw/test_images_v2/test_set_v2/{r.name}").convert("L")).astype(float)
    best = None
    # search the bottom 40 px for the row band with the most regularly spaced ticks
    for y0 in range(a.shape[0] - 40, a.shape[0] - 2):
        xs = tick_positions(a[y0:y0 + 3])
        if len(xs) >= 4:
            d = np.diff(xs)
            if d.std() < 2.5 and (best is None or len(xs) > len(best[1])):
                best = (y0, xs)
    if best is None:
        rows.append((r.family, r.name, r.px_per_cm, None, None))
        continue
    d = np.median(np.diff(best[1]))
    rows.append((r.family, r.name, r.px_per_cm, round(d, 1), round(d / r.px_per_cm, 3)))
print(pd.DataFrame(rows, columns=["family", "image", "vert_px_cm", "bottom_tick_px", "ratio_h_over_v"]).to_string(index=False))
