"""Check notebooks/umud_scale.py against eda/test_meta.csv and depth plausibility per family."""
import sys
from pathlib import Path

sys.path.insert(0, "notebooks")
import numpy as np
import pandas as pd
from PIL import Image

from umud_scale import decode_scale

m = pd.read_csv("eda/test_meta.csv")
m["name"] = m.image_id.map(lambda s: Path(s.replace("\\", "/")).name)
bad = 0
for r in m.itertuples():
    a = np.asarray(Image.open(f"data/raw/test_images_v2/test_set_v2/{r.name}"))
    px, box = decode_scale(a, r.filetype)
    if abs(px - r.px_per_cm) > 0.3 or box[0] != r.l or box[2] != r.r:
        bad += 1
        print(r.name, px, r.px_per_cm, box, (r.l, r.t, r.r, r.b))
print("mismatches", bad, "of", len(m))

m["depth_off"] = (m.depth_cm - (m.depth_cm * 2).round() / 2).abs()
print(m.groupby("family").agg(n=("id", "size"), depth_mean=("depth_cm", "mean"),
                              max_off_from_half_cm=("depth_off", "max"),
                              width_min=("width_cm", "min"), width_max=("width_cm", "max")).round(3).to_string())
print(m[m.family == "tif_800x1200"].groupby(m.px_per_cm.round(1)).size().to_string())
print(m[m.family == "png_800x1200"].groupby(m.px_per_cm.round(1)).size().to_string())
