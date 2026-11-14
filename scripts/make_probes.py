"""Single-column probes of the best submission: one target shifted, the other two unchanged."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
base_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work/sub_v3_5fold_pa2.csv"
base = pd.read_csv(base_path)
probes = {
    "probe_mt_minus075.csv": base.assign(mt_mm=(base.mt_mm - 0.75).clip(10, 50)),
    "probe_fl_x095.csv": base.assign(fl_mm=(base.fl_mm * 0.95).clip(30, 200)),
    "probe_pa_plus1_fl_fixed.csv": base.assign(pa_deg=(base.pa_deg + 1.0).clip(5, 45)),
}
for name, df in probes.items():
    df.to_csv(ROOT / "work/probes" / name, index=False)
    changed = [c for c in ("pa_deg", "fl_mm", "mt_mm") if (df[c] != base[c]).any()]
    print(f"{name}: changed {changed}, rows {len(df)}")
