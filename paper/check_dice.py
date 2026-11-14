"""Validation Dice of the square aponeurosis U-Nets, each fold on its own held-out groups. CPU is enough.

    CUDA_VISIBLE_DEVICES= python paper/check_dice.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from core import load_pipeline

P = load_pipeline()
CFG = P["CFG"]
d = np.load(P["WORK"] / f"cache_apo_{CFG['img_size']}.npz")
x, y, groups = d["x"], d["y"], d["groups"]
scores = []
for fold, (_, va) in enumerate(GroupKFold(CFG["n_folds"]).split(x, groups=groups)):
    m = P["make_model"](2, pretrained=False)
    m.load_state_dict(torch.load(P["WORK"] / f"apo_fold{fold}.pt", map_location="cpu"))
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(va), 8):
            b = va[i:i + 8]
            xb = torch.from_numpy(x[b].astype(np.float32) / 255)[:, None]
            out.append(P["dice_score"](m(xb), torch.from_numpy(y[b]).float()))
    scores.append(np.mean(out, axis=0))
    print(f"fold {fold}: n {len(va)}, Dice superficial {scores[-1][0]:.3f}, deep {scores[-1][1]:.3f}", flush=True)
print("mean", np.mean(scores, axis=0).round(3), flush=True)
