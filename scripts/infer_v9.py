"""Probability maps of the v9 networks for the test set and the expert benchmark.

Usage: python scripts/infer_v9.py [tag] [run ...]
  tag prefixes the output folders (default none); runs limit which networks are used (default all).
Writes work/v9/<variant>/{test,bench}_probs.pkl in the formats of work/{test,bench}_probs.pkl, with two
extra channels (cos 2t, sin 2t) of the fascicle direction:
  cnx      aponeuroses apo_cnx, fascicles fasc_cnx
  ens      aponeuroses apo_cnx, fascicles mean of fasc_cnx and fasc_eff
  mix      as ens, plus the old square-input networks averaged into both mask maps
A run without weights falls back to the old maps for its channels.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import load_pipeline
from umud_aspect import predict_tiles

P = load_pipeline()
DEVICE = P["DEVICE"]
torch.set_grad_enabled(False)
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(P["CFG"]["vram_fraction"])
W9 = ROOT / "work/v9"
ENC = {"fasc_cnx": ("tu-convnext_tiny", 3), "apo_cnx": ("tu-convnext_tiny", 2), "fasc_eff": ("efficientnet-b4", 3)}


def load_run(name):
    enc, classes = ENC[name]
    models = []
    for p in sorted(W9.glob(f"{name}_fold*.pt")):
        m = smp.Unet(enc, encoder_weights=None, in_channels=1, classes=classes)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models.append(m.to(DEVICE).eval())
    print(name, len(models), "folds")
    return models


def maps(roi, old, nets):
    """Channels: sup, deep, fascicle, cos 2t, sin 2t for each variant."""
    old = old.astype(np.float32)
    apo = predict_tiles(nets["apo_cnx"], roi, 512, DEVICE, 2) if nets["apo_cnx"] else old[:2]
    zeros = np.zeros((2, *roi.shape), np.float32)
    fc = predict_tiles(nets["fasc_cnx"], roi, 512, DEVICE, 1, orient=True) if nets["fasc_cnx"] \
        else np.concatenate([old[2:3], zeros])
    fe = predict_tiles(nets["fasc_eff"], roi, 512, DEVICE, 1, orient=True) if nets["fasc_eff"] else fc
    ens_f = (fc + fe) / 2
    out = {"cnx": np.concatenate([apo, fc]), "ens": np.concatenate([apo, ens_f])}
    mix_apo = (apo + old[:2]) / 2
    mix_f = np.concatenate([(2 * ens_f[:1] + old[2:3]) / 3, ens_f[1:]])
    out["mix"] = np.concatenate([mix_apo, mix_f])
    return {k: v.astype(np.float16) for k, v in out.items()}


if __name__ == "__main__":
    tag, use = (sys.argv[1] if len(sys.argv) > 1 else ""), (sys.argv[2:] or list(ENC))
    nets = {name: load_run(name) if name in use else [] for name in ENC}
    t_rois, t_old = pickle.load(open(ROOT / "work/test_probs.pkl", "rb"))
    b_rois, b_old = pickle.load(open(ROOT / "work/bench_probs.pkl", "rb"))
    res_t = {v: {} for v in ("cnx", "ens", "mix")}
    for n, (iid, roi) in enumerate(t_rois.items()):
        for v, m in maps(roi, t_old[iid], nets).items():
            res_t[v][iid] = m
        if n % 100 == 0:
            print("test", n)
    res_b = {v: [] for v in res_t}
    for roi, old in zip(b_rois, b_old):
        for v, m in maps(roi, old, nets).items():
            res_b[v].append(m)
    for v in res_t:
        (W9 / f"{tag}{v}").mkdir(exist_ok=True)
        pickle.dump((t_rois, res_t[v]), open(W9 / f"{tag}{v}" / "test_probs.pkl", "wb"))
        pickle.dump((b_rois, res_b[v]), open(W9 / f"{tag}{v}" / "bench_probs.pkl", "wb"))
    print("written", [str(W9 / f"{tag}{v}") for v in res_t])
