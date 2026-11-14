"""Rotation test-time augmentation for the fascicle network.

Each test frame is rotated by a few degrees, the 5 fascicle folds (with their usual horizontal-flip
TTA) predict it, the map is rotated back, and the maps are averaged. Fascicle orientation is the
weakest part of the pipeline; averaging over small rotations removes orientation-dependent quirks of
the network. Aponeurosis maps are taken as they are from the input file.

Usage: python scripts/rotation_tta.py <in test_probs.pkl> <out test_probs.pkl> [angles, e.g. -4,0,4]
"""
import io
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import load_pipeline

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
angles = [float(a) for a in (sys.argv[3] if len(sys.argv) > 3 else "-4,0,4").split(",")]
P = load_pipeline()
torch.set_grad_enabled(False)
if P["DEVICE"] == "cuda":
    torch.cuda.set_per_process_memory_fraction(P["CFG"]["vram_fraction"])
models = P["load_models"]("fasc")
rois, probs = pickle.load(open(src, "rb"))
out = {}
for n, (iid, roi) in enumerate(rois.items()):
    H, W = roi.shape
    acc = np.zeros((H, W), np.float32)
    for a in angles:
        R = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
        rot = cv2.warpAffine(roi, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        p = P["predict_probs"](models, rot)[0]
        acc += cv2.warpAffine(p, cv2.invertAffineTransform(R), (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    out[iid] = np.concatenate([probs[iid][:2].astype(np.float32), (acc / len(angles))[None]]).astype(np.float16)
    if n % 50 == 0:
        print(n, iid)
dst.parent.mkdir(exist_ok=True, parents=True)
pickle.dump((rois, out), open(dst, "wb"))
print("written", dst)
