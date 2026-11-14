"""Experiment: fascicle model at 768x768 input (fold 0), saved to work/exp768/.

Hypothesis: at 512 thin, faint fibres near the deep aponeurosis vanish on detailed test frames.
Uses the notebook's own build_cache / train_fold with CFG overrides.
"""
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import load_pipeline

P = load_pipeline()
P["CFG"].update(img_size=768, batch=4, epochs={"apo": 0, "fasc": 20})
P["WORK"] = ROOT / "work/exp768"
P["WORK"].mkdir(exist_ok=True)
if P["DEVICE"] == "cuda":
    torch.cuda.set_per_process_memory_fraction(P["CFG"]["vram_fraction"])
P["cache"] = {"fasc": P["build_cache"]("fasc")}
print("cache ready:", P["cache"]["fasc"][0].shape)
best = P["train_fold"]("fasc", 0)
print("BEST DICE", best)
