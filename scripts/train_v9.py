"""v9 networks: aspect-preserving input, fascicle direction head, second encoder for fascicles.

Usage: python scripts/train_v9.py [run ...]     runs: fasc_cnx, apo_cnx, fasc_eff (default: all, in order)
UMUD_SMOKE=1 trains 1 epoch on a few images to check the code.
Weights go to work/v9/<run>_fold{k}.pt; a fold whose weights exist is skipped. Trained on the training
set only, with the same GroupKFold folds as the notebook.
"""
import hashlib
import io
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import load_pipeline
from umud_aspect import AspectDataset, scale_short

P = load_pipeline()
CFG, DEVICE = P["CFG"], P["DEVICE"]
SMOKE = os.environ.get("UMUD_SMOKE") == "1"
OUT = ROOT / ("work/v9_smoke" if SMOKE else "work/v9")
OUT.mkdir(parents=True, exist_ok=True)
S = 512
RUNS = {  # task, encoder, classes (masks), orientation head, epochs, batch
    "fasc_cnx": dict(task="fasc", encoder="tu-convnext_tiny", n_mask=1, orient=True, epochs=20, batch=8),
    "apo_cnx": dict(task="apo", encoder="tu-convnext_tiny", n_mask=2, orient=False, epochs=20, batch=8),
    "fasc_eff": dict(task="fasc", encoder="efficientnet-b4", n_mask=1, orient=True, epochs=12, batch=6, folds=3),
}
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(CFG["vram_fraction"])


def build_aspect_cache(task: str):
    """Like build_cache in the notebook, but frames keep their aspect ratio (shorter side = S)."""
    path = ROOT / f"work/v9/cache_{task}_aspect{S}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pickle.load(open(path, "rb"))
    DATA = P["DATA"]
    img_dir = DATA / f"{task}_imgs_v1/{task}_images_new_model_v1"
    msk_dir = DATA / f"{task}_masks_v1/{task}_masks_new_model_v1"
    xs, ys, seen = [], [], set()
    for p in sorted(img_dir.glob("*.tif")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        g, m = P["read_gray"](p), P["read_mask"](msk_dir / p.name)
        if g.shape != m.shape:
            if m.shape == (864, 1152):
                m = cv2.resize(m.astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            else:
                g = cv2.resize(g, m.shape[::-1], interpolation=cv2.INTER_AREA)
        l, t, r, b = P["find_roi"](g)
        g, m = g[t:b, l:r], m[t:b, l:r]
        target = P["split_apo"](m) if task == "apo" else m[None]
        thr = 0.5 if task == "apo" else 0.1
        x = scale_short(g, S)
        xs.append(x)
        ys.append(np.stack([cv2.resize(c.astype(np.float32), x.shape[::-1], interpolation=cv2.INTER_AREA) > thr
                            for c in target]).astype(np.uint8))
    groups = np.array(P["frame_groups"](xs))
    pickle.dump((xs, ys, groups), open(path, "wb"))
    return xs, ys, groups


def make_net(encoder, classes, pretrained=True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=classes)


def losses(logits, yb, n_mask, orient):
    seg_l, seg_y = logits[:, :n_mask], yb[:, :n_mask]
    loss = F.binary_cross_entropy_with_logits(seg_l, seg_y) + P["dice_loss"](seg_l, seg_y)
    if orient:
        o, ot, w = torch.tanh(logits[:, n_mask:n_mask + 2]), yb[:, n_mask:n_mask + 2], yb[:, n_mask + 2:n_mask + 3]
        loss = loss + ((o - ot).abs() * w).sum() / (2 * w.sum() + 1)
    return loss


@torch.no_grad()
def angle_error(logits, yb, n_mask):
    """Mean absolute difference (deg) between predicted and target direction on the target lines."""
    o, ot, w = logits[:, n_mask:n_mask + 2], yb[:, n_mask:n_mask + 2], yb[:, n_mask + 2]
    a = 0.5 * torch.atan2(o[:, 1], o[:, 0])
    b = 0.5 * torch.atan2(ot[:, 1], ot[:, 0])
    d = torch.remainder(a - b + np.pi / 2, np.pi) - np.pi / 2
    return float((d.abs() * w).sum() / (w.sum() + 1) * 180 / np.pi)


def train_run(name, cfg, fold, xs, ys, groups):
    path = OUT / f"{name}_fold{fold}.pt"
    if path.exists():
        print(f"{name} fold{fold}: weights exist, skipping")
        return
    tr, va = list(GroupKFold(CFG["n_folds"]).split(np.arange(len(xs)), groups=groups))[fold]
    if SMOKE:
        tr, va = tr[:24], va[:8]
    n_mask, orient = cfg["n_mask"], cfg["orient"]
    mk = lambda idx, train: AspectDataset(xs, ys, idx, train, S, P["augment"], seed=fold, orient=orient)
    dl_tr = torch.utils.data.DataLoader(mk(tr, True), cfg["batch"], shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(mk(va, False), cfg["batch"])
    model = make_net(cfg["encoder"], n_mask + (2 if orient else 0)).to(DEVICE)
    epochs = 1 if SMOKE else cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=epochs * len(dl_tr), pct_start=0.1)
    best = -1.0
    for ep in range(epochs):
        t0, ls = time.time(), []
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(xb)
            loss = losses(logits.float(), yb, n_mask, orient)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ls.append(loss.item())
        model.eval()
        dices, angs = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    logits = model(xb).float()
                dices.append(P["dice_score"](logits[:, :n_mask], yb[:, :n_mask]))
                if orient:
                    angs.append(angle_error(logits, yb, n_mask))
        dice = np.mean(dices, axis=0)
        # model selection on the masks only; the direction error is reported
        if dice.mean() > best:
            best = dice.mean()
            torch.save(model.state_dict(), path)
        print(f"{name} fold{fold} ep{ep:02d} loss {np.mean(ls):.3f} val dice {np.round(dice, 3)}"
              + (f" angle err {np.mean(angs):.1f} deg" if orient else "")
              + f" {'*' if best == dice.mean() else ' '} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    runs = sys.argv[1:] or list(RUNS)
    caches = {}
    for name in runs:
        cfg = RUNS[name]
        if cfg["task"] not in caches:
            t0 = time.time()
            caches[cfg["task"]] = build_aspect_cache(cfg["task"])
            xs = caches[cfg["task"]][0]
            print(f"cache {cfg['task']}: {len(xs)} frames, sizes {min(x.shape for x in xs)}..{max(x.shape for x in xs)},"
                  f" {time.time() - t0:.0f}s")
        for fold in range(1 if SMOKE else cfg.get("folds", CFG["n_folds"])):
            train_run(name, cfg, fold, *caches[cfg["task"]])
    print("ALL DONE")
