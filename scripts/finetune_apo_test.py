"""Fine-tune the aponeurosis U-Net on the test domain, with the manual aponeurosis lines as targets.

Cross-fitted: test series are split into 5 groups; fold k starts from work/apo_fold{k}.pt and is trained
on its own part of the training set plus the test series of the other 4 groups. Every test frame is
then predicted only by the fold that never saw its series, so the local score stays honest.

Targets: for each test frame a band along each manual line, as thick as the band the current network
predicts there, so the masks look like the training masks and the inner-edge convention is kept.

Writes work/ft_apo/apo_fold{k}.pt and work/ft_apo/test_probs.pkl (fine-tuned aponeuroses + the
original fascicle maps), in the format of work/test_probs.pkl.
"""
import io
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from core import load_pipeline

P = load_pipeline()
CFG, DEVICE = P["CFG"], P["DEVICE"]
OUT = ROOT / "work/ft_apo"
OUT.mkdir(exist_ok=True)
EPOCHS, LR, S = 8, 1e-4, CFG["img_size"]
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(CFG["vram_fraction"])


def band_mask(line, shape, half):
    """Rows within `half` px of the line y = a*x + b, for every column."""
    a, b = line
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.abs(yy - (a * xx + b)) <= half


def test_targets(meta, rois, probs):
    """512x512 images and 2-channel band masks from the manual lines, plus each frame's series."""
    xs, ys, ok = [], [], []
    for r in meta.itertuples():
        f = ROOT / f"annotations/{Path(r.image_id).stem}.json"
        if not f.exists():
            continue
        pts = json.loads(f.read_text(encoding="utf-8"))["points"]
        roi, pr = rois[r.image_id], probs[r.image_id].astype(np.float32)
        chans = []
        for c, key in enumerate(["upper_aponeurosis", "lower_aponeurosis"]):
            p = np.array([[q["x_px"] - r.l, q["y_px"] - r.t] for q in pts[key]])
            line = np.polyfit(p[:, 0], p[:, 1], 1)
            near = band_mask(line, roi.shape, 3 * r.px_per_cm / 10)          # +-3 mm around the line
            thick = np.median(((pr[c] > 0.5) & near).sum(axis=0)[((pr[c] > 0.5) & near).any(axis=0)]) \
                if ((pr[c] > 0.5) & near).any() else 0
            half = max(thick, 1.2 * r.px_per_cm / 10) / 2                    # at least 1.2 mm thick
            chans.append(band_mask(line, roi.shape, half))
        xs.append(cv2.resize(roi, (S, S), interpolation=cv2.INTER_AREA))
        ys.append(np.stack([cv2.resize(m.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA) > 0.5
                            for m in chans]))
        ok.append(r.image_id)
    return np.stack(xs), np.stack(ys).astype(np.uint8), ok


def run_epoch(model, dl, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    losses, dices = [], []
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(train), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(xb)
        logits = logits.float()
        if train:
            loss = F.binary_cross_entropy_with_logits(logits, yb) + P["dice_loss"](logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        else:
            dices.append(P["dice_score"](logits, yb))
    return np.mean(losses) if train else np.mean(dices, axis=0)


if __name__ == "__main__":
    meta = P["test_meta"]()
    rois, probs = pickle.load(open(ROOT / "work/test_probs.pkl", "rb"))
    series = np.array(P["frame_groups"]([rois[i] for i in meta.image_id]))
    xt, yt, ids = test_targets(meta, rois, probs)
    s_of = dict(zip(meta.image_id, series))
    tgroup = np.zeros(len(ids), int)
    for k, (_, te) in enumerate(GroupKFold(5).split(ids, groups=[s_of[i] for i in ids])):
        tgroup[te] = k
    print(f"test targets: {len(ids)} frames, band fraction {yt.mean():.3%}")

    x, y, groups = P["build_cache"]("apo")
    X, Y = np.concatenate([x, xt]), np.concatenate([y, yt])
    n_train = len(x)
    folds = list(GroupKFold(CFG["n_folds"]).split(x, groups=groups))
    for k in range(5):
        path = OUT / f"apo_fold{k}.pt"
        if path.exists():
            print(f"fold{k}: уже есть, пропускаю")
            continue
        tr, va = folds[k]
        tr_idx = np.concatenate([tr, n_train + np.flatnonzero(tgroup != k)])
        dl_tr = torch.utils.data.DataLoader(P["SegDataset"](X, Y, tr_idx, True), CFG["batch"], shuffle=True,
                                            drop_last=True)
        dl_va = torch.utils.data.DataLoader(P["SegDataset"](X, Y, va, False), CFG["batch"])
        dl_te = torch.utils.data.DataLoader(P["SegDataset"](X, Y, n_train + np.flatnonzero(tgroup == k), False),
                                            CFG["batch"])
        model = P["make_model"](2, pretrained=False)
        model.load_state_dict(torch.load(ROOT / f"work/apo_fold{k}.pt", map_location="cpu"))
        model.to(DEVICE)
        print(f"fold{k} before: train-val dice {np.round(run_epoch(model, dl_va), 3)}, "
              f"held-out test dice {np.round(run_epoch(model, dl_te), 3)}")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(dl_tr), pct_start=0.15)
        for ep in range(EPOCHS):
            t0 = time.time()
            loss = run_epoch(model, dl_tr, opt, sched)
            print(f"fold{k} ep{ep} loss {loss:.3f} train-val dice {np.round(run_epoch(model, dl_va), 3)} "
                  f"held-out test dice {np.round(run_epoch(model, dl_te), 3)} {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), path)

    # each test frame by the fold that never saw its series; fascicle maps stay as they were
    new = {}
    for k in range(5):
        m = P["make_model"](2, pretrained=False)
        m.load_state_dict(torch.load(OUT / f"apo_fold{k}.pt", map_location="cpu"))
        m.to(DEVICE).eval()
        for i in np.flatnonzero(tgroup == k):
            iid = ids[i]
            apo = P["predict_probs"]([m], rois[iid])
            new[iid] = np.concatenate([apo, probs[iid][2:].astype(np.float32)]).astype(np.float16)
    for iid in probs:
        new.setdefault(iid, probs[iid])
    pickle.dump((rois, new), open(OUT / "test_probs.pkl", "wb"))
    print("written", OUT / "test_probs.pkl")
