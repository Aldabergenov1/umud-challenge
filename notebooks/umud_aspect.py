"""Aspect-preserving input and a fascicle-orientation head for the U-Nets.

The first version squeezed every ultrasound box to 512x512. Boxes differ in shape between scanners, so
a 15-degree fascicle reached the network at 12 to 19 degrees depending on the device. Here a frame is
scaled so that its shorter side is S, the network trains on random S x S crops and predicts the whole
frame with overlapping tiles. Angles are the same as in the original image.

The fascicle network also predicts the local fascicle direction as (cos 2t, sin 2t), learned from the
training masks: the direction of the mask lines at every fascicle pixel, after augmentation.
"""
import math

import cv2
import numpy as np
import torch


def scale_short(g: np.ndarray, S: int, interp=cv2.INTER_AREA) -> np.ndarray:
    """Resize so that the shorter side is S, keeping the aspect ratio."""
    h, w = g.shape[:2]
    s = S / min(h, w)
    return cv2.resize(g, (max(S, round(w * s)), max(S, round(h * s))), interpolation=interp)


def orientation_targets(mask: np.ndarray):
    """(cos 2t, sin 2t) of the line direction t at each pixel, and a weight that is 1 on and next to the
    lines. The direction comes from the structure tensor of the blurred mask."""
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(m, cv2.CV_32F, 1, 0), cv2.Sobel(m, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), 2.0) for v in (gx * gx, gy * gy, gx * gy))
    t = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    w = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    return np.stack([np.cos(2 * t), np.sin(2 * t)]).astype(np.float32), w


class AspectDataset(torch.utils.data.Dataset):
    """Random S x S crops (centre crops for validation) of aspect-preserved frames.

    Targets: the masks, plus for `orient=True` two direction channels and their weight.
    """

    def __init__(self, xs, ys, idx, train: bool, S: int, augment, seed=0, orient=False):
        self.xs, self.ys, self.idx, self.train, self.S = xs, ys, idx, train, S
        self.augment, self.orient = augment, orient
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.xs[self.idx[i]], self.ys[self.idx[i]]
        H, W = x.shape
        S = self.S
        if self.train:
            r0, c0 = self.rng.integers(0, H - S + 1), self.rng.integers(0, W - S + 1)
        else:
            r0, c0 = (H - S) // 2, (W - S) // 2
        x, y = x[r0:r0 + S, c0:c0 + S], y[:, r0:r0 + S, c0:c0 + S]
        if self.train:
            x, y = self.augment(x, y, self.rng)
        else:
            x = x.astype(np.float32) / 255
        y = np.ascontiguousarray(y).astype(np.float32)
        if self.orient:
            o, w = orientation_targets(y[0])
            y = np.concatenate([y, o, w[None]])
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(y)


def tile_starts(L: int, S: int, stride: int):
    if L <= S:
        return [0]
    n = math.ceil((L - S) / stride) + 1
    return sorted({round(v) for v in np.linspace(0, L - S, n)})


@torch.no_grad()
def predict_tiles(models, g: np.ndarray, S: int, device: str, n_mask: int, orient=False, stride=384):
    """Probability maps at the size of `g`: n_mask sigmoid channels, then (cos 2t, sin 2t) if orient.

    Each model predicts every tile and its horizontal mirror; mirroring flips the sign of sin 2t.
    """
    x = scale_short(g, S).astype(np.float32) / 255
    H, W = x.shape
    n_out = n_mask + (2 if orient else 0)
    acc, cnt = np.zeros((n_out, H, W), np.float32), np.zeros((H, W), np.float32)
    tiles = [(r, c) for r in tile_starts(H, S, stride) for c in tile_starts(W, S, stride)]
    batch = torch.from_numpy(np.stack([x[r:r + S, c:c + S] for r, c in tiles]))[:, None].to(device)
    out = 0
    for m in models:
        for flip in (False, True):
            xb = batch.flip(-1) if flip else batch
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                o = m(xb).float()
            if flip:
                o = o.flip(-1)
                if orient:
                    o[:, n_mask + 1] = -o[:, n_mask + 1]
            o = torch.cat([torch.sigmoid(o[:, :n_mask]), torch.tanh(o[:, n_mask:n_out])], 1)
            out = out + o
    out = (out / (2 * len(models))).cpu().numpy()
    for (r, c), o in zip(tiles, out):
        acc[:, r:r + S, c:c + S] += o
        cnt[r:r + S, c:c + S] += 1
    acc /= cnt
    return np.stack([cv2.resize(ch, g.shape[::-1], interpolation=cv2.INTER_LINEAR) for ch in acc])


def orientation_pa(pm: np.ndarray, band: np.ndarray, t_deep: float, weight_pow=2.0):
    """PA from the predicted direction field inside `band`: the mean doubled-angle vector, weighted by
    fascicle probability and by how sure the field is (vector length). Channels: 2 = fascicle
    probability, 3 = cos 2t, 4 = sin 2t."""
    p, c2, s2 = pm[2], pm[3], pm[4]
    conf = np.sqrt(c2 ** 2 + s2 ** 2)
    w = (p ** weight_pow) * conf * band
    if w.sum() < 1e-3:
        return np.nan
    t = 0.5 * np.arctan2((w * s2).sum(), (w * c2).sum())
    d = (t - t_deep + np.pi / 2) % np.pi - np.pi / 2
    return abs(np.degrees(d))
