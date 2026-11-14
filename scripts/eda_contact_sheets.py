"""Contact sheets of test images and train image+mask overlays for visual inspection."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def thumb(p, w=300):
    im = Image.open(p).convert("RGB")
    h = int(im.height * w / im.width)
    return im.resize((w, h))


def sheet(items, cols, path, w=300):
    ths = [(thumb(p, w), label) for p, label in items]
    hmax = max(t.height for t, _ in ths)
    rows = (len(ths) + cols - 1) // cols
    S = Image.new("RGB", (cols * w, rows * (hmax + 14)), "black")
    d = ImageDraw.Draw(S)
    for i, (t, label) in enumerate(ths):
        x, y = (i % cols) * w, (i // cols) * (hmax + 14)
        S.paste(t, (x, y + 14))
        d.text((x + 2, y), label, fill="yellow")
    S.save(path, quality=85)


test = sorted((raw / "test_images_v2/test_set_v2").iterdir())
for k in range(0, len(test), 60):
    chunk = test[k:k + 60]
    sheet([(p, f"{p.name} {Image.open(p).size}") for p in chunk], 6, out / f"test_{k:03d}.jpg", w=260)


def overlay(img_p, mask_p, color):
    im = Image.open(img_p).convert("RGB")
    m = Image.open(mask_p)
    arr = np.array(m)
    fg = (arr == 0) if m.mode == "P" else (np.array(m.convert("L")) > 127)
    fg = Image.fromarray((fg * 255).astype(np.uint8)).resize(im.size, Image.NEAREST)
    col = Image.new("RGB", im.size, color)
    return Image.composite(Image.blend(im, col, 0.5), im, fg)


rng = np.random.default_rng(0)
for kind, color in [("apo", (255, 0, 0)), ("fasc", (0, 255, 0))]:
    imgs = sorted((raw / f"{kind}_imgs_v1/{kind}_images_new_model_v1").glob("*.tif"))
    sel = sorted(rng.choice(len(imgs), 24, replace=False))
    tiles = []
    for i in sel:
        ip = imgs[i]
        mp = raw / f"{kind}_masks_v1/{kind}_masks_new_model_v1" / ip.name
        ov = overlay(ip, mp, color)
        tmp = out / f"_tmp_{kind}_{i}.png"
        ov.save(tmp)
        tiles.append((tmp, f"{ip.name} img{Image.open(ip).size} mask{Image.open(mp).size}"))
    sheet(tiles, 4, out / f"train_{kind}_overlay.jpg", w=400)
    for t, _ in tiles:
        t.unlink()
print("done")
