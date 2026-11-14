"""Inventory of the UMUD dataset: folders, image sizes, modes, mask modes, duplicates."""
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

by_dir = defaultdict(list)
for p in root.rglob("*"):
    if p.suffix.lower() in exts:
        by_dir[p.parent.relative_to(root)].append(p)

for d, files in sorted(by_dir.items()):
    files.sort()
    shapes, modes = Counter(), Counter()
    hashes = Counter()
    for p in files:
        with Image.open(p) as im:
            shapes[(im.size[1], im.size[0])] += 1
            modes[im.mode] += 1
        hashes[hashlib.md5(p.read_bytes()).hexdigest()] += 1
    dup = sum(c - 1 for c in hashes.values() if c > 1)
    print(f"\n== {d}  n={len(files)}  exact-dup-files={dup}")
    print("  first/last:", files[0].name, files[-1].name)
    print("  modes:", dict(modes))
    print("  shapes (h,w):", shapes.most_common(12))
