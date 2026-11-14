"""Audit of the worst Telemed series: our fragments and aponeuroses against the annotator's lines.

For every frame the PA error (raw pa_seg against the annotator's PA) is split into
  apo   - our deep line against his deep line (same fragments, two reference lines);
  pick  - all our fragments against the fragments lying along his fascicle (which fascicle is measured);
  map   - fragments along his fascicle against his fascicle itself (orientation of the map there).
FL of the submission is split the same way: FL on his lines with our PA (angle part) against his FL.
Writes work/diag/audit_series.csv and overlays of the central frame of each series to work/diag/audit_*.png.

Usage: python scripts/audit_series.py [n_series]   (default 20; maps from work/v9/b_mix, as in LB 0.383)
"""
import io
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: plt.close("all")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import cv2
import nbformat
import numpy as np
import pandas as pd
from umud_scale import decode_scale, layout_name

# Only the data and geometry cells of the notebook: no torch, no model loading.
nb = nbformat.read(ROOT / "notebooks/umud_pipeline.ipynb", as_version=4)
src = [c.source for c in nb.cells if c.cell_type == "code"]
g = dict(np=np, pd=pd, cv2=cv2, plt=plt, Path=Path, ROOT=ROOT, DATA=ROOT / "data/raw", json=json,
         decode_scale=decode_scale, layout_name=layout_name, orientation_pa=lambda *a, **k: np.nan,
         Image=__import__("PIL.Image").Image, CFG=dict(px_per_mm=8.0, pa_method="segments", h_over_v={}))
for i in (1, 2, 8, 14):        # reading + ROI, test meta, geometry (section 7), ray length (section 10)
    try:
        exec(src[i], g)
    except Exception:          # the demo part at the end of a cell; the functions above it are defined
        pass
meta = g["meta"].copy()
meta["image_id"] = meta.image_id.str.split("\\").str[-1]
groups = pd.read_csv(ROOT / "eda/test_meta.csv", usecols=["image_id", "group", "group_size"])
groups["image_id"] = groups.image_id.str.split("\\").str[-1]
meta = meta.drop(columns=[c for c in ("group", "group_size") if c in meta]).merge(groups, on="image_id")
K = g["CFG"]["px_per_mm"]
ab, wm, ray = g["angle_between"], g["weighted_median"], g["fascicle_ray_length"]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SHIFT = {"pa": 0.75, "fl": 1.04, "mt": 0.97}
lab = pd.read_csv(ROOT / "labels/manual_arman.csv").set_index("image_id")
sub = pd.read_csv(ROOT / "work/v9/sub_b_mix_segments.csv").set_index("image_id")


def frame_error(r):
    p, q = sub.loc[r.image_id], lab.loc[r.image_id]
    return (abs(p.pa_deg - q.pa - SHIFT["pa"]) / 6 + abs(p.fl_mm - q.fl * SHIFT["fl"]) / 12
            + abs(p.mt_mm - q.mt * SHIFT["mt"]) / 3) / 3


meta["err"] = [frame_error(r) for r in meta.itertuples()]
series = (meta[(meta.group_size > 1) & meta.layout.str.startswith("telemed")]
          .groupby("group").agg(err=("err", "sum"), layout=("layout", "first"), first=("image_id", "first"))
          .sort_values("err", ascending=False).head(N))
share = series.err / meta.err.sum()
print(f"{N} worst Telemed series: {share.sum():.0%} of the total error")

rois, probs = pickle.load(open(ROOT / "work/v9/b_mix/test_probs.pkl", "rb"))


def to_metric_pts(pts, r):
    s = K / (r.px_per_cm / 10)
    return np.array([[(p["x_px"] - r.l) * s, (p["y_px"] - r.t) * s] for p in pts])


def line_of(p):
    a, b = np.polyfit(p[:, 0], p[:, 1], 1)
    return a, b


def frag_line_dist(s, f):
    """Perpendicular distance (px) of a fragment centre to the annotator's fascicle line, and its
    position along that line relative to the clicked segment (0..1 inside)."""
    (x1, y1), (x2, y2) = f
    d = np.array([x2 - x1, y2 - y1]); L = np.hypot(*d); u = d / L
    v = np.array([s[2] - x1, s[3] - y1])
    return abs(u[0] * v[1] - u[1] * v[0]), (u @ v) / L


rows, panels = [], []
for gid, srow in series.iterrows():
    frames = meta[meta.group == gid]
    for j, r in enumerate(frames.itertuples()):
        ann = json.loads((ROOT / f"annotations/{r.image_id.split('.')[0]}.json").read_text(encoding="utf-8"))["points"]
        res, d = g["measure"](rois[r.image_id], probs[r.image_id].astype(np.float32), r.px_per_cm / 10, 1.0,
                              return_debug=True)
        if d is None or len(d["segs"]) < 3:
            continue
        hs, hd, hf = (to_metric_pts(ann[k], r) for k in ("upper_aponeurosis", "lower_aponeurosis", "fascicle"))
        his_sup, his_deep = line_of(hs), line_of(hd)
        t_hf = np.arctan2(hf[1, 1] - hf[0, 1], hf[1, 0] - hf[0, 0])
        t_hd, t_od = np.arctan(his_deep[0]), np.arctan(d["deep"][0])
        segs = d["segs"]                                    # (pa, length, x, y, t)
        W = d["gm"].shape[1]
        pa_his = ab(t_hf, t_hd)
        pa_ours = res["pa_seg"]
        pa_ours_hisdeep = wm([ab(s[4], t_hd) for s in segs], [s[1] for s in segs])
        near = [s for s in segs if (lambda dd: dd[0] < 3 * K and -0.25 < dd[1] < 1.25)(frag_line_dist(s, hf))]
        pa_near = wm([ab(s[4], t_hd) for s in near], [s[1] for s in near]) if len(near) >= 2 else np.nan
        p = sub.loc[r.image_id]
        side = np.sign(((t_hf - t_hd + np.pi / 2) % np.pi) - np.pi / 2)
        fl_his_lines_our_pa = ray(his_sup, his_deep, t_hd + side * np.radians(p.pa_deg - SHIFT["pa"]), W) / K
        fl_our_lines_his_pa = ray(d["sup"], d["deep"], t_od + side * np.radians(pa_his), W) / K
        q = lab.loc[r.image_id]
        hx = np.array([0, W])
        rows.append(dict(
            group=gid, image_id=r.image_id, layout=r.layout,
            pa_sub=p.pa_deg, pa_lab=q.pa + SHIFT["pa"], fl_sub=p.fl_mm, fl_lab=q.fl * SHIFT["fl"],
            mt_sub=p.mt_mm, mt_lab=q.mt * SHIFT["mt"],
            pa_raw_err=pa_ours - pa_his, apo=pa_ours - pa_ours_hisdeep,
            pick=pa_ours_hisdeep - pa_near, map=pa_near - pa_his, n_near=len(near), n_segs=len(segs),
            deep_angle_diff=np.degrees(t_od - t_hd), sup_angle_diff=np.degrees(np.arctan(d["sup"][0]) - np.arctan(his_sup[0])),
            deep_offset_mm=((d["deep"][0] - his_deep[0]) * W / 2 + d["deep"][1] - his_deep[1]) / K,
            sup_offset_mm=((d["sup"][0] - his_sup[0]) * W / 2 + d["sup"][1] - his_sup[1]) / K,
            his_x=hf[:, 0].mean() / W, his_len_mm=np.hypot(*(hf[1] - hf[0])) / K,
            his_depth=(hf[:, 1].mean() - (his_sup[0] * hf[:, 0].mean() + his_sup[1]))
                      / ((his_deep[0] - his_sup[0]) * hf[:, 0].mean() + his_deep[1] - his_sup[1]),
            fl_his_ray=ray(his_sup, his_deep, t_hd + side * np.radians(pa_his), W) / K,
            fl_his_lines_our_pa=fl_his_lines_our_pa, fl_our_lines_his_pa=fl_our_lines_his_pa,
            converge_ours=np.degrees(t_od - np.arctan(d["sup"][0])), converge_his=np.degrees(t_hd - np.arctan(his_sup[0])),
            width_mm=W / K, frac_frag_upper=np.mean([(s[3] - (d["sup"][0] * s[2] + d["sup"][1]))
                                                    / max(1, (d["deep"][0] - d["sup"][0]) * s[2] + d["deep"][1] - d["sup"][1]) < 0.5
                                                    for s in segs])))
        if j == len(frames) // 2:
            panels.append((gid, srow, r, d, res, hs, hd, hf, his_sup, his_deep, near, rows[-1]))

df = pd.DataFrame(rows)
out = ROOT / "work/diag"
out.mkdir(exist_ok=True)
df.to_csv(out / "audit_series.csv", index=False)

# overlays: 4 per sheet, the central frame of every series
for sheet in range(0, len(panels), 4):
    fig, axes = plt.subplots(2, 2, figsize=(22, 15))
    for ax, (gid, srow, r, d, res, hs, hd, hf, his_sup, his_deep, near, row) in zip(axes.ravel(), panels[sheet:sheet + 4]):
        gm, pm, H, W = d["gm"], d["pm"], *d["gm"].shape
        ax.imshow(gm, cmap="gray")
        ax.imshow(np.ma.masked_less(pm[2], 0.3), cmap="summer", alpha=0.35)
        xs = np.array([-0.3 * W, 1.3 * W])
        for (a, b), c, ls in ((d["sup"], "#ff4040", "-"), (d["deep"], "#ff4040", "-"),
                              (his_sup, "#40e0ff", "--"), (his_deep, "#40e0ff", "--")):
            ax.plot(xs, a * xs + b, color=c, lw=1.4, ls=ls)
        pa_his = res["pa_seg"] - row["pa_raw_err"]
        for pa_s, L, x, y, t in d["segs"]:
            dev = pa_s - pa_his
            col = plt.cm.coolwarm(np.clip(0.5 + dev / 10, 0, 1))
            ax.plot([x - L / 2 * np.cos(t), x + L / 2 * np.cos(t)], [y - L / 2 * np.sin(t), y + L / 2 * np.sin(t)],
                    color=col, lw=2.2 if any(s[2] == x and s[3] == y for s in near) else 1.2)
        ax.plot(hf[:, 0], hf[:, 1], color="#ffd000", lw=3)
        tf = np.arctan2(hf[1, 1] - hf[0, 1], hf[1, 0] - hf[0, 0])
        for (a, b) in (his_sup, his_deep):      # his fascicle extended to his lines
            s = g["line_hit"](hf[0, 0], hf[0, 1], tf, (a, b))
            ax.plot([hf[0, 0], hf[0, 0] + s * np.cos(tf)], [hf[0, 1], hf[0, 1] + s * np.sin(tf)], color="#ffd000", lw=1, ls=":")
        ax.set_xlim(-0.05 * W, 1.05 * W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"серия {gid} ({srow['first']}), {r.layout}, доля ошибки {srow.err / meta.err.sum():.1%}\n"
                     f"PA наш {row['pa_sub']:.1f} / Arman {row['pa_lab']:.1f}   FL {row['fl_sub']:.0f} / {row['fl_lab']:.0f}   "
                     f"MT {row['mt_sub']:.1f} / {row['mt_lab']:.1f}\n"
                     f"PA сырой−его {row['pa_raw_err']:+.1f}° = апоневроз {row['apo']:+.1f} + выбор {row['pick']:+.1f} "
                     f"+ карта {row['map']:+.1f}   (рядом с его фасцикулой {row['n_near']} фрагм.)", fontsize=10)
    for ax in axes.ravel()[len(panels[sheet:sheet + 4]):]:
        ax.axis("off")
    fig.suptitle("красное — наши апоневрозы, голубой пунктир — Arman, жёлтое — его фасцикула; фрагменты: синий = положе его, "
                 "красный = круче, толстые — рядом с его фасцикулой", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"audit_{sheet // 4 + 1}.png", dpi=80)
    plt.close(fig)

s = df.groupby("group")[["pa_raw_err", "apo", "pick", "map", "deep_angle_diff", "sup_angle_diff", "deep_offset_mm",
                         "sup_offset_mm", "his_x", "his_depth", "his_len_mm", "n_near", "fl_sub", "fl_lab",
                         "fl_his_lines_our_pa", "fl_our_lines_his_pa", "fl_his_ray", "converge_ours", "converge_his"]].median()
s = s.loc[series.index]
s.insert(0, "share", share.values)
print(s.round(2).to_string())
