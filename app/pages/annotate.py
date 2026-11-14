"""Manual measurement of the test frames: aponeuroses and fascicles by clicks.

Labels go to labels/<annotator>.json in full-frame pixel coordinates; PA, FL and MT are computed
from them by app/manual.py. scripts/manual_labels.py turns all label files into a table and scores
submissions against it.
"""
import base64
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v2 import component
from PIL import Image

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP.parent / "notebooks"))
from core import ROOT, load_pipeline
from umud_manual import measure_manual

st.set_page_config(page_title="UMUD разметка", layout="wide")
LABELS = ROOT / "labels"

JS = r"""
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  parentElement.querySelectorAll('.ann').forEach(e => e.remove());
  const root = document.createElement('div');
  root.className = 'ann';
  root.innerHTML = `
    <div class="bar">
      <button data-mode="sup">1 · поверхностный</button>
      <button data-mode="deep">2 · глубокий</button>
      <button data-mode="fasc">3 · фасцикула</button>
      <span class="gap"></span>
      <button data-act="undo">отменить (Z)</button>
      <button data-act="clear">очистить</button>
      <button data-act="skip">не размечается</button>
      <button data-act="save" class="save">сохранить (S)</button>
    </div>
    <div class="info"></div>
    <canvas></canvas>`;
  parentElement.appendChild(root);

  const [l, t, r, b] = data.box;
  const drafts = (window.__umudDrafts = window.__umudDrafts || {});
  let ann = drafts[data.key] || JSON.parse(JSON.stringify(data.ann || {}));
  ann.sup = ann.sup || []; ann.deep = ann.deep || []; ann.fasc = ann.fasc || [];
  let pending = null, mode = ann.sup.length < 2 ? 'sup' : ann.deep.length < 2 ? 'deep' : 'fasc';
  const history = [];
  const canvas = root.querySelector('canvas'), ctx = canvas.getContext('2d');
  const info = root.querySelector('.info');
  const img = new Image();
  let scale = 1;

  const fit = pts => {
    if (pts.length < 2) return null;
    const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
    let sxx = 0, sxy = 0;
    pts.forEach(p => { sxx += (p[0] - mx) ** 2; sxy += (p[0] - mx) * (p[1] - my); });
    if (sxx < 1) return null;
    const a = sxy / sxx; return [a, my - a * mx];
  };
  const hit = (x, y, th, ln) => { const d = Math.sin(th) - ln[0] * Math.cos(th); return Math.abs(d) < 1e-9 ? NaN : (ln[0] * x + ln[1] - y) / d; };
  const ang = (t1, t2) => { let d = ((t1 - t2 + Math.PI / 2) % Math.PI + Math.PI) % Math.PI - Math.PI / 2; return Math.abs(d * 180 / Math.PI); };

  function numbers() {
    const sup = fit(ann.sup), deep = fit(ann.deep), pxmm = data.px_mm;
    if (!sup || !deep) return 'Нужно по две и больше точки на каждом апоневрозе.';
    const mt = [0.25, 0.5, 0.75].map(f => { const x = l + f * (r - l); return Math.abs(deep[0] * x + deep[1] - sup[0] * x - sup[1]) / Math.hypot(deep[0], 1); });
    const MT = mt.reduce((s, v) => s + v, 0) / 3 / pxmm;
    const fs = ann.fasc.map(([[x1, y1], [x2, y2]]) => {
      const th = Math.atan2(y2 - y1, x2 - x1), xm = (x1 + x2) / 2, ym = (y1 + y2) / 2;
      return [ang(th, Math.atan(deep[0])), Math.abs(hit(xm, ym, th, sup) - hit(xm, ym, th, deep)) / pxmm];
    });
    if (!fs.length) return `MT ${MT.toFixed(1)} мм · фасцикул пока нет`;
    const PA = fs.reduce((s, f) => s + f[0], 0) / fs.length, FL = fs.reduce((s, f) => s + f[1], 0) / fs.length;
    return `PA ${PA.toFixed(1)}° · FL ${FL.toFixed(1)} мм · MT ${MT.toFixed(1)} мм · фасцикул ${fs.length}`
      + (fs.length < 3 ? ' (лучше 3 и больше)' : '');
  }

  const toCanvas = ([x, y]) => [(x - l) * scale, (y - t) * scale];
  function drawLine(ln, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(0, (ln[0] * l + ln[1] - t) * scale); ctx.lineTo(canvas.width, (ln[0] * r + ln[1] - t) * scale); ctx.stroke();
  }
  function dot(p, color) { const [x, y] = toCanvas(p); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); }

  function draw() {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const sup = fit(ann.sup), deep = fit(ann.deep);
    if (sup) drawLine(sup, '#ff5a5a');
    if (deep) drawLine(deep, '#5aa0ff');
    ann.sup.forEach(p => dot(p, '#ff5a5a'));
    ann.deep.forEach(p => dot(p, '#5aa0ff'));
    ann.fasc.forEach(([p1, p2]) => {
      const th = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]), xm = (p1[0] + p2[0]) / 2, ym = (p1[1] + p2[1]) / 2;
      if (sup && deep) {
        const s1 = hit(xm, ym, th, sup), s2 = hit(xm, ym, th, deep);
        const a = toCanvas([xm + s1 * Math.cos(th), ym + s1 * Math.sin(th)]), c = toCanvas([xm + s2 * Math.cos(th), ym + s2 * Math.sin(th)]);
        ctx.strokeStyle = '#00e6e6'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...c); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.strokeStyle = '#ffd700'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(...toCanvas(p1)); ctx.lineTo(...toCanvas(p2)); ctx.stroke();
    });
    if (pending) dot(pending, '#ffd700');
    root.querySelectorAll('[data-mode]').forEach(bt => bt.classList.toggle('on', bt.dataset.mode === mode));
    info.textContent = numbers();
    drafts[data.key] = ann;
  }

  function resize() {
    const w = Math.min(root.clientWidth || 900, 1400);
    scale = w / (r - l);
    canvas.width = w; canvas.height = Math.round((b - t) * scale);
    draw();
  }

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const p = [l + (e.clientX - rect.left) * (canvas.width / rect.width) / scale,
               t + (e.clientY - rect.top) * (canvas.height / rect.height) / scale];
    if (mode === 'fasc') {
      if (pending) { ann.fasc.push([pending, p]); pending = null; history.push('fasc'); }
      else { pending = p; history.push('pending'); }
    } else { ann[mode].push(p); history.push(mode); }
    draw();
  });

  function act(a) {
    if (a === 'undo') {
      const h = history.pop();
      if (h === 'pending') pending = null;
      else if (h === 'fasc') ann.fasc.pop();
      else if (h) ann[h].pop();
      else if (pending) pending = null;
      else if (ann.fasc.length) ann.fasc.pop();
    } else if (a === 'clear') { ann = { sup: [], deep: [], fasc: [] }; pending = null; history.length = 0; }
    else if (a === 'save') { setTriggerValue('save', { key: data.key, ann: { sup: ann.sup, deep: ann.deep, fasc: ann.fasc } }); }
    else if (a === 'skip') { setTriggerValue('save', { key: data.key, ann: { skip: true } }); }
    draw();
  }
  root.querySelectorAll('[data-mode]').forEach(bt => bt.onclick = () => { mode = bt.dataset.mode; pending = null; draw(); });
  root.querySelectorAll('[data-act]').forEach(bt => bt.onclick = () => act(bt.dataset.act));

  const keys = e => {
    const tag = (e.target && e.target.tagName) || '';
    if (/INPUT|TEXTAREA|SELECT/.test(tag)) return;
    if (e.key === '1') mode = 'sup'; else if (e.key === '2') mode = 'deep'; else if (e.key === '3') mode = 'fasc';
    else if (e.key === 'z' || e.key === 'я') return act('undo');
    else if (e.key === 's' || e.key === 'ы') return act('save');
    else return;
    pending = null; draw();
  };
  document.addEventListener('keydown', keys);
  const ro = new ResizeObserver(resize);
  img.onload = () => { resize(); ro.observe(root); };
  img.src = data.img;
  return () => { document.removeEventListener('keydown', keys); ro.disconnect(); };
}
"""

CSS = """
.ann .bar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
.ann .gap { flex: 1; }
.ann button { padding: 4px 10px; border-radius: 6px; border: 1px solid #888; background: transparent;
              color: var(--st-text-color); cursor: pointer; font-size: 14px; }
.ann button.on { background: var(--st-primary-color); color: white; border-color: var(--st-primary-color); }
.ann button.save { border-color: var(--st-primary-color); font-weight: 600; }
.ann .info { font-size: 15px; margin: 4px 0 8px; min-height: 1.4em; }
.ann canvas { width: 100%; cursor: crosshair; display: block; }
"""

annotator_canvas = component("umud_annotator", js=JS, css=CSS)


@st.cache_resource(show_spinner="Загружаю функции из ноутбука...")
def pipeline():
    return load_pipeline()


@st.cache_data(persist="disk", show_spinner="Читаю шкалы и серии тестовых кадров (один раз)...")
def test_frames() -> pd.DataFrame:
    P = pipeline()
    meta = P["test_meta"]()
    rois = [P["read_gray"](r.path)[r.t:r.b, r.l:r.r] for r in meta.itertuples()]
    meta["series"] = P["frame_groups"](rois)
    meta["series_size"] = meta.groupby("series").image_id.transform("size")
    return meta


ANNOTATORS = {"makhambet": "A", "arman": "B"}   # name typed in the sidebar -> own half of the frames
CHECK_PER_LAYOUT = 3                              # Telemed series labelled by both, to measure agreement


def with_portions(meta: pd.DataFrame) -> pd.DataFrame:
    """Split of the series between two annotators, the same on every machine.

    The first CHECK_PER_LAYOUT series of each Telemed layout (our weakest) are labelled by both
    ("общая"), which shows how much the two annotators disagree. Everything else alternates between
    halves A and B within each layout. `order` interleaves layouts, so a half done only partly still
    covers all of them.
    """
    units = meta.groupby("series").agg(layout=("layout", "first"), first=("image_id", "first")).sort_values("first")
    units["k"] = units.groupby("layout").cumcount()
    check = units.layout.str.startswith("telemed") & (units.k < CHECK_PER_LAYOUT)
    units["k"] = units.k - np.where(units.layout.str.startswith("telemed"), CHECK_PER_LAYOUT, 0)
    units["order"] = np.where(check, -1.0, (units.k + 0.5) / units.groupby("layout").k.transform("size"))
    units["portion"] = np.where(check, "общая", np.where(units.k % 2 == 0, "A", "B"))
    return meta.join(units[["portion", "order"]], on="series")


@st.cache_data(max_entries=8)
def crop_png(path: str, box: tuple) -> str:
    l, t, r, b = box
    im = Image.open(path).convert("L").crop((l, t, r, b))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_labels(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_labels(path: Path, labels: dict):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------- sidebar ----------

st.sidebar.title("Разметка теста")
if not (ROOT / "data/raw/test_images_v2").exists():
    st.error("Нет тестовых кадров в data/raw/test_images_v2, см. README.")
    st.stop()
name = re.sub(r"[^a-z0-9_-]", "", st.sidebar.text_input("Кто размечает (латиницей)", key="annotator").lower())
if not name:
    st.info("Впиши слева своё имя латиницей: метки сохранятся в labels/<имя>.json.")
    st.stop()
path = LABELS / f"{name}.json"
labels = load_labels(path)

meta = with_portions(test_frames())
own = ANNOTATORS.get(name)
if own:
    st.sidebar.success(f"Твоя часть: {own} и {(meta.portion == 'общая').sum() // 5} общих проверочных серий")
    scope = st.sidebar.radio("Показывать", ["моя часть", "все кадры"], horizontal=True)
    portions = ["общая", own] if scope == "моя часть" else None
else:
    st.sidebar.warning(f"Имени нет в списке ({', '.join(ANNOTATORS)}), показываю все кадры")
    portions = None
one_per_series = st.sidebar.checkbox("Один кадр из серии", True,
                                     help="Кадры серии почти одинаковые, метка разойдётся на всю серию")
frames = meta[meta.image_id == meta.groupby("series").image_id.transform(lambda s: s.iloc[len(s) // 2])] \
    if one_per_series else meta
if portions:
    frames = frames[frames.portion.isin(portions)].sort_values(["order", "image_id"])
layouts = st.sidebar.multiselect("Раскладка экрана", sorted(meta.layout.unique()))
if layouts:
    frames = frames[frames.layout.isin(layouts)]
done_ids = set(labels)
only_todo = st.sidebar.checkbox("Только неразмеченные", False)
ids = [i for i in frames.image_id if not (only_todo and i in done_ids)] or frames.image_id.tolist()
n_done = sum(i in done_ids for i in frames.image_id)
st.sidebar.progress(n_done / max(1, len(frames)), f"готово {n_done} из {len(frames)}")
show_model = st.sidebar.checkbox("Показать прогноз модели", False,
                                 help="Лучше не смотреть до разметки, чтобы не подстраиваться под модель")

with st.sidebar.expander("Как размечать"):
    st.markdown(
        "- **1**: 2–4 точки по нижнему краю поверхностного апоневроза (граница с мышцей).\n"
        "- **2**: 2–4 точки по верхнему краю глубокого апоневроза.\n"
        "- **3**: каждая фасцикула двумя кликами вдоль хорошо видного волокна. Хотя бы 3 штуки, "
        "ближе к центру кадра и такие, которые меньше всего надо продлевать. Пунктир показывает продление.\n"
        "- Если в кадре две мышцы друг над другом, меряем верхнюю.\n"
        "- **Z** отменяет последний клик, **S** сохраняет и переходит к следующему кадру.\n"
        "- Кадр, на котором ничего не разобрать, отмечай «не размечается».")

# ---------- choose a frame ----------

if "goto" in st.session_state:
    st.session_state.ann_id = st.session_state.pop("goto")
if st.session_state.get("ann_id") not in ids:
    st.session_state.ann_id = next((i for i in ids if i not in done_ids), ids[0])


def step(delta):
    st.session_state.ann_id = ids[(ids.index(st.session_state.ann_id) + delta) % len(ids)]


c1, c2, c3 = st.columns([1, 6, 1])
c1.button("← назад", on_click=step, args=(-1,))
c3.button("вперёд →", on_click=step, args=(1,))
image_id = c2.selectbox("Кадр", ids, key="ann_id", label_visibility="collapsed",
                        format_func=lambda i: f"{i}  ✓" if i in done_ids else i)
row = meta.set_index("image_id").loc[image_id]
box = tuple(int(v) for v in (row.l, row.t, row.r, row.b))
series_ids = meta[meta.series == row.series].image_id.tolist()
st.subheader(f"{image_id}  ·  {row.layout}  ·  {row.px_per_cm:.1f} px/см"
             + (f"  ·  серия из {len(series_ids)} кадров" if len(series_ids) > 1 else ""))

saved = labels.get(image_id, {})
result = annotator_canvas(
    data=dict(key=image_id, img=crop_png(row.path, box), box=list(box), px_mm=row.px_per_cm / 10,
              ann={k: saved.get(k, []) for k in ("sup", "deep", "fasc")}),
    key=f"canvas_{name}", on_save_change=lambda: None)

if result.save and result.save.get("key") == image_id:
    ann = result.save["ann"]
    labels[image_id] = dict(ann, series=series_ids if one_per_series else [image_id], layout=row.layout,
                            saved=datetime.now().isoformat(timespec="seconds"))
    save_labels(path, labels)
    nxt = [i for i in ids if i not in labels]
    st.session_state.goto = nxt[0] if nxt else image_id
    st.toast(f"{image_id} сохранён в {path.relative_to(ROOT)}")
    st.rerun()

if saved:
    if saved.get("skip"):
        st.caption("Кадр отмечен как неразмечаемый.")
    else:
        m = measure_manual(saved, row.px_per_cm, box)
        st.caption(f"Сохранено {saved.get('saved', '')}: PA {m['pa']:.1f}°, FL {m['fl']:.1f} мм, MT {m['mt']:.1f} мм, "
                   f"фасцикул {m['n_fasc']}" + (f"; метка действует на {len(saved['series'])} кадров серии"
                                                  if len(saved.get("series", [])) > 1 else ""))
if show_model:
    best = ROOT / "work/sub_v3_5fold_pa2.csv"
    if best.exists():
        s = pd.read_csv(best).set_index("image_id").loc[image_id]
        st.caption(f"Модель (v3): PA {s.pa_deg:.1f}°, FL {s.fl_mm:.1f} мм, MT {s.mt_mm:.1f} мм")
