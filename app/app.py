"""UMUD viewer: what the models found on a frame and which numbers they produced.

Run from the repository root:  streamlit run app/app.py
"""
import io
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import available_folds, load_calibration, load_pipeline

st.set_page_config(page_title="UMUD viewer", layout="wide")

COLORS = {"sup": (255, 70, 70), "deep": (70, 150, 255), "fasc": (60, 220, 60), "segs": (255, 215, 0),
          "mt": (255, 80, 255), "fl": (0, 230, 230)}


# ---------- loading ----------

@st.cache_resource(show_spinner="Загружаю конвейер из ноутбука...")
def pipeline():
    return load_pipeline()


P = pipeline()
WORK, CFG = P["WORK"], P["CFG"]


@st.cache_resource(show_spinner="Загружаю веса моделей...")
def models(folds: tuple):
    torch.set_grad_enabled(False)
    out = {}
    for task, n in (("apo", 2), ("fasc", 1)):
        out[task] = []
        for k in folds:
            m = P["make_model"](n, pretrained=False)
            m.load_state_dict(torch.load(WORK / f"{task}_fold{k}.pt", map_location="cpu"))
            out[task].append(m.to(P["DEVICE"]).eval())
    return out


@st.cache_data(persist="disk", show_spinner="Читаю шкалы 309 тестовых кадров (один раз)...")
def test_table() -> pd.DataFrame:
    return P["test_meta"]()


@st.cache_resource
def cached_test_probs():
    """Probability maps saved by the last full notebook run (all folds), if any."""
    path = WORK / "test_probs.pkl"
    return pickle.load(open(path, "rb"))[1] if path.exists() else {}


@st.cache_data(max_entries=64, show_spinner="Модели размечают кадр...")
def infer(key: str, _roi: np.ndarray, folds: tuple) -> np.ndarray:
    """`key` identifies the frame; the leading underscore keeps Streamlit from hashing the array."""
    m = models(folds)
    return np.concatenate([P["predict_probs"](m["apo"], _roi), P["predict_probs"](m["fasc"], _roi)]).astype(np.float16)


@st.cache_data
def submission_files() -> list:
    return sorted(p.name for p in WORK.glob("*.csv"))


# ---------- drawing ----------

def blend(img, mask, color, alpha=0.45):
    img[mask] = (img[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def draw_measurement(dbg: dict, res: dict, layers: set, zoom: int = 2) -> np.ndarray:
    """Overlay on the metric-grid image (8 px per mm), enlarged `zoom` times."""
    k = CFG["px_per_mm"] * zoom
    gm, pm = dbg["gm"], dbg["pm"]
    img = cv2.cvtColor(cv2.resize(gm, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)
    pm = np.stack([cv2.resize(p, img.shape[1::-1]) for p in pm])
    H, W = img.shape[:2]
    if "masks" in layers:
        blend(img, pm[0] > 0.5, COLORS["sup"])
        blend(img, pm[1] > 0.5, COLORS["deep"])
        blend(img, pm[2] > 0.3, COLORS["fasc"], 0.6)
    (a1, b1), (a2, b2) = dbg["sup"], dbg["deep"]
    b1, b2 = b1 * zoom, b2 * zoom          # slopes are unchanged by uniform zoom
    line = lambda a, b, x: int(round(a * x + b))
    if "lines" in layers:
        cv2.line(img, (0, line(a1, b1, 0)), (W, line(a1, b1, W)), COLORS["sup"], 2)
        cv2.line(img, (0, line(a2, b2, 0)), (W, line(a2, b2, W)), COLORS["deep"], 2)
    if "mt" in layers:
        for x in dbg["xs_mt"] * zoom:
            x = int(x)
            cv2.line(img, (x, line(a1, b1, x)), (x, line(a2, b2, x)), COLORS["mt"], 2)
    if "segs" in layers:
        for _, length, x, y, t in dbg.get("segs", []):
            dx, dy = np.cos(t) * length * zoom / 2, np.sin(t) * length * zoom / 2
            cv2.line(img, (int(x * zoom - dx), int(y * zoom - dy)), (int(x * zoom + dx), int(y * zoom + dy)),
                     COLORS["segs"], 2)
    if "fl" in layers and np.isfinite(res["pa"]):
        # the fascicle that PA and FL describe: from the deep aponeurosis at the centre, at angle PA to it
        t_deep = np.arctan(a2)
        side = 1.0 if ((dbg["t_band"] - t_deep + np.pi / 2) % np.pi - np.pi / 2) > 0 else -1.0
        th = t_deep + side * np.radians(res["pa"])
        d = -1 if np.sin(th) > 0 else 1                     # walk towards the superficial aponeurosis
        ux, uy = d * np.cos(th), d * np.sin(th)

        def to_superficial(x0):
            y0 = a2 * x0 + b2
            s = (a1 * x0 + b1 - y0) / (uy - a1 * ux)
            return y0, x0 + s * ux, y0 + s * uy

        _, x1, _ = to_superficial(W / 2)
        x0 = W / 2 - (x1 - W / 2) / 2                       # shift so the fascicle is centred in the frame
        y0, x1, y1 = to_superficial(x0)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), COLORS["fl"], 2)
    # 10 mm scale bar
    cv2.line(img, (15, H - 15), (15 + int(10 * k), H - 15), (255, 255, 255), 3)
    cv2.putText(img, "10 mm", (15, H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------- sidebar ----------

st.sidebar.title("UMUD viewer")
folds_all = available_folds(WORK)
if not folds_all:
    st.error(f"В {WORK} нет весов моделей (apo_fold*.pt, fasc_fold*.pt). "
             "Их можно обучить ноутбуком или положить готовые, см. app/README.md.")
    st.stop()

source = st.sidebar.radio("Кадр", ["Тестовый набор", "Свой файл"], horizontal=True)
folds = tuple(st.sidebar.multiselect("Фолды в ансамбле", folds_all, default=folds_all,
                                     help="Меньше фолдов быстрее на CPU, но шумнее") or folds_all[:1])
st.sidebar.caption(f"Устройство: {P['DEVICE']}")

st.sidebar.subheader("Настройки геометрии")
CFG["pa_method"] = st.sidebar.radio("PA", ["segments", "tensor"], horizontal=True,
                                    help="segments: по фрагментам фасцикул; tensor: средняя ориентация у глубокого апоневроза")
CFG["pa_offset"] = st.sidebar.slider("Сдвиг PA, °", -2.0, 4.0, float(CFG["pa_offset"]), 0.1)
CFG["fl_scale"] = st.sidebar.slider("Множитель FL", 0.8, 1.2, float(CFG["fl_scale"]), 0.01)
telemed_h = st.sidebar.radio("Telemed со шкалой слева: h/v", [1.0, 0.905], horizontal=True,
                             help="1.0: пиксели квадратные (как в ImageJ). 0.905: поправка по нижним рискам")

layers = set(st.sidebar.multiselect(
    "Слои", ["masks", "lines", "mt", "segs", "fl"], default=["masks", "lines", "mt", "segs", "fl"],
    format_func={"masks": "маски", "lines": "апоневрозы", "mt": "замеры MT", "segs": "фрагменты для PA",
                 "fl": "фасцикула (PA, FL)"}.get))

# ---------- choose a frame ----------

frame = None
if source == "Тестовый набор":
    if not (P["DATA"] / "test_images_v2").exists():
        st.error(f"Нет тестовых кадров в {P['DATA']}. Скачай данные соревнования, см. README.")
        st.stop()
    meta = test_table()
    layouts = st.sidebar.multiselect("Раскладка экрана", sorted(meta.layout.unique()))
    ids = meta[meta.layout.isin(layouts)].image_id.tolist() if layouts else meta.image_id.tolist()
    if st.session_state.get("image_id") not in ids:
        st.session_state.image_id = ids[0]

    def step(delta):
        st.session_state.image_id = ids[(ids.index(st.session_state.image_id) + delta) % len(ids)]

    c1, c2, c3 = st.columns([1, 6, 1])
    c1.button("← назад", on_click=step, args=(-1,))
    c3.button("вперёд →", on_click=step, args=(1,))
    image_id = c2.selectbox("Кадр", ids, key="image_id", label_visibility="collapsed")
    row = meta.set_index("image_id").loc[image_id]
    img = Image.open(row.path)
    frame = dict(key=image_id, gray=np.asarray(img.convert("L")), rgb=np.asarray(img.convert("RGB")),
                 box=(row.l, row.t, row.r, row.b), px_per_cm=row.px_per_cm, layout=row.layout)
else:
    up = st.sidebar.file_uploader("Снимок УЗИ (tif, png, jpg)", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up is None:
        st.info("Загрузи кадр слева. Если это скриншот одного из аппаратов теста, масштаб найдётся по шкале; "
                "иначе укажи его вручную.")
        st.stop()
    img = Image.open(io.BytesIO(up.getvalue()))
    gray, ext = np.asarray(img.convert("L")), Path(up.name).suffix[1:].lower()
    try:
        px_cm, box = P["decode_scale"](np.asarray(img), ext)
        layout = P["layout_name"](np.asarray(img), ext)
    except Exception:
        box, layout = P["find_roi"](gray), "unknown"
        px_cm = st.sidebar.number_input("Масштаб, px/см (шкала не распознана)", 20.0, 400.0, 100.0)
    frame = dict(key=up.name + str(len(up.getvalue())), gray=gray, rgb=np.asarray(img.convert("RGB")),
                 box=box, px_per_cm=px_cm, layout=layout)

# ---------- inference and measurement ----------

l, t, r_, b = (int(v) for v in frame["box"])
roi = np.ascontiguousarray(frame["gray"][t:b, l:r_])        # same grayscale conversion as in training
cached = cached_test_probs()
use_cache = source == "Тестовый набор" and frame["key"] in cached and folds == tuple(folds_all)
probs = (cached[frame["key"]] if use_cache else infer(frame["key"], roi, folds)).astype(np.float32)

h_over_v = telemed_h if frame["layout"] == "telemed_left_ruler" else 1.0
res, dbg = P["measure"](roi, probs, frame["px_per_cm"] / 10, h_over_v, return_debug=True)
calib = load_calibration(WORK)
final = P["apply_calibration"](pd.DataFrame([res]), calib).iloc[0]

# ---------- page ----------

st.subheader(f"{frame['key']}  ·  {frame['layout']}  ·  {frame['px_per_cm']:.1f} px/см"
             + ("  ·  маски из кеша последнего прогона" if use_cache else ""))
m1, m2, m3 = st.columns(3)
m1.metric("PA", f"{final.pa:.1f}°", f"сырой {res['pa']:.1f}°", delta_color="off", delta_arrow="off")
m2.metric("FL", f"{final.fl:.1f} мм", f"MT/sin PA {res['fl_trig']:.1f} · прямая {res['fl_line']:.1f}", delta_color="off", delta_arrow="off")
m3.metric("MT", f"{final.mt:.1f} мм", f"сырой {res['mt']:.1f} мм", delta_color="off", delta_arrow="off")

left, right = st.columns([3, 2])
if dbg is None:
    left.warning("Геометрия не сошлась: не нашлись оба апоневроза.")
else:
    left.image(draw_measurement(dbg, res, layers),
               caption="Кадр в масштабе 1 мм = 16 px. Красный: поверхностный апоневроз, синий: глубокий, "
                       "зелёный: карта фасцикул, жёлтый: фрагменты, по которым считается PA, "
                       "розовый: три замера MT, бирюзовый: фасцикула с итоговым PA",
               width="stretch")

full_rgb = frame["rgb"].copy()
cv2.rectangle(full_rgb, (l, t), (r_, b), (255, 255, 0), 3)
right.image(full_rgb, caption="Исходный кадр, жёлтым — найденная УЗ-область", width="stretch")

with right.expander("Все промежуточные числа"):
    st.dataframe(pd.DataFrame({
        "значение": [res["pa_seg"], res["pa_tensor"], res["n_segs"], res["mt"], res["fl_trig"], res["fl_line"],
                     final.pa, final.fl, final.mt],
    }, index=["PA по фрагментам", "PA tensor", "фрагментов", "MT", "FL = MT/sin PA", "FL по прямой",
              "PA итог", "FL итог", "MT итог"]).round(2))
    st.caption("Итог = калибровка из work/calibration.json + ручные сдвиги слева. "
               "В сабмите ещё берётся медиана внутри 5-кадровой серии.")

if source == "Тестовый набор" and submission_files():
    with right.expander("Сравнить с сабмитом"):
        name = st.selectbox("Файл из work/", submission_files())
        sub = pd.read_csv(WORK / name).set_index("image_id")
        if frame["key"] in sub.index:
            s = sub.loc[frame["key"]]
            st.write(f"PA {s.pa_deg:.1f}°, FL {s.fl_mm:.1f} мм, MT {s.mt_mm:.1f} мм")
