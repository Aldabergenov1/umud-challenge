# Manual measurements of the test frames

One JSON file per frame: two points per aponeurosis and one fascicle, as pixel coordinates, plus the values shown by the annotation tool. No images. Measured by Arman Zhalgasbayev with `umud-annotator/`.

The millimetre values in these files use the annotator's own scale and should not be used; the pipeline recomputes them from the points with the ruler scale (`notebooks/umud_manual.py`).

These measurements are derived from the competition test set and are licensed under CC BY-NC-SA 4.0, unlike the code. The use of them is declared as external data, as the competition host requires (discussion 690868 on the competition page).
