# Implementation Guide

This brief guide describes how to reproduce the latest M5 (YOLOv8-cls) baseline and run the comparison notebook.

1. Reproduce M5 (YOLOv8-cls)

   - Ensure `datasets/yolo_frames/` exists and `FRAME_DIR / metadata.csv` is present (see `notebooks/05-uldd-preprocessing.ipynb`).
   - Install dependencies:

   ```powershell
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   - Run `notebooks/09-uldd-m5-yolo-cls.ipynb` (train/eval). The notebook will skip training if `models/checkpoints/yolo_cls/M5_fold{n}/weights/best.pt` exists and will directly evaluate.

   - M5 results (already produced) — mean accuracy: 0.5451, mean macro-F1: 0.4439. Individual fold best checkpoints are saved under `models/checkpoints/yolo_cls/` and a consolidated copy is saved as `models/checkpoints/M5_fold{n}.pt`.

2. Update & run comparison

   - The comparison notebook `notebooks/10-comparison.ipynb` loads JSON result files from `results/reports/`:
     - `M1_results.json`, `M2_results.json`, `M3_results.json`, `M5_results.json`.
   - To refresh the comparison: ensure the four JSON files exist, then run all cells in `10-comparison.ipynb`.

3. Notes & recommendations

   - M5 is the image-based baseline for this thesis. Use it as the starting point for further experiments (e.g., add temporal LSTM, EAR/MAR features, attention fusion with telemetry).
   - Do not commit large datasets or model checkpoints — they are ignored by `.gitignore`.

4. Useful commands

```powershell
# run comparison notebook (from project root)
pip install nbconvert nbformat
python -m nbformat notebooks/10-comparison.ipynb -o /dev/null  # quick syntax check
# open notebook in VS Code or Jupyter and run all cells
```

---

For more detailed reproduction steps see the specific notebooks under `notebooks/`.