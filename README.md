# Driver Drowsiness Detection Using Camera and CAN Bus Signals

> **MTech Thesis Project** — Multimodal Drowsiness Detection on the UL-DD Dataset  
> Framework: TensorFlow 2.x | Python 3.10+ | Real-time via YOLOv8 + MediaPipe

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset — UL-DD](#dataset--ul-dd)
3. [Model Architectures](#model-architectures)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Notebook Guide](#notebook-guide)
7. [Results](#results)
8. [Real-Time Demo](#real-time-demo)
9. [Citation](#citation)

---

## Overview

This project investigates **multimodal driver drowsiness detection** by fusing two complementary signals:

| Modality | Source | Features |
|----------|--------|----------|
| **Facial Action Units (FAU)** | Front-facing camera | 30 AUs @ 60 Hz (OpenFace) |
| **Vehicle Telemetry (CAN)** | OBD-II / CAN bus | pitch, roll, speed, RPM, gear @ 60 Hz |

Drowsiness level is defined via the **Karolinska Sleepiness Scale (KSS)**:

| KSS | Label | Class |
|-----|-------|-------|
| 1–3 | Alert | 0 |
| 4–6 | Low Vigilant | 1 |
| 7–9 | Drowsy | 2 |

The **thesis contribution** is **M3 — Cross-Modal Attention Fusion**: a bidirectional LSTM-based model with learnable cross-modal attention that lets the camera stream attend to CAN telemetry and vice versa, producing interpretable attention weight maps.

---

## Dataset — UL-DD

**University of Louisiana Drowsiness Detection (UL-DD) Dataset**  
19 subjects (A–S), 2 sessions each (Alert / Drowsy), 40 min @ 60 Hz, KSS annotations every 4 min.

```
UL-DD/
├── Extracted_Features/…/{Subject}/{Session}/{S}_FAU_{Session}.csv   # 30 AUs
├── CSV_Files/…/{Subject}/{Session}/{S}_Telemetry_{Session}.csv      # CAN
└── Labels.csv                                                        # 10 KSS per row
```

**Known gaps:**
- Subject **A** — no Telemetry session
- Subjects **C, F, L** — no Drowsy session

**5-Fold subject-independent cross-validation split:**

| Fold | Test subjects |
|------|--------------|
| 0 | A B C D |
| 1 | E F G H |
| 2 | I J K |
| 3 | L M N O |
| 4 | P Q R S |

---

## Model Architectures

| Model | Input | Architecture | Params (approx.) |
|-------|-------|-------------|-----------------|
| **M1** — Facial BiLSTM | FAU (240 × 30) | BiLSTM(128) → BiLSTM(64) → Dense(3) | ~420K |
| **M2** — Telemetry LSTM | CAN (240 × 5) | LSTM(64) → LSTM(32) → Dense(3) | ~45K |
| **M3** — Cross-Modal Fusion ⭐ | FAU + CAN | BiLSTM + LSTM + CrossModalAttention → Dense(3) | ~750K |
| **M4** — Real-Time BiLSTM | 10 live features | BiLSTM(64) → BiLSTM(32) → Dense(3) | ~82K |

`WINDOW = 240` timesteps = 60 seconds at 4 Hz (downsampled from 60 Hz).

### M3 Attention Mechanism

```
H_fau  = BiLSTM(FAU)   → (B, 240, 256)
H_tele = LSTM(Tele)    → (B, 240, 64)

FAU→Tele: ctx_f2t = Attention(Q=H_fau,  K=H_tele, V=H_tele)
Tele→FAU: ctx_t2f = Attention(Q=H_tele, K=H_fau,  V=H_fau )

output = Dense3(GAP[ctx_f2t | ctx_t2f | H_fau | H_tele])
```

---

## Project Structure

```
driver-drowsiness-detection-system/
├── src/
│   ├── models/
│   │   ├── architecture.py     # M1, M2, M3, M4 + build_model()
│   │   └── train.py            # compile_model, make_callbacks, cross_validate()
│   ├── data/
│   │   ├── preprocess.py       # ULDDProcessor (sliding windows, fold splits)
│   │   ├── load_data.py        # load_fold, load_all_folds, class_distribution
│   │   └── telemetry_replay.py # TelemetryReplayer + RingBuffer (live demo)
│   ├── inference/
│   │   └── predict.py          # ULDDPredictor (M1/M2/M3/M4)
│   ├── utils/
│   │   └── helpers.py          # metrics, confusion matrix, attention plots
│   ├── face_detection.py       # YOLOv8 + MediaPipe → 10 real-time features
│   ├── scoring.py              # DrowsinessScorer + AlertGenerator
│   └── realtime_demo.py        # Live webcam + telemetry replay demo
├── notebooks/
│   ├── 04-uldd-eda.ipynb           # Exploratory data analysis
│   ├── 05-uldd-preprocessing.ipynb # Run ULDDProcessor, save folds
│   ├── 06-uldd-m1-facial.ipynb     # Train & evaluate M1
│   ├── 07-uldd-m2-telemetry.ipynb  # Train & evaluate M2
│   ├── 08-uldd-m3-fusion.ipynb     # Train & evaluate M3 (main contribution)
│   └── 10-comparison.ipynb         # All-model comparison + attention viz
├── datasets/processed/ul_dd/       # fold_0.npz … fold_4.npz (git-ignored)
├── models/
│   ├── checkpoints/                # Best .keras per fold (git-ignored)
│   └── exports/                    # Final exported models
├── tests/
│   ├── test_data.py
│   └── test_models.py
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (if not done)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Preprocess the UL-DD dataset

Open `notebooks/05-uldd-preprocessing.ipynb` and set `ULDD_ROOT` to your local UL-DD path:

```python
ULDD_ROOT = r"C:/Users/raka1005/Documents/IISC/UL-DD"
```

Run all cells — this produces `datasets/processed/ul_dd/fold_0.npz` … `fold_4.npz`.

### 3. Train models

Run notebooks in order:

```
06-uldd-m1-facial.ipynb      → M1 checkpoints
07-uldd-m2-telemetry.ipynb   → M2 checkpoints
08-uldd-m3-fusion.ipynb      → M3 checkpoints  ← thesis main result
```

Or run from the terminal:

```python
from src.models.architecture import build_model
from src.models.train import cross_validate

results = cross_validate(
    model_fn   = lambda: build_model('m3'),
    model_name = 'M3',
    fold_dir   = 'datasets/processed/ul_dd',
    input_keys = (['fau_windows', 'tele_windows'], 'labels'),
)
print(f"M3: {results['mean_acc']:.3f} ± {results['std_acc']:.3f}")
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Real-time demo

```bash
python src/realtime_demo.py
```

Press **q** to quit.

---

## Notebook Guide

| Notebook | Purpose |
|----------|---------|
| `04-uldd-eda.ipynb` | Inspect FAU/CAN distributions, KSS histogram, missing-data heatmap |
| `05-uldd-preprocessing.ipynb` | Run `ULDDProcessor`, verify fold shapes, visualise windows |
| `06-uldd-m1-facial.ipynb` | 5-fold CV on FAU-only BiLSTM (baseline camera) |
| `07-uldd-m2-telemetry.ipynb` | 5-fold CV on CAN-only LSTM (baseline telemetry) |
| `08-uldd-m3-fusion.ipynb` | 5-fold CV on cross-modal attention fusion (main contribution) |
| `10-comparison.ipynb` | Side-by-side accuracy/F1 table, confusion matrices, attention weight maps |

---

## Results

*(Fill after experiments)*

| Model | Val Accuracy | Macro-F1 | Notes |
|-------|-------------|----------|-------|
| M1 — Facial BiLSTM | 0.3913 | 0.3475 | Camera-only baseline (FAU features) |
| M2 — Telemetry LSTM | 0.3867 | 0.3386 | CAN-only baseline |
| M3 — Fusion (earlier fusion) | 0.3533 | 0.3332 | Early fusion attempt — underperforms on IR data |
| **M5 — YOLOv8-cls (Baseline)** | **0.5451** | **0.4439** | Image-based baseline (224×224 IR frames) — best performing so far |
| M4 — Real-Time | — | — | Live YOLO features |

---

## Real-Time Demo

The demo fuses a **live webcam** with a **CSV telemetry replay** (any UL-DD subject):

```
Camera (30 fps)
  └─ YOLOv8 face detect
  └─ MediaPipe FaceMesh → EAR, MAR, PERCLOS, blink rate, head pose (10 features)
  └─ 4 Hz ring buffer (240 samples = 60 s)

Telemetry CSV (looped, background thread)
  └─ TelemetryReplayer @ 4 Hz drain
  └─ 4 Hz ring buffer (240 × 5)

Every 15 s → M3 inference → DrowsinessScorer → AlertGenerator → HUD overlay
```

Configure paths in `src/realtime_demo.py`:

```python
TELEMETRY_CSV = r"C:/path/to/UL-DD/CSV_Files/.../B_Telemetry_A.csv"
MODEL_PATH    = "models/checkpoints/M3_fold0.keras"  # optional
```

---

## Citation

If you use this code or the UL-DD dataset, please cite:

```bibtex
@dataset{uldd2023,
  title  = {UL-DD: University of Louisiana Drowsiness Detection Dataset},
  year   = {2023},
}

@mastersthesis{your2025,
  author = {Your Name},
  title  = {Driver Drowsiness Detection Using Camera and CAN Bus Signals},
  school = {Indian Institute of Science},
  year   = {2025},
}
```

---

## License

See [LICENSE](LICENSE).
