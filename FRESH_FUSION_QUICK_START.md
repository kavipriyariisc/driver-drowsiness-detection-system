# Fresh Fusion Quick Start Guide

This guide helps you quickly train and evaluate the Fresh Fusion driver drowsiness detection architecture.

## Setup

```bash
# Install dependencies (if not already installed)
pip install torch torchvision scikit-learn pandas numpy

# Navigate to project root
cd /path/to/driver-drowsiness-detection-system
```

## Quick Training (Debug Mode - 2 epochs, fast)

```bash
# Train all models in debug mode
python -m src.models.fresh_fusion.train_camera --debug
python -m src.models.fresh_fusion.train_facial --debug
python -m src.models.fresh_fusion.train_telemetry --debug
python -m src.models.fresh_fusion.train_fusion --debug

# Evaluate
python -m src.models.fresh_fusion.evaluate
```

**Expected Time:** ~5-10 minutes on GPU

## Full Training (30 epochs)

```bash
# Train individual models
python -m src.models.fresh_fusion.train_camera --epochs 30
python -m src.models.fresh_fusion.train_facial --epochs 30
python -m src.models.fresh_fusion.train_telemetry --epochs 30

# Train fusion (requires above 3)
python -m src.models.fresh_fusion.train_fusion --epochs 30

# Comprehensive evaluation
python -m src.models.fresh_fusion.evaluate
```

**Expected Time:** ~2-4 hours on GPU

## View Results

```bash
# Results summary table
cat results/fresh_fusion/fresh_fusion_summary.csv

# Detailed metrics
cat results/fresh_fusion/camera_results.json
cat results/fresh_fusion/facial_results.json
cat results/fresh_fusion/telemetry_results.json
cat results/fresh_fusion/fusion_results.json

# Fusion modality weights
cat results/fresh_fusion/fusion_modality_weights.json
```

## Key Files

| File | Purpose |
|------|---------|
| `src/models/fresh_fusion/dataset.py` | Subject-wise dataset loading |
| `src/models/fresh_fusion/models.py` | 4 model architectures |
| `src/models/fresh_fusion/utils.py` | Telemetry feature engineering |
| `src/models/fresh_fusion/train_camera.py` | Camera model training |
| `src/models/fresh_fusion/train_facial.py` | Facial model training |
| `src/models/fresh_fusion/train_telemetry.py` | Telemetry model training + feature engineering |
| `src/models/fresh_fusion/train_fusion.py` | Learned fusion training |
| `src/models/fresh_fusion/evaluate.py` | Final evaluation & comparison |
| `src/models/fresh_fusion/README.md` | Detailed documentation |

## Expected Outputs

```
results/fresh_fusion/
├── fresh_fusion_summary.csv          ← Comparison table
├── fresh_fusion_results.json         ← All metrics
├── fusion_modality_weights.json      ← Learned weights
├── camera_results.json
├── facial_results.json
├── telemetry_results.json
├── fusion_results.json
└── *_embeddings.npy                 ← Extracted features
```

## Model Overview

| Model | Input | Output | Key Feature |
|-------|-------|--------|------------|
| Camera | 8 frames (224×224) | 128-dim | Temporal attention |
| Facial | FAU sequence (T, 30) | 192-dim | BiLSTM + attention |
| Telemetry | CAN sequence (T, 5) | 96-dim | Transformer + feature engineering |
| Fusion | All 3 modalities | Logits + weights | Learned modality weighting |

## Architecture Diagrams

### Visual Pipeline
```
Frames (B, 8, 3, 224, 224)
  ↓ MobileNetV3-Small
Frame embeddings (B, 8, 128)
  ↓ Temporal Attention
Visual embedding (B, 128)
  ↓ Classifier
Logits (B, 3)
```

### Facial Pipeline
```
FAU (B, T, 30)
  ↓ Project to 64
  ↓ BiLSTM (→ 192)
  ↓ Attention pooling
Facial embedding (B, 192)
  ↓ Classifier
Logits (B, 3)
```

### Telemetry Pipeline
```
Raw CAN (B, T, 5)
  ↓ Engineer features (→ 25)
  ↓ Project to 96
  ↓ Positional encoding
  ↓ Transformer
  ↓ Attention pooling
Telemetry embedding (B, 96)
  ↓ Classifier
Logits (B, 3)
```

### Fusion Pipeline
```
Visual (128), Facial (192), Telemetry (96)
  ↓ Project all to 128
  ↓ Compute confidence scores
  ↓ Softmax weights
  ↓ Weighted fusion (128) + concat
Fused features (B, 512)
  ↓ Classifier
Logits (B, 3) + Weights
```

## Expected Performance

**Baseline (M5):** Accuracy 54.51%, Macro-F1 0.4439

**Target:** Accuracy > 60%, Macro-F1 > 0.45

## Troubleshooting

**CUDA out of memory:**
```bash
python -m src.models.fresh_fusion.train_camera --batch-size 8
```

**Slow training:**
- Reduce epochs for testing: `--epochs 5`
- Use debug mode: `--debug`

**Missing imports:**
```bash
pip install torch torchvision scikit-learn pandas
```

## For More Details

See [src/models/fresh_fusion/README.md](src/models/fresh_fusion/README.md) for comprehensive documentation including:
- Detailed architecture descriptions
- All hyperparameters
- Dataset handling
- Feature engineering specifics
- Known limitations

## Next Steps

1. Start with debug mode to verify setup
2. Train individual models (can be parallel)
3. Train fusion model
4. Review comparison table in `results/fresh_fusion/fresh_fusion_summary.csv`
5. Analyze modality weights in `fusion_modality_weights.json`
