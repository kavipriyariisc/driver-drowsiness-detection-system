# M7 Final Hardening — Quick Reference

## What Changed

### Core Pipeline (m7_dataset.py)
- ✅ **IR-only videos** enforced (no AI, L3D, Pose, R3D mixing)
- ✅ **No sample substitution** in `__getitem__()` (strict mode)
- ✅ **Renamed validation**: `_can_decode_any()` → `_can_decode_window()`
- ✅ **Strict frame validation**: ALL 16 frames must be in-bounds AND decodable
- ✅ **Strong logging**: Shows which samples filtered and why

### Feature Extraction (m7_embed.py)
- ✅ **Counters**: `processed_windows`, `saved_windows`, `failed_windows`
- ✅ **Error if empty**: Raises RuntimeError if no windows extracted
- ✅ **Extra metadata**: Saves `img_size`, `fold_idx`, `split` in .npz

### Model (m7_model.py)
- ✅ **Bigger BiLSTM**: 128 → **256** hidden units
- ✅ **Deeper classifier**: 4-layer with staged dropout (0.5, 0.3, 0.2)

### Training (m7_train.py)
- ✅ **Leakage detection**: Explicit check for train/test subject overlap
- ✅ **Shape validation**: Ensures (N, 16, D) format
- ✅ **Feature-mode mandatory**: Frame mode raises NotImplementedError
- ✅ **Focal Loss**: Optional via `--loss focal`
- ✅ **Class distribution**: Printed per-split (train/val/test)

### Notebook (12-uldd-m7-temporal-vision.ipynb)
- ✅ **5-step workflow**: Extract → Verify → QuickTest → FullTrain → CV
- ✅ **No fallbacks**: Clean, explicit pipeline
- ✅ **IR-only**: Missing VIDEO_ROOT handled gracefully

---

## Running M7

### Step 1: Extract Features (One-time)
```bash
python -m src.models.m7_embed --all-folds --video-root /path/to/Video_Data
```

**Expected output:**
```
[M7 embed] fold=0 split=train backbone=efficientnet_b0
   [M7WindowDataset] split=train
     original samples    : 4294
     kept samples        : 3847
     dropped samples     : 447
     missing video       : 123
   saved fold_0_train.npz  X=(3847, 16, 1280)
```

### Step 2: Train (Quick Test)
```bash
python -m src.models.m7_train --fold 0 --epochs 2 --mode feature
```

### Step 3: Train (Full)
```bash
python -m src.models.m7_train --fold 0 --epochs 20 --mode feature
```

### Step 4: 5-Fold CV
```bash
python -m src.models.m7_train --all-folds --epochs 20 --mode feature
```

**OR use Jupyter notebook:**
```bash
jupyter notebook notebooks/12-uldd-m7-temporal-vision.ipynb
```

---

## CLI Options

```bash
python -m src.models.m7_train --help

--fold FOLD                     Single fold (0-4)
--all-folds                     Run full 5-fold CV (default if --fold not set)
--mode {feature}                Feature mode only (frame mode disabled)
--backbone {efficientnet_b0, resnet18, mobilenet_v3_small}
--freeze-backbone               Keep backbone frozen (default)
--unfreeze-last-block           Fine-tune last block
--epochs EPOCHS                 Number of epochs (default: 20)
--batch-size BS                 Batch size (default: 32)
--lr LR                         Learning rate (default: 3e-4)
--weight-decay WD               L2 regularization (default: 1e-4)
--loss {ce, focal}              Loss function (default: ce)
--seed SEED                     Random seed (default: 42)
```

---

## Key Metrics

**Expected realistic results (after hardening):**
- Accuracy: 50–60%
- Macro-F1: 0.40–0.50
- Balanced Accuracy: 45–55%

**No hardcoded expectations** — actual values reported.

---

## Error Messages & Meanings

| Error | Cause | Fix |
|-------|-------|-----|
| `Embeddings missing` | Feature files not found | Run m7_embed first |
| `Subject leakage detected` | Train/test overlap | Dataset error; investigate fold_*.npz |
| `No frames could be decoded` | Video frame range invalid | Sample filtered by strict validation |
| `Frame mode is disabled` | Trying to use frame mode | Feature mode only; extract embeddings first |
| `No valid windows extracted` | All samples filtered | Check VIDEO_ROOT and dataset integrity |
| `T_VIS mismatch` | Wrong number of frames | Ensure embeddings generated with T_VIS=16 |

---

## Dataset Filtering Example

When dataset initializes, it prints:

```
[M7WindowDataset] split=train
  original samples    : 4294          ← Total samples in fold
  kept samples        : 3847          ← Used for training
  dropped samples     : 447           ← Excluded (details below)
  missing video       : 123           ← Video file not found
  invalid range       : 45            ← Frame range ≤ 0
  decode failures     : 279           ← Strict frame validation failed
  subjects (kept)     : ['A', 'D', 'E', ...]  ← Available subjects
  sessions (kept)     : ['A', 'D']    ← Available sessions
```

**Analysis:**
- 123 missing videos → Some subject/session combinations don't exist in dataset
- 45 invalid ranges → Corrupted metadata
- 279 decode failures → Frames out-of-bounds or undecodable

**This is OK!** Filtering is expected and healthy.

---

## File Sizes & Expectations

| File | Size | Notes |
|------|------|-------|
| `fold_0_train.npz` | ~150 MB | ~3800 samples × 16 frames × 1280D float32 |
| `M5_fold0.pt` | ~50 MB | Model checkpoint |
| `M7_results.json` | ~10 KB | CV summary (5 folds) |

---

## Success Indicators

✅ Feature extraction completes for all folds
✅ No "Subject leakage" errors
✅ Training shows decreasing loss
✅ Validation F1 improves over epochs
✅ Results saved to `results/reports/M7_results.json`
✅ 5-fold CV produces stats (mean ± std)

---

## Troubleshooting

**Problem:** VIDEO_ROOT not found
```
WARNING: VIDEO_ROOT not found at ...
Skipping feature extraction.
```
**Solution:** 
- Ensure `Video_Data` exists or set correct path in notebook
- If using Colab, upload and extract videos first

**Problem:** No embeddings available
```
FileNotFoundError: Embedding files missing
```
**Solution:**
```bash
python -m src.models.m7_embed --all-folds --video-root /path/to/Video_Data
```

**Problem:** Training freezes or runs out of CUDA memory
**Solution:**
```bash
--batch-size 16          # Reduce from 32
--num-workers 0          # Disable workers if on Colab
```

**Problem:** Class imbalance (one class dominates)
**Solution:**
```bash
--loss focal             # Switch to Focal Loss
```

---

## Architecture Summary

```
Input: IR Video [60fps × 60sec = 3600 frames]
   ↓
Temporal window [4Hz align: 16 frames sampled from T_start to T_end]
   ↓
Strict validation [ALL 16 frames in-bounds? Decodable?]
   ↓
EfficientNet-B0 (frozen) → Extract per-frame feature (1280D)
   ↓
Temporal sequence [16 × 1280]
   ↓
BiLSTM(256, bidirectional) → [16 × 512]
   ↓
Attention pooling → [512]
   ↓
Classifier (4-layer, staged dropout) → [3 logits]
   ↓
Weighted CE Loss
   ↓
Output: Alert / LowVigilant / Drowsy
```

---

## Thesis Points

1. **No YOLO embeddings** — Pure temporal vision from raw IR frames
2. **Subject-independent CV** — No data leakage; subjects are held-out
3. **Strict window alignment** — Exact 4Hz ↔ 60fps mapping preserved
4. **Reproducible** — Fixed seeds, deterministic hyperparams, saved metadata
5. **Transparent** — No hidden substitutions, explicit filtering, clear logging
6. **Robust** — Handles missing videos, undecodable frames, session token corruption
7. **Production-ready** — All errors caught, no silent failures

---

**Last Updated:** Phase 10, All 14 fixes applied ✅
**Status:** Production Ready
**Next Step:** Run `jupyter notebook notebooks/12-uldd-m7-temporal-vision.ipynb`
