# M7 Clean Workflow (Preprocessed Metadata)

## Overview

Now that preprocessing is complete, use this **clean, simplified workflow** for feature extraction and training.

---

## Notebooks

### **13. M7 Preprocessing** ✅ DONE
- Creates clean fold files: `datasets/processed/uldd_m7/fold_{0..4}.npz`
- Tracks metadata during window creation (no reconstruction)
- Validates subject-independent splits
- **Status:** Complete, run locally only

### **14. M7 Training** ← RUN THIS NEXT
- Assumes clean preprocessing is done
- Extracts CNN features from Video_Data (if available)
- Trains 5-fold CV
- Generates results

---

## Quick Start (Colab)

### Step 1: Upload to Google Drive

```
My Drive/
├── src.zip                 (updated M7 code)
├── datasets.zip            (with uldd_m7/ folds - CLEAN metadata)
└── Video_Data.zip          (for feature extraction - recommended)
```

### Step 2: Run Notebook 14

```python
# In Colab cell:
!jupyter notebook 14-uldd-m7-training.ipynb
```

Or open directly and run sections in order:
1. **Setup & Imports**
2. **Verify Clean Folds** (checks no leakage)
3. **Import M7 Modules**
4. **Extract Features** (all folds, ~1-2 hours with GPU)
5. **Train Single Fold** (quick test, ~10 min)
6. **Full Fold-0 Training** (20 epochs, ~30 min)
7. **Full 5-Fold CV** (all folds, ~2-3 hours)
8. **Results Summary**
9. **Final Validation**

---

## Local Alternative

Run everything locally if you have resources:

```bash
cd notebooks
jupyter notebook 14-uldd-m7-training.ipynb
```

Set `VIDEO_ROOT` to your local Video_Data path.

---

## What Changed from Old Notebook 12?

| Aspect | Old (Notebook 12) | New (Notebook 14) |
|--------|---|---|
| **Purpose** | Mixed preprocessing + training | Training only |
| **Assumes** | Corrupted metadata, manual splits | Clean preprocessed metadata |
| **Folder** | `ul_dd/` (corrupted) | `uldd_m7/` (clean) |
| **Sections** | 38 cells (messy) | 9 clear sections |
| **Validation** | Manual checks | Automatic leakage detection |
| **Error handling** | Generic exceptions | Subject leakage specific |

---

## Key Differences

### ✅ **Clean Metadata**
- Each window has explicit subject, session, win_idx, start_4hz, end_4hz
- Metadata created during windowing (not reconstructed)
- Saved directly to fold file

### ✅ **Automatic Validation**
```python
# On startup, verify no leakage:
for k in range(5):
    fold = np.load(f'fold_{k}.npz')
    train_subjects ∩ test_subjects == ∅  ✓
```

### ✅ **Clear Sections**
1. Setup (2 cells)
2. Verify (1 cell)
3. Extract (3 cells)
4. Train (4 cells)
5. Results (1 cell)
6. Validate (1 cell)

---

## Timeline

- **Preprocessing (local):** 10-20 minutes (done)
- **Feature extraction (Colab GPU):** 1-2 hours
- **5-fold CV training (Colab GPU):** 2-3 hours

**Total:** ~4-5 hours (mostly compute)

---

## Files Reference

| File | Purpose | Updated |
|------|---------|---------|
| `src/data/m7_preprocess.py` | Preprocessing logic | ✓ |
| `notebooks/13-uldd-m7-preprocessing.ipynb` | Run preprocessing | ✓ |
| `notebooks/14-uldd-m7-training.ipynb` | **← NEW, run this** | ✓ |
| `datasets/processed/uldd_m7/` | Clean fold output | ✓ |
| `src/models/m7_embed.py` | Uses `uldd_m7` | ✓ |
| `src/models/m7_train.py` | Uses `uldd_m7` | ✓ |
| `notebooks/12-uldd-m7-temporal-vision.ipynb` | Old (can keep) | - |

---

## Troubleshooting

### "fold_X: MISSING"
→ Run preprocessing notebook first (notebook 13 locally)

### "Subject leakage detected"
→ Preprocessing failed, re-run notebook 13 with verbose output

### "Video_Data not found"
→ Optional, skip feature extraction if you have precomputed embeddings

### "No embeddings available"
→ Either extract (if Video_Data exists) or provide precomputed fold_X_train/test.npz

---

## Next Steps

1. ✅ Preprocessing: DONE (notebook 13)
2. → **Training: Run notebook 14**
3. → Results & evaluation
4. → Thesis submission!

