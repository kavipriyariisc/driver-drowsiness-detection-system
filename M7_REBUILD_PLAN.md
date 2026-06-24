# M7 Rebuild: Clean Preprocessing Pipeline

## Problem

M7 was failing because metadata in `datasets/processed/ul_dd/fold_*.npz` was corrupted:

```
Expected (Fold 0):
  test_subjects:   A B C D
  train_subjects:  E F G H I J K L M N O P Q R S

Actual (Fold 0):
  test_subjects:   A B C D
  train_subjects:  A C D E F G H J K L N O P Q R S  ← WRONG! Contains test subjects
  test_subjects:   A C D E F G H J K L N O P Q R S  ← WRONG! Same as train!
```

Root cause: The M6 metadata-generation script inferred metadata heuristically, causing train/test subject leakage.

---

## Solution

**Complete rebuild of preprocessing pipeline with proper metadata tracking.**

### New Files Created

1. **`src/data/m7_preprocess.py`** (174 lines)
   - M7Preprocessor class
   - Tracks metadata DURING window creation (not reconstructed later)
   - Each window stores: subject, session, win_idx, start_4hz, end_4hz
   - Validates before saving to prevent leakage

2. **`notebooks/13-uldd-m7-preprocessing.ipynb`**
   - Runs M7 preprocessing pipeline
   - Validates subject-independent splits
   - Generates diagnostic reports

### Modified Files

1. **`src/models/m7_embed.py`**
   - Updated: `PROCESSED_DIR = ROOT / "datasets" / "processed" / "uldd_m7"`

2. **`src/models/m7_train.py`**
   - Updated: `PROCESSED_DIR = ROOT / "datasets" / "processed" / "uldd_m7"`

---

## Usage

### Step 1: Run Preprocessing

```bash
cd notebooks/
jupyter notebook 13-uldd-m7-preprocessing.ipynb
```

This will:
- Load original UL-DD data from `Extracted_Features/` and `CSV_Files/`
- Track metadata during window creation
- Build 5 subject-independent folds with metadata
- Validate train/test split before saving
- Save to `datasets/processed/uldd_m7/fold_{0..4}.npz`

Expected output:
```
[Fold 0]
  Test subjects: ['A', 'B', 'C', 'D']
  Train subjects: ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
  ✓ Validation passed
  train metadata subjects: ['E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
  test metadata subjects:  ['A', 'B', 'C', 'D']
```

### Step 2: Extract Features

```bash
python -m src.models.m7_embed --all-folds --video-root /path/to/Video_Data
```

### Step 3: Train M7

```bash
python -m src.models.m7_train --mode feature --epochs 20
```

---

## Key Improvements

### ✅ Metadata Tracking

Metadata is created during window generation, not reconstructed:

```python
# OLD (broken)
mm_subject_train = [infer from indices]  # ← Heuristic, unreliable

# NEW (correct)
for window in sliding_windows:
    metadata.append({
        'subject': subject,
        'session': session,
        'win_idx': win_idx,
        'start_4hz': start,
        'end_4hz': end,
    })
```

### ✅ Validation Before Saving

```python
# Check 1: Train metadata does not contain test subjects
assert (train_subjects_meta & test_subjects_expected) == ∅

# Check 2: Test metadata has only test subjects
assert (test_subjects_meta - test_subjects_expected) == ∅

# Check 3: Train/test metadata disjoint
assert (train_subjects_meta & test_subjects_meta) == ∅
```

### ✅ Separate Output Folder

- New: `datasets/processed/uldd_m7/` (clean)
- Old: `datasets/processed/ul_dd/` (corrupted, preserved)
- No risk of overwriting existing M1/M2/M3/M5 data

---

## Verification

After preprocessing, verify clean splits:

```python
for k in range(5):
    fold = np.load(f'datasets/processed/uldd_m7/fold_{k}.npz')
    
    train_subjects = set(fold['mm_subject_train'].astype(str))
    test_subjects = set(fold['mm_subject_test'].astype(str))
    
    assert train_subjects & test_subjects == set(), f"Fold {k}: leakage!"
    print(f"✓ Fold {k}: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")
```

---

## M7 Workflow (Updated)

```
13-uldd-m7-preprocessing.ipynb
        ↓
Generate clean folds (uldd_m7/)
        ↓
Validate train/test subjects ✓
        ↓
src/models/m7_embed.py
        ↓
Extract EfficientNet features
        ↓
src/models/m7_train.py
        ↓
5-fold CV (no leakage!)
```

---

## Files Unchanged

- `datasets/processed/ul_dd/` (original, corrupted, preserved for reference)
- `src/models/m7_dataset.py` (no changes needed)
- `src/models/m7_model.py` (no changes needed)
- M1/M2/M3/M5 data (untouched)

---

## Summary

M7 metadata corruption was caused by heuristic reconstruction of metadata after training/test split. 

**Solution:** Generate and track metadata correctly during window creation, then validate before saving.

This ensures 100% subject-independent folds without silent failures.
