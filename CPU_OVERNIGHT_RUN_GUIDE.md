# M8 Temporal Ensemble - CPU Overnight Run Guide

## Time Estimates (Local CPU)

| Phase | Duration | Notes |
|-------|----------|-------|
| **Embedding Extraction** | 2.5-3 hours | Sequential for 5 folds × 2 splits |
| **Quick Test (Fold 0, 2 epochs)** | 5-10 min | Verification that training works |
| **Full 5-Fold CV (15 epochs)** | 4-5 hours | 5 folds × 15 epochs × ~50 min/epoch |
| **Comparison & Results** | <1 min | Instant |
| **TOTAL** | **~7-8 hours** (ideal) / **~8-10 hours** (realistic with OS overhead) |

---

## Overnight Run Schedule

**Ideal Start Time:** 10:00 PM  
**Expected Completion:** 7:00-9:00 AM next morning

```
10:00 PM: Run Section 1 (Setup) — instant
10:00 PM: Run Section 2 (Embedding Extraction) — ~3 hours
  └─ You can sleep, this runs automatically
  
1:00 AM:  Embeddings complete
1:00 AM:  Run Section 4 (Quick Test) — ~10 min
  └─ Verify training works (or manual check if you wake up)

1:10 AM:  Run Section 5 (Full 5-Fold CV) — ~5 hours
  └─ Sleep again, full CV runs
  
6:00 AM:  CV complete
6:00 AM:  Run Section 6 (Comparison) — instant
  └─ Check M5 vs M8 results
```

---

## CPU Configuration (Already Set in Notebook)

```python
CPU_CONFIG = {
    'batch_size_embed': 4,      # Embedding extraction batch size
    'batch_size_train': 8,      # Training batch size
    'num_workers': 2,           # Data loading workers
    'n_sparse_frames': 5,       # Sparse frames (minimal)
    'epochs_quick_test': 2,     # Quick test: 2 epochs
    'epochs_full_cv': 15,       # Full CV: 15 epochs
}
```

### Why These Settings?

| Setting | Value | Reason |
|---------|-------|--------|
| `batch_size_embed: 4` | Small | CPU has limited memory; smaller batches fit |
| `batch_size_train: 8` | Small | 8 samples × 5 frames × 1280 dims = manageable |
| `num_workers: 2` | 2 | CPU cores available for data loading (don't exceed 4 on 8-core) |
| `n_sparse_frames: 5` | Minimal | Already at temporal sweet spot (1 frame/sec) |
| `epochs_full_cv: 15` | Reduced from 20 | Still thorough; saves 1.5 hours |

---

## Performance vs GPU

| Hardware | Time | Speed | Cost |
|----------|------|-------|------|
| **GPU (T4)** | 4-6 hours | 1x (baseline) | Free (Colab) |
| **CPU (Local)** | 8-10 hours | 0.5-0.7x (slower) | Free (your machine) |
| **GPU (RTX 3060)** | 2-3 hours | 2-3x faster | $300+ hardware |

**CPU is 2x slower but still acceptable for overnight runs.**

---

## Key Optimization Tips

### 1. **Close unnecessary applications**
   - Close browser, Discord, Spotify
   - Frees up CPU and RAM
   - Reduces context switching overhead

### 2. **Monitor CPU usage (optional)**
   ```python
   # Add this to notebook to check CPU load
   import psutil
   print(f"CPU usage: {psutil.cpu_percent()}%")
   print(f"RAM usage: {psutil.virtual_memory().percent}%")
   ```

### 3. **Set power plan to "High Performance"** (Windows)
   - Control Panel → Power Options → High Performance
   - Prevents CPU throttling during sleep

### 4. **Don't use machine for other tasks**
   - Avoid video calls, large downloads
   - Compilation tasks will steal CPU cycles

### 5. **If extraction times out**
   - Video_Root might be slow (external drive?)
   - Copy Video_Data to internal SSD if possible
   - Or pre-extract on GPU, copy embeddings locally

---

## Expected Results

### Realistic CPU Performance
- **M8 Quick Test (Fold 0, 2 epochs):** 80-95% of GPU speed
- **M8 Full CV (15 epochs):** 60-75% of GPU speed
  - GPU: ~5 min/fold/epoch
  - CPU: ~8-10 min/fold/epoch

### Potential Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| CPU maxes at 100% | Normal | Expected; CPU cores working hard |
| Memory warnings | Batch too large | Already reduced (batch_size=8) |
| Slow embedding extraction | Video_Root on slow drive | Copy to local SSD |
| Process crashes | Ran out of RAM | Close apps, reduce num_workers to 1 |

---

## Notebook Sections

### Section 1: Setup & Imports (Instant)
✅ Loads CPU config, M8 modules, prints time estimates

### Section 2: Embedding Extraction (2.5-3 hours)
✅ Extracts sparse frame features from IR videos  
⚠️ **This is the longest step**  
💡 Skip if embeddings already exist (files in `models/m8_embeddings/`)

### Section 3: Module Verification (Instant)
✅ Just checks modules are loaded

### Section 4: Quick Test (5-10 min)
✅ Trains on fold 0 for 2 epochs  
✅ Verifies pipeline works before full CV  
✅ **Run this even if embeddings take time** (can do ~1:10 AM)

### Section 5: Full 5-Fold CV (4-5 hours)
⚠️ **Main computation step**  
✅ 5 folds × 15 epochs of training  
✅ Saves best checkpoint per fold  
✅ Saves `M8_results.json` automatically

### Section 6: M5 vs M8 Comparison (Instant)
✅ Loads results, compares models, shows winner

---

## What to Do While Running

### Option A: Sleep (Recommended)
- Set up run at 10 PM
- Sleep 8 hours
- Check results at 6 AM
- ✅ Most efficient for overnight

### Option B: Monitor (Optional)
- Keep terminal visible
- Check progress every 30-60 min
- First 3 hours: embedding extraction (watch for errors)
- Hour 3+: training starts (more predictable)

### Option C: Remote Access (Advanced)
```powershell
# Start notebook in background (PowerShell)
$job = Start-Job -ScriptBlock {
    cd "C:\Users\raka1005\Documents\IISC\..."
    & ".\.venv\Scripts\Activate.ps1"
    jupyter notebook 15-uldd-m8-ensemble.ipynb
}

# Check progress later
Get-Job $job
```

---

## Success Criteria

✅ **All sections complete:**
- Section 1: Module import messages
- Section 2: "✓ Feature extraction complete"
- Section 4: "✓ Quick test complete"
- Section 5: "✓✓✓ M8 Cross-Validation Complete ✓✓✓"
- Section 6: "✓ M8 WINS" or "✗ M5 still better"

✅ **Results saved:**
- `results/reports/M8_results.json` created
- Contains mean_acc, std_acc, mean_f1, per-fold results

✅ **Checkpoints saved:**
- `models/checkpoints/M8_fold{0..4}.pt` (5 files)

---

## Post-Run: Analyzing Results

After CV completes:

1. **Compare M5 vs M8**
   - If M8 ≥ M5: Great! Ensemble works
   - If M8 < M5: Still good; shows simple models beat complex ones

2. **Check per-fold variability**
   - Look at M8_results.json for std_acc
   - High variance? → Data imbalance or small folds

3. **Confusion matrices**
   - Which classes misclassified most?
   - Does M8 fix M7's "all drowsy" problem?

---

## Example Output

```
[M8 Sparse Frame Embeddings (EfficientNet-B0 on CPU)]

CONFIG:
  n_sparse_frames = 5
  batch_size = 4
  device = cpu
  Estimated time: 2.5-3 hours

Extracting fold_0_train...
  ✓ Saved (125.3 MB)
Extracting fold_0_test...
  ✓ Saved (31.5 MB)
[... 8 more folds ...]

✓ Feature extraction complete

========================================================================
M8 TRAINING: FOLD 0
========================================================================
Epoch 1/15
  Train: acc=0.3467, f1=0.2891
  Test:  acc=0.3604, f1=0.2745, bal_acc=0.3456
[... 14 more epochs ...]
Epoch 15/15
  Train: acc=0.6234, f1=0.5123
  Test:  acc=0.5500, f1=0.4890, bal_acc=0.5234

[... fold 1-4 ...]

========================================================================
M8 5-FOLD CV SUMMARY
========================================================================
Mean Accuracy:     0.5239 ± 0.0892
Mean Macro-F1:     0.4567 ± 0.1023
Mean Bal.Accuracy: 0.5012 ± 0.0945
========================================================================

[M5 vs M8 Comparison]

✓ M5: acc=0.5451 ± 0.1000
✓ M8: acc=0.5239 ± 0.0892

======================================================================
✗ M5 still better: M8 -3.88% vs M5
======================================================================
```

---

## Troubleshooting for CPU Runs

### Problem: Very slow embedding extraction (>1 hour for fold_0_train)

**Diagnosis:**
- Video_Data on external/slow drive → CPU limited by I/O
- Large video files being read frame-by-frame

**Solution:**
```python
# Copy Video_Data to local SSD first
import shutil
shutil.copytree(
    'C:/Users/raka1005/Documents/IISC/UL-DD/Video_Data',
    'D:/Video_Data_Local',  # Faster local drive
    dirs_exist_ok=True
)
# Update VIDEO_ROOT in notebook to D:/Video_Data_Local
```

### Problem: Process killed/crashed (Out of Memory)

**Diagnosis:**
- Batch size too large for CPU RAM
- Too many num_workers competing for memory

**Solution:**
```python
# Reduce in notebook Section 1
CPU_CONFIG = {
    'batch_size_embed': 2,      # Halve this
    'batch_size_train': 4,      # Halve this
    'num_workers': 1,           # Reduce to 1
}
```

### Problem: Extraction finishes but training doesn't start

**Diagnosis:**
- Embeddings corrupted or incomplete

**Solution:**
```python
# Check embeddings before training
import numpy as np
from pathlib import Path

emb_file = Path('models/m8_embeddings/fold_0_train.npz')
with np.load(emb_file) as f:
    print(f"Embeddings shape: {f['embeddings'].shape}")
    print(f"Labels shape: {f['labels'].shape}")
    # Expected: (N, 5, 1280) and (N,)
```

---

## Final Notes

✅ **You're ready for overnight run!**

- Notebook already optimized for CPU
- Realistic time: 8-10 hours (perfect for overnight)
- All hyperparameters tuned for your local machine
- Results comparable to GPU (just slower)

**Start at 10 PM → Sleep → Check results at 8 AM** ✨

