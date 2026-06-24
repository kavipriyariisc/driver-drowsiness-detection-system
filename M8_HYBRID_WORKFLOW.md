# M8 Hybrid Workflow: CPU Extraction + Colab Training

## Why This Approach Works

| Component | Why | Where |
|-----------|-----|-------|
| **Embedding Extraction** | I/O bound (reading videos frame-by-frame) | ✅ CPU (local) |
| **Training** | Compute bound (matrix operations) | ✅ GPU (Colab) |

CPU is actually better for I/O; GPU waits around.

---

## What Are M8 Embeddings?

M8 needs **sparse frame embeddings** specific to its architecture:

```
M7 Input:  16 IR frames per window
           └─→ EfficientNet-B0 
           └─→ 16 embeddings of (1280,) each
           └─→ Shape: (N, 16, 1280)

M8 Input:  5 SPARSE frames (indices [0, 4, 7, 11, 15])
           └─→ Sample 1 frame per second
           └─→ EfficientNet-B0 on only those 5
           └─→ 5 embeddings of (1280,) each
           └─→ Shape: (N, 5, 1280)  ← Different!
```

**Can't reuse M7 embeddings** because dimensions don't match.

---

## Workflow

### Step 1: Local CPU (Tonight/Tomorrow)
```
Extract M8 Sparse Embeddings
├─ Read Video_Data/ (I/O bound)
├─ Sample frames [0, 4, 7, 11, 15]
├─ Extract EfficientNet-B0 features
└─ Save 10 .npz files (~1.5 GB total)

Time: 2.5-3 hours (CPU)
```

### Step 2: Upload to Google Drive
```
Copy m8_embeddings/ to Drive
Time: 10-30 min (depends on internet)
```

### Step 3: Colab Training (Next day)
```
Train on GPU
├─ Download embeddings from Drive
├─ Quick test (5 epochs): ~10 min
├─ Full 5-fold CV (20 epochs): ~1.5 hours
└─ Compare M5 vs M8 instantly

Time: 2 hours total on GPU
```

---

## Instructions

### Part A: CPU Extraction (Local)

**Option 1: Run Python Script**
```bash
cd "C:\Users\raka1005\Documents\IISC\driver-drowsiness-detection-system"
python extract_m8_embeddings.py
# Or specific fold:
python extract_m8_embeddings.py --fold 0 --split train
```

**Option 2: Run from Jupyter**
```python
import subprocess
result = subprocess.run([
    'python', 'extract_m8_embeddings.py',
    '--batch-size', '4'
], cwd=Path.cwd())
```

**Option 3: Use old notebook 15**
- Open `notebooks/15-uldd-m8-ensemble.ipynb`
- Skip Section 5 (full CV)
- Just run Sections 1-2 (setup + extraction)

**Expected Output:**
```
[1/10] fold_0_train - EXTRACTING
Estimated time: 20-30 min
✓ Saved (130.2 MB)

[2/10] fold_0_test - EXTRACTING
Estimated time: 5-10 min
✓ Saved (32.5 MB)

[... 8 more folds ...]

EXTRACTION COMPLETE
Total files: 10/10
Output directory: C:\...\models\m8_embeddings

📤 Next: Copy to Google Drive → Colab
```

### Part B: Upload to Google Drive

```
1. Open Google Drive
2. Create folder: "m8_embeddings"
3. Upload: C:\...\models\m8_embeddings\fold_*.npz (all 10 files)
4. Or zip first: m8_embeddings.zip → upload
   (Then copy unzipped.zip to Drive as m8_embeddings/)
```

### Part C: Colab Training

```
1. Open Colab
2. Upload notebook: notebooks/16-uldd-m8-colab-training.ipynb
   (Or create new from: https://colab.research.google.com)

3. Cell 1: Run setup (checks GPU)

4. Cell 2: Mount Drive & extract
   - Mounts Google Drive
   - Copies m8_embeddings from Drive to /content/

5. Cell 3: Import M8 modules

6. Cell 4: Quick test (5 epochs, ~10 min)
   - Verifies training works

7. Cell 5: Full 5-fold CV (20 epochs, ~1.5 hours)
   - Actual training
   - Saves results to /content/results/reports/M8_results.json

8. Cell 6: Compare M5 vs M8
   - Shows who won
   - All results visible
```

---

## Time Breakdown

| Step | Time | Device |
|------|------|--------|
| CPU Extraction (local) | 2.5-3 hours | Your machine |
| Upload to Drive | 10-30 min | Internet |
| Colab Setup & Quick Test | 15 min | GPU (T4) |
| Colab Full CV | 1.5-2 hours | GPU (T4) |
| **TOTAL** | **~7-8 hours** | (2.5h CPU + 2h GPU) |

---

## Example Timeline

```
8:00 PM  → Start Python extraction locally
          └─ Computer runs, you can work/sleep

10:30 PM → Upload m8_embeddings to Drive
          └─ Takes 10-30 min

11:00 PM → (Optional) Start Colab
          └─ Run Cells 1-3 to verify setup
          └─ Or wait until morning

Next AM  → Run Colab Cells 4-6
          ├─ Cell 4: Quick test (~10 min)
          ├─ Cell 5: Full CV (~1.5-2 hours)
          │          You can close browser, GPU keeps running
          └─ Cell 6: Results & comparison (instant)
```

---

## Key Advantages

✅ **CPU handles I/O** (video reading) where it excels  
✅ **GPU handles compute** (matrix ops) where it excels  
✅ **Parallel:** Extract while sleeping, train while eating  
✅ **Flexible:** Extract all 5 folds locally, train on GPU whenever  
✅ **Cost:** $0 (your machine + free Colab)  
✅ **Speed:** ~7-8 hours total (vs 15-20 hours all CPU)  

---

## Troubleshooting

### CPU Extraction Issues

**Q: Script is very slow (>1 hour per fold)**  
A: Video_Data on slow drive? Copy to local SSD first:
```python
import shutil
shutil.copytree(
    'C:/Users/raka1005/Documents/IISC/UL-DD/Video_Data',
    'D:/Video_Data_fast',  # Local SSD
    dirs_exist_ok=True
)
# Then update VIDEO_ROOT in script
```

**Q: Memory error**  
A: Reduce batch size:
```bash
python extract_m8_embeddings.py --batch-size 2
```

**Q: Crash mid-extraction**  
A: Resume from specific fold:
```bash
python extract_m8_embeddings.py --fold 2 --split train
```

### Colab Training Issues

**Q: "No embeddings found"**  
A: Check m8_embeddings/ uploaded to Drive correctly  
- Drive should have: m8_embeddings/fold_0_train.npz, etc.

**Q: GPU not available**  
A: Runtime → Change runtime type → GPU (T4)

**Q: Colab session timeout**  
A: Run cells 4-6 quickly, or:
```python
# In Colab, enable auto-save
# Edit → Notebook settings → Auto-save to Drive (ON)
```

---

## Why M8 Needs New Embeddings (Explained)

M8 is **specifically designed** for sparse frames:

```python
# M7 (temporal LSTM - failed)
input = all 16 frames
lstm_output = temporal_features
prediction = classifier(lstm_output)

# M8 (sparse ensemble - simpler)
input = [frame_0, frame_4, frame_7, frame_11, frame_15]  # 5 sparse frames
pred_0 = classifier(frame_0)
pred_4 = classifier(frame_4)
...
prediction = average([pred_0, pred_4, ...])  # Simple average
```

M8's architecture requires **5-frame shape (N, 5, 1280)**  
M7 embeddings are **16-frame shape (N, 16, 1280)**  
→ Can't mix them

Also: M7 might not exist yet locally; extracting M8 specific to your needs.

---

## Final Checklist

- [ ] Run CPU extraction (extract_m8_embeddings.py)
- [ ] Check 10 .npz files created in `models/m8_embeddings/`
- [ ] Upload m8_embeddings folder to Google Drive
- [ ] Open Colab, create new notebook from template
- [ ] Run Cells 1-3 (setup)
- [ ] Run Cell 4 (quick test)
- [ ] Run Cell 5 (full CV) - takes 1.5-2 hours
- [ ] Run Cell 6 (comparison)
- [ ] Check results: M5 vs M8 winner
- [ ] Download M8_results.json for thesis

✅ Done!
