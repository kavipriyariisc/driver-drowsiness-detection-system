# M8 Temporal Ensemble - Quick Start Guide

## What is M8?

**Lightweight temporal modeling** using sparse frame ensemble:
- Takes 5-10 frames sampled ~1 second apart (instead of all 16)
- Extracts EfficientNet-B0 features for each sparse frame
- **Averages predictions** across frames (temporal ensemble)
- No LSTM complexity → avoids M7's overfitting

## Why M8 Instead of M7?

| Aspect | M7 (LSTM) | M8 (Ensemble) |
|--------|-----------|---------------|
| Temporal Modeling | BiLSTM (complex) | Average logits (simple) |
| Performance | 43.39% (worse than M5!) | TBD (hopefully better) |
| Computational Cost | High | Lower (5 frames vs 16) |
| Interpretability | Black box | Transparent (frame-level predictions) |
| Risk | High (new failure mode) | Low (proven averaging works) |

## Architecture

```
Input (16-frame window)
  ↓
Sample sparse frames [0, 4, 7, 11, 15]  (5 frames per second)
  ↓
EfficientNet-B0 (frozen, ImageNet pre-trained)
  ↓
Embeddings: (5, 1280)
  ↓
Per-frame classifier: 1280 → 512 → 256 → 128 → 3 (logits per frame)
  ↓
Average logits across 5 frames
  ↓
Output: (3,) logits → softmax → prediction
```

## File Structure

```
src/models/
├── m8_dataset.py      # Dataset classes for sparse frames
├── m8_embed.py        # Feature extraction (EfficientNet-B0 on sparse frames)
└── m8_train.py        # M8TemporalEnsembleModel + training functions

notebooks/
└── 15-uldd-m8-ensemble.ipynb  # Training & comparison notebook
```

## How to Run

### Step 1: Extract Sparse Frame Embeddings

```python
from src.models.m8_embed import extract_sparse_fold_split

for k in range(5):
    for split in ['train', 'test']:
        extract_sparse_fold_split(
            fold_idx=k,
            split=split,
            video_root=Path('Video_Data'),
            processed_dir=Path('datasets/processed/uldd_m7'),
            out_dir=Path('models/m8_embeddings'),
            n_sparse_frames=5,  # 5 frames per second
            device='cuda',
        )
```

Output: `models/m8_embeddings/fold_{0..4}_{train,test}.npz` (5 × 2 = 10 files)

### Step 2: Train M8 (5-Fold CV)

```python
from src.models.m8_train import cross_validate

summary = cross_validate(
    epochs=20,
    batch_size=32,
    lr=3e-4,
    emb_dir=Path('models/m8_embeddings'),
    device='cuda',
)
```

Output: `results/reports/M8_results.json` with mean accuracy, F1, etc.

### Step 3: Compare M5 vs M8

Results are automatically saved and compared in notebook Section 6.

## Expected Results

- **M5:** 54.51% ± 10.00% accuracy
- **M8:** ??? (TBD - hypothesis: similar or slightly better)

If M8 ≥ M5, thesis narrative becomes:
> "Temporal ensemble (averaging frame predictions) performs as well as single-frame classification, suggesting that frame-level predictions are already temporally stable."

If M8 < M5, stick with M5 + document why ensemble didn't help.

## Hyperparameters

- `n_sparse_frames`: 5 (can try 7-10 if time permits)
- `lr`: 3e-4 (learning rate)
- `batch_size`: 32
- `weight_decay`: 1e-4 (L2 regularization)
- `epochs`: 20 (should be enough)
- `loss`: CrossEntropyLoss (simpler than focal for this architecture)

## Estimated Runtime

- Feature extraction: 30-60 min (on T4 GPU, 10 files total)
- 5-Fold CV training: 3-5 hours (20 epochs × 5 folds)
- **Total: 4-6 hours on GPU** (~2 hours local CPU, slower)

## What if M8 Works?

✅ **Thesis Path:**
1. M5 is your primary result (54.51%)
2. M8 validates robustness (ensemble maintains performance)
3. Show why LSTM (M7) fails vs ensemble (M8) succeeds
4. Conclude: Simple temporal modeling beats complex architectures

✅ **Key Finding:** "Pretrained vision models + simple temporal averaging outperform hand-crafted features or complex temporal models"

## Troubleshooting

**Issue:** `No module named m8_dataset`
→ Ensure `sys.path.insert(0, str(ROOT / 'src'))` before import

**Issue:** Memory error during extraction
→ Reduce `batch_size` from 16 to 8, or `n_sparse_frames` from 5 to 4

**Issue:** M8 accuracy lower than expected
→ Try `n_sparse_frames=7` or `n_sparse_frames=10` (more temporal info)
→ Try `use_confidence_weighting=True` in model (confidence-weighted ensemble)

## Next Steps

1. ✅ Run notebook 15 Section 1-3 (setup + embedding extraction)
2. ✅ Run Section 4 (quick test on fold 0, 3 epochs)
3. ✅ If successful, run Section 5 (full 5-fold CV)
4. ✅ Compare M5 vs M8 in Section 6
5. ✅ If M8 ≥ M5: Use M5 for thesis with M8 as validation
6. ✅ If M8 < M5: Still valuable negative result (complexity doesn't help)

---

**Status:** Ready to run. Start with notebook 15, Section 1.
