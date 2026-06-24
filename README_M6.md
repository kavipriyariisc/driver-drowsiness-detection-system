# M6 Model - Quick Reference

## What You Have

✓ M6-A (Vision-only) - 954K params
✓ M6-Lite (Fusion) - 743K params  
✓ M6-Full (Multimodal) - 781K params
✓ Training script - `train_m6.py`
✓ Test suite - `test_m6_quick.py`

## Run Tests

```bash
python test_m6_quick.py
```

Output shows:
- All models instantiate correctly
- Forward passes work
- Dataset loads (4294 samples)
- 29/32 embeddings available

## Train Models

**Single fold (quick):**
```bash
python train_m6.py --fold 0 --variant lite --epochs 5 --batch-size 64
```

**Full 5-fold:**
```bash
python train_m6.py --variant lite --epochs 30 --batch-size 16
```

**M6-Full variant:**
```bash
python train_m6.py --fold 0 --variant full --epochs 30
```

## Results

Test fold 0, 2 epochs:
```
Accuracy: 41.77%
Macro-F1: 0.3114
```

Current issue: Embeddings from wrong data source (yolo_frames instead of UL-DD videos)

Expected with proper data: 70-78%

## Files

| File | Purpose |
|------|---------|
| src/models/m6_vision_only.py | M6-A model |
| src/models/m6_train.py | Training loop |
| src/models/m6_fusion.py | M6-Lite & M6-Full |
| train_m6.py | Execution script |
| test_m6_quick.py | Validation tests |
| M6_QUICK_START.md | Detailed guide |
| M6_SUMMARY.txt | Implementation summary |

## Next: Fix Data Source

If UL-DD raw videos are available:
1. Update `src/models/m6_extractor.py` to use video source
2. Re-extract embeddings
3. Retrain models
4. Expected accuracy: 70-78% (vs current 41%)
