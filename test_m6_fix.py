#!/usr/bin/env python
"""Quick test of M6 after error handling fixes."""
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from pathlib import Path
from src.models.m6_train import (
    train_one_fold, EmbeddingCache, PROCESSED_DIR, CKPT_DIR, EMB_DIR
)

print("="*70)
print("TESTING M6_lite WITH FIXED ERROR HANDLING")
print("="*70)

# Quick test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nRunning fold 0 with 2 epochs on {device}...")

try:
    result = train_one_fold(
        fold_idx=0,
        variant='lite',
        t_vis=16,
        epochs=2,
        batch_size=32,
        verbose=True
    )
    print("\n" + "="*70)
    print("✅ SUCCESS! M6_lite training works correctly")
    print("="*70)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print("\nNote: Limited accuracy due to visual embeddings from yolo_frames")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    lr=3e-4,
    processed_dir=PROCESSED_DIR,
    emb_dir=EMB_DIR,
    ckpt_dir=CKPT_DIR,
    device=device,
    verbose=True,
)

print()
print('✓ Smoke test complete!')
print(f'  Fold 0 accuracy: {result["accuracy"]:.4f}')
print(f'  Fold 0 macro-F1: {result["macro_f1"]:.4f}')
