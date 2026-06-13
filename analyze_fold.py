#!/usr/bin/env python
"""Analyze fold data dimensions."""
import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path

fold_path = Path('datasets/processed/ul_dd/fold_0.npz')
with np.load(str(fold_path), allow_pickle=False) as f:
    print("Fold 0 structure:")
    print()
    
    for key in sorted(f.keys()):
        arr = f[key]
        if hasattr(arr, 'shape'):
            print(f"  {key:20} {str(arr.shape):30} {arr.dtype}")
        else:
            print(f"  {key:20} {arr}")
    
    print()
    print(f"mm_tele_train shape: {f['mm_tele_train'].shape}")
    print(f"  Dimension 2 (time steps per window): {f['mm_tele_train'].shape[1]}")
    print()
    
    # If windows are 160 timesteps, that's 40 seconds at 4 Hz
    ts_per_window = f['mm_tele_train'].shape[1]
    secs = ts_per_window / 4
    print(f"  Window duration: {ts_per_window} @ 4Hz = {secs}s")
