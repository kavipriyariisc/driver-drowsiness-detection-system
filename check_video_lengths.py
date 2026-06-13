#!/usr/bin/env python
"""Check video lengths from embeddings."""
import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path

emb_dir = Path('models/embeddings')

sessions = ['E_A', 'E_D', 'F_A', 'G_A', 'G_D']

for sess in sessions:
    emb_path = emb_dir / f'{sess}.npz'
    if not emb_path.exists():
        continue
    
    with np.load(str(emb_path), allow_pickle=False) as f:
        frame_max = f['frame_idx'].max()
        frame_count = len(f['frame_idx'])
        video_length_sec_60fps = (frame_max + 1) / 60
        
        # Convert to 4 Hz
        n_4hz = (frame_max + 1) // 15
        video_length_sec_4hz = n_4hz / 4
        
        print(f"{sess}: {frame_count} frames, max_idx={frame_max} ({video_length_sec_60fps:.1f}s @ 60fps, {n_4hz} @ 4Hz, {video_length_sec_4hz:.1f}s)")
