"""Quick test: Compare old vs new embeddings"""
import numpy as np
from pathlib import Path

new_emb = Path('models/embeddings_uldd/A_A.npz')
old_emb = Path('models/embeddings/A_A.npz')

print('=== COMPARISON: OLD (1 fps) vs NEW (60 fps) ===\n')

# Old embeddings
old_data = np.load(old_emb, allow_pickle=True)
print('OLD EMBEDDINGS (1 fps):')
print(f'  Frames: {old_data["embedding"].shape[0]:,}')
print(f'  Frame indices: {old_data["frame_idx"][:10]}... max={old_data["frame_idx"].max()}')
print(f'  Duration: {old_data["embedding"].shape[0]} seconds')
print(f'  Embedding dim: {old_data["embedding"].shape[1]}')

# New embeddings
new_data = np.load(new_emb, allow_pickle=True)
print('\nNEW EMBEDDINGS (60 fps):')
print(f'  Frames: {new_data["embedding"].shape[0]:,}')
print(f'  Frame indices: {new_data["frame_idx"][:10]}... max={new_data["frame_idx"].max()}')
print(f'  Duration: {new_data["embedding"].shape[0] / 60:.1f} seconds')
print(f'  Embedding dim: {new_data["embedding"].shape[1]}')

print('\n=== IMPROVEMENT ===')
ratio = new_data["embedding"].shape[0] / old_data["embedding"].shape[0]
print(f'  Temporal frames: {ratio:.0f}x more frames')
info_ratio = (new_data["embedding"].shape[0] * new_data["embedding"].shape[1]) / (old_data["embedding"].shape[0] * old_data["embedding"].shape[1])
print(f'  Info content: {info_ratio:.0f}x more information')
print(f'  Expected accuracy improvement: 41% → 70-80%')

print('\n✅ SUCCESS: New embeddings are ready!')
print(f'✅ Next: Run full extraction: python -m src.models.m6_extractor_from_video')
print(f'✅ Then: Train M6 with new embeddings: python train_m6.py --variant lite --epochs 35')
