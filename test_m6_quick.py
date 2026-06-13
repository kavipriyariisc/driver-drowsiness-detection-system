#!/usr/bin/env python
"""M6 Quick Test - Models, Forward Pass, Dataset"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.models.m6_vision_only import build_m6a, count_parameters as count_m6a_params
from src.models.m6_fusion import build_m6, count_parameters
from src.models.m6_train import (
    EmbeddingCache, M6Dataset, build_session_window_index,
    PROCESSED_DIR, EMB_DIR, FOLDS_TEST, ALL_SUBJECTS, NO_TELEMETRY, AWAKE_ONLY
)

print("="*70)
print("M6 MODELS QUICK TEST")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load fold 0
fold_path = PROCESSED_DIR / "fold_0.npz"
fold_data = np.load(str(fold_path))
mm_tele_train = fold_data["mm_tele_train"]
mm_y_train = fold_data["mm_y_train"]

print(f"Data: {mm_tele_train.shape[0]} train samples\n")

# Build & test models
print("MODEL COMPARISON:")
m6a = build_m6a().to(device)
m6_lite = build_m6('lite').to(device)
m6_full = build_m6('full').to(device)

print(f"  M6-A (Vision-only):  {count_m6a_params(m6a):,} params")
print(f"  M6-Lite (Fusion):    {count_parameters(m6_lite):,} params")
print(f"  M6-Full (Full):      {count_parameters(m6_full):,} params")

# Forward pass test
print("\nFORWARD PASS TEST:")
vis = torch.randn(4, 16, 512).to(device)
can = torch.randn(4, 240, 5).to(device)

out_a = m6a(vis)
out_lite = m6_lite(vis, can)
out_full = m6_full(vis, can)

print(f"  M6-A:    {tuple(out_a.shape)}")
print(f"  M6-Lite: {tuple(out_lite.shape)}")
print(f"  M6-Full: {tuple(out_full.shape)}")

# Embedding cache
print("\nEMBEDDING CACHE:")
cache = EmbeddingCache(EMB_DIR)
available = cache.available_sessions()
print(f"  Available: {len(available)}/32 sessions")

missing = []
for subj in ALL_SUBJECTS:
    if subj in NO_TELEMETRY:
        continue
    for sess in ("A", "D"):
        if subj in AWAKE_ONLY and sess == "D":
            continue
        if not cache.has(subj, sess):
            missing.append(f"{subj}_{sess}")
print(f"  Missing: {missing if missing else 'None'}")

# Dataset
print("\nDATASET:")
fold_subjects = FOLDS_TEST[0]
index = build_session_window_index(fold_subjects, cache)
dataset = M6Dataset(
    mm_tele=mm_tele_train, mm_y=mm_y_train, fold_subjects=fold_subjects,
    emb_cache=cache, t_vis=16
)
vis_s, can_s, y_s = dataset[0]
print(f"  Samples: {len(dataset)}")
print(f"  Shapes: vis{tuple(vis_s.shape)}, can{tuple(can_s.shape)}")

print("\n" + "="*70)
print("SUCCESS: ALL TESTS PASSED")
print("="*70)
