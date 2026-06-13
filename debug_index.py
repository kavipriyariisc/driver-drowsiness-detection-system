#!/usr/bin/env python
"""Debug why session window index is empty."""
import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from src.models.m6_train import (
    build_session_window_index, EmbeddingCache, PROCESSED_DIR, 
    EMB_DIR, ALL_SUBJECTS, FOLDS_TEST, NO_TELEMETRY, AWAKE_ONLY
)

fold_path = PROCESSED_DIR / "fold_0.npz"
with np.load(str(fold_path), allow_pickle=False) as f:
    test_subjects = list(f['test_subjects'])

train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]

print(f"Fold 0 test subjects: {test_subjects}")
print(f"Fold 0 train subjects: {train_subjects}")
print(f"NO_TELEMETRY: {NO_TELEMETRY}")
print(f"AWAKE_ONLY: {AWAKE_ONLY}")
print()

cache = EmbeddingCache(EMB_DIR)

# Check each subject
for subj in train_subjects:
    print(f"Subject {subj}:")
    if subj in NO_TELEMETRY:
        print(f"  >> Filtered out (in NO_TELEMETRY)")
        continue
    for sess in ("A", "D"):
        if subj in AWAKE_ONLY and sess == "D":
            print(f"  {sess}: >> Filtered out (awake only)")
            continue
        has_emb = cache.has(subj, sess)
        print(f"  {sess}: has_embedding={has_emb}")

print()
print("Building index for train subjects...")
index = build_session_window_index(train_subjects, cache)
print(f"Index length: {len(index)}")
if index:
    print(f"First few entries: {index[:3]}")
else:
    print("Index is EMPTY - investigating further...")
    print()
    print("Manual check:")
    for subj in train_subjects:
        if subj in NO_TELEMETRY:
            continue
        for sess in ("A", "D"):
            if subj in AWAKE_ONLY and sess == "D":
                continue
            has_emb = cache.has(subj, sess)
            if has_emb:
                print(f"  {subj}_{sess}: YES (should be in index)")
