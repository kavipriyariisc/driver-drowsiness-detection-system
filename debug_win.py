import numpy as np
from pathlib import Path
import sys

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'src'))

from models.m6_train import (
    ALL_SUBJECTS, AWAKE_ONLY, NO_TELEMETRY,
    EmbeddingCache, _windows_per_session,
    DOWNSAMPLE, STRIDE_SEC, TARGET_HZ, T_CAN, WIN_FRAMES_60HZ
)

EMB_DIR = ROOT / "models" / "embeddings"
cache = EmbeddingCache(EMB_DIR)

print("Available embeddings:")
for subj in sorted(ALL_SUBJECTS):
    for sess in ("A", "D"):
        if cache.has(subj, sess):
            try:
                rec = cache.load(subj, sess)
                n_frames = int(rec["frame_idx"].max()) + 1
                n_windows = _windows_per_session(subj, sess, cache)
                print(f"  {subj}_{sess}: {n_frames} frames @ 60fps, {n_windows} windows @ 4Hz")
            except Exception as e:
                print(f"  {subj}_{sess}: Error - {e}")

print(f"\nConstants: DOWNSAMPLE={DOWNSAMPLE}, STRIDE_SEC={STRIDE_SEC}, T_CAN={T_CAN}, WIN_FRAMES_60HZ={WIN_FRAMES_60HZ}")
