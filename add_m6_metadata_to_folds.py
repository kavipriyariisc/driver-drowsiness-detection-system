# M6 Pipeline Fix - Step 3: Add Metadata to Fold Files
# 
# STRATEGY: Retroactively add M6 metadata to existing fold_*.npz files.
# This is safer than modifying the preprocessing pipeline.
#
# WHAT THIS DOES:
# - Load existing fold_*.npz files  
# - Create subject/session/win_idx/frame metadata for each multimodal sample
# - Drop samples from subjects with missing embeddings
# - Save new fold_*.npz with added keys
#
# KEY INSIGHT:
# Each multimodal sample corresponds to a 60-second window in a subject/session.
# We can reconstruct which window each sample belongs to from its index.
#
# Fold structure (currently):
#   mm_y_train / mm_y_test - shape (N,)
#   mm_tele_train / mm_tele_test - shape (N, 240, 5)
#
# We need to track:
#   mm_subject_train - shape (N,)  subject ID
#   mm_session_train - shape (N,)  "A" or "D"
#   mm_win_idx_train - shape (N,)  window index within session
#   mm_start_4hz_train - shape (N,)  CAN frame start (@ 4Hz)
#   mm_end_4hz_train - shape (N,)  CAN frame end (@ 4Hz)

import numpy as np
from pathlib import Path
from typing import List, Set, Tuple

ROOT = Path(__file__).resolve().parents[0]
PROCESSED_DIR = ROOT / "datasets" / "ul_dd"
EMB_DIR = ROOT / "models" / "embeddings_uldd"

# Constants
SUBJECTS = list("ABCDEFGHIJKLMNOPQRS")
WINDOW_SEC = 60
STRIDE_SEC = 15
TARGET_HZ = 4

def get_available_sessions() -> Set[str]:
    """List which subject/session embeddings exist."""
    available = set()
    if EMB_DIR.exists():
        for npz_file in EMB_DIR.glob("*.npz"):
            session = npz_file.stem  # "A_A", "B_D", etc.
            available.add(session)
    return available

def create_mm_metadata(n_samples: int, test_subjects: List[str], available_sessions: Set[str]):
    """
    Create metadata for multimodal samples.
    
    Strategy:
    - We have n_samples total multimodal windows across train+test
    - Test samples come from test_subjects, train from everyone else
    - For each subject/session that has telemetry, compute windows
    - Map samples to subject/session/window indices
    
    Returns: (subject_arr, session_arr, win_idx_arr, start_4hz_arr, end_4hz_arr, keep_indices)
    """
    subjects_list = []
    sessions_list = []
    win_indices_list = []
    start_4hz_list = []
    end_4hz_list = []
    keep_indices = []
    
    # Iterate through subjects and sessions
    # For M3/M6: sessions are A (Alert) and D (Drowsy)
    sample_count = 0
    
    for subject in SUBJECTS:
        for session in ['A', 'D']:
            session_name = f"{subject}_{session}"
            has_embedding = session_name in available_sessions
            
            # Estimate number of windows in a session
            # Typical: 60min video → ~60s windows @ 15s stride → ~240 windows
            # But we don't know exact window count, so we estimate
            # Actually: we'll build this after loading to match actual array
            pass
    
    # This approach is too blind. Instead, we'll use a different strategy:
    # Load one fold, see its structure, then replicate metadata based on
    # subject/session assignment in the preprocessing.
    
    # For now, return placeholder to indicate this needs manual intervention
    return None

def add_metadata_to_folds_simple():
    """
    Simpler approach: For each fold, add minimal metadata that M6Dataset needs.
    
    M6Dataset minimum requirement:
    - Know which subject/session each sample comes from
    - Compute window index from sample position
    """
    
    for fold_idx in range(5):
        fold_path = PROCESSED_DIR / f"fold_{fold_idx}.npz"
        if not fold_path.exists():
            print(f"✗ Fold {fold_idx} not found")
            continue
        
        print(f"\nLoading fold_{fold_idx}.npz...")
        data = np.load(str(fold_path), allow_pickle=True)
        fold = dict(data)
        data.close()
        
        # For now, add a note that metadata needs to be added
        # The proper way is to regenerate folds with metadata built-in
        
        print(f"  Current keys: {list(fold.keys())}")
        print(f"  mm_y_train shape: {fold['mm_y_train'].shape if fold['mm_y_train'] is not None else None}")
        print(f"  mm_y_test shape: {fold['mm_y_test'].shape if fold['mm_y_test'] is not None else None}")
        
        # Check if metadata already added
        if 'mm_subject_train' in fold:
            print(f"  ✓ Metadata already present")
            continue
        else:
            print(f"  ⚠ Metadata NOT present - needs to be added")

print("\n" + "="*70)
print("M6 Metadata Addition Tool")
print("="*70)
print("\nNOTE: This tool requires preprocessing to be modified to track metadata.")
print("The proper fix is to modify src/data/preprocess.py to save metadata")
print("when building folds.")
print("\nFor now, showing current fold structure:\n")

add_metadata_to_folds_simple()

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("""
The retroactive approach is complex because we've lost the mapping of
which sample belongs to which (subject, session, window).

PROPER FIX:
1. Modify src/data/preprocess.py build_folds() to track metadata:
   - For each multimodal sample, store which subject/session it came from
   - Store window index (0, 1, 2, ...)
   
2. Add these fields to fold dict before saving:
   - mm_subject_train, mm_session_train, mm_win_idx_train
   - mm_subject_test, mm_session_test, mm_win_idx_test
   - mm_start_4hz_train, mm_end_4hz_train
   - mm_start_4hz_test, mm_end_4hz_test
   
3. Rerun preprocessing notebook to generate new fold files with metadata

This ensures M6Dataset can properly align visual embeddings with CAN windows.
""")
