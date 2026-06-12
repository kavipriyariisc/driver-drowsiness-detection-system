"""
M6 Data Validation Module
========================
Comprehensive checks before M6 training to ensure:
1. Fold files contain correct data
2. Embeddings cache is complete
3. Index reconstruction matches fold data
4. No silent failures with random data
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from .m6_train import (
    EmbeddingCache, build_session_window_index,
    ALL_SUBJECTS, AWAKE_ONLY, NO_TELEMETRY, FOLDS_TEST
)


def validate_fold_file(fold_path: Path, verbose: bool = True) -> dict:
    """
    Validate a fold .npz file has all required keys and correct structure.
    
    Returns:
        {
            'is_valid': bool,
            'n_train': int,
            'n_test': int,
            'train_keys': set,
            'test_keys': set,
            'issues': [str],
        }
    """
    issues = []
    
    if not fold_path.exists():
        return {
            'is_valid': False,
            'n_train': 0,
            'n_test': 0,
            'train_keys': set(),
            'test_keys': set(),
            'issues': [f"Fold file not found: {fold_path}"],
        }
    
    try:
        f = np.load(str(fold_path), allow_pickle=False)
        files = set(f.files)
    except Exception as e:
        return {
            'is_valid': False,
            'n_train': 0,
            'n_test': 0,
            'train_keys': set(),
            'test_keys': set(),
            'issues': [f"Failed to load fold file: {e}"],
        }
    
    # Check for required keys
    required_train_keys = {'mm_y_train', 'mm_tele_train'}
    required_test_keys = {'mm_y_test', 'mm_tele_test'}
    
    train_keys = files & required_train_keys
    test_keys = files & required_test_keys
    
    if len(train_keys) < len(required_train_keys):
        issues.append(f"Missing train keys: {required_train_keys - train_keys}")
    if len(test_keys) < len(required_test_keys):
        issues.append(f"Missing test keys: {required_test_keys - test_keys}")
    
    # Load and check shapes
    try:
        y_train = f['mm_y_train']
        y_test = f['mm_y_test']
        tele_train = f['mm_tele_train']
        tele_test = f['mm_tele_test']
        
        n_train = len(y_train)
        n_test = len(y_test)
        
        # Check alignment
        if len(tele_train) != n_train:
            issues.append(
                f"Train mismatch: {len(tele_train)} telemetry vs {n_train} labels"
            )
        if len(tele_test) != n_test:
            issues.append(
                f"Test mismatch: {len(tele_test)} telemetry vs {n_test} labels"
            )
        
        # Check value ranges
        if np.any((y_train < 0) | (y_train >= 3)):
            issues.append(f"Invalid train labels: not in [0, 1, 2]")
        if np.any((y_test < 0) | (y_test >= 3)):
            issues.append(f"Invalid test labels: not in [0, 1, 2]")
        
        # Check telemetry shape: N windows × 240 timesteps × 5 features
        if tele_train.ndim != 3 or tele_train.shape[2] != 5:
            issues.append(
                f"Train telemetry shape: {tele_train.shape}, expected (N, T, 5)"
            )
        if tele_test.ndim != 3 or tele_test.shape[2] != 5:
            issues.append(
                f"Test telemetry shape: {tele_test.shape}, expected (N, T, 5)"
            )
    except Exception as e:
        issues.append(f"Error checking fold data: {e}")
        n_train = 0
        n_test = 0
    
    f.close()
    
    is_valid = len(issues) == 0
    
    if verbose:
        if is_valid:
            print(f"✓ Fold valid: {n_train} train + {n_test} test samples")
        else:
            print(f"❌ Fold INVALID:")
            for issue in issues:
                print(f"   - {issue}")
    
    return {
        'is_valid': is_valid,
        'n_train': n_train,
        'n_test': n_test,
        'train_keys': train_keys,
        'test_keys': test_keys,
        'issues': issues,
    }


def validate_embeddings_cache(
    emb_dir: Path, 
    fold_subjects: List[str],
    verbose: bool = True
) -> dict:
    """
    Validate embeddings cache has all required sessions.
    
    Returns:
        {
            'is_complete': bool,
            'n_available': int,
            'n_expected': int,
            'missing': [str],
            'extra': [str],
        }
    """
    cache = EmbeddingCache(emb_dir)
    
    # What's expected?
    expected = set()
    for subj in fold_subjects:
        if subj in NO_TELEMETRY:
            continue  # Subject A has no telemetry
        for sess in ('A', 'D'):
            if subj in AWAKE_ONLY and sess == 'D':
                continue  # Subject C only has Alert
            expected.add(f"{subj}_{sess}")
    
    # What's actually cached?
    available = set(p.stem for p in emb_dir.glob("*.npz"))
    
    missing = expected - available
    extra = available - expected
    
    is_complete = len(missing) == 0
    
    if verbose:
        if is_complete:
            print(f"✓ Embeddings complete: {len(available)} sessions")
        else:
            print(f"❌ Embeddings INCOMPLETE:")
            print(f"   Missing ({len(missing)}): {sorted(missing)}")
            if extra:
                print(f"   Extra ({len(extra)}): {sorted(extra)}")
    
    return {
        'is_complete': is_complete,
        'n_available': len(available),
        'n_expected': len(expected),
        'missing': sorted(missing),
        'extra': sorted(extra),
    }


def validate_index_reconstruction(
    fold_path: Path,
    emb_dir: Path,
    fold_idx: int,
    verbose: bool = True
) -> dict:
    """
    Verify that index reconstruction matches fold data sizes.
    
    This checks the critical alignment between:
      1. Fold telemetry/label arrays (mm_tele_*, mm_y_*)
      2. Reconstructed window indices
      3. Available embeddings
    
    Returns:
        {
            'is_aligned': bool,
            'fold_train_size': int,
            'fold_test_size': int,
            'index_train_size': int,
            'index_test_size': int,
            'issues': [str],
        }
    """
    issues = []
    
    # Load fold file
    f = np.load(str(fold_path), allow_pickle=False)
    fold_train_size = len(f['mm_y_train'])
    fold_test_size = len(f['mm_y_test'])
    test_subjects = list(f['test_subjects'])
    f.close()
    
    # Get train/test subject lists
    all_subj = [s for s in ALL_SUBJECTS if s not in test_subjects]
    
    # Build indices
    cache = EmbeddingCache(emb_dir)
    try:
        index_train = build_session_window_index(
            all_subj, cache, target_size=fold_train_size
        )
        index_test = build_session_window_index(
            test_subjects, cache, target_size=fold_test_size
        )
    except Exception as e:
        return {
            'is_aligned': False,
            'fold_train_size': fold_train_size,
            'fold_test_size': fold_test_size,
            'index_train_size': 0,
            'index_test_size': 0,
            'issues': [f"Failed to build index: {e}"],
        }
    
    index_train_size = len(index_train)
    index_test_size = len(index_test)
    
    # Check alignment
    if index_train_size != fold_train_size:
        issues.append(
            f"Train size mismatch: "
            f"index={index_train_size} vs fold={fold_train_size}"
        )
    if index_test_size != fold_test_size:
        issues.append(
            f"Test size mismatch: "
            f"index={index_test_size} vs fold={fold_test_size}"
        )
    
    is_aligned = len(issues) == 0
    
    if verbose:
        if is_aligned:
            print(f"✓ Index aligned: "
                  f"train {fold_train_size}→{index_train_size}, "
                  f"test {fold_test_size}→{index_test_size}")
        else:
            print(f"❌ Index MISALIGNED:")
            for issue in issues:
                print(f"   - {issue}")
    
    return {
        'is_aligned': is_aligned,
        'fold_train_size': fold_train_size,
        'fold_test_size': fold_test_size,
        'index_train_size': index_train_size,
        'index_test_size': index_test_size,
        'issues': issues,
    }


def full_validation(
    fold_idx: int,
    processed_dir: Path,
    emb_dir: Path,
    verbose: bool = True
) -> Tuple[bool, dict]:
    """
    Run complete M6 data validation for a fold.
    
    Returns:
        (all_valid: bool, report: dict)
    """
    if verbose:
        print("=" * 70)
        print(f"M6 DATA VALIDATION — FOLD {fold_idx}")
        print("=" * 70)
        print()
    
    fold_path = processed_dir / f"fold_{fold_idx}.npz"
    test_subjects = FOLDS_TEST[fold_idx]
    train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]
    all_subjects = train_subjects + test_subjects

    # 1. Validate fold file
    if verbose:
        print(f"1️⃣  Checking fold file: {fold_path.name}")
    fold_check = validate_fold_file(fold_path, verbose=verbose)
    print()

    # 2. Validate embeddings cache for all subjects used by the fold
    if verbose:
        print(f"2️⃣  Checking embeddings cache:")
    emb_check = validate_embeddings_cache(emb_dir, all_subjects, verbose=verbose)
    # 3. Validate index reconstruction
    if verbose:
        print(f"3️⃣  Checking index reconstruction:")
    idx_check = validate_index_reconstruction(
        fold_path, emb_dir, fold_idx, verbose=verbose
    )
    print()
    
    all_valid = (
        fold_check['is_valid'] and
        emb_check['is_complete'] and
        idx_check['is_aligned']
    )
    
    report = {
        'fold_idx': fold_idx,
        'all_valid': all_valid,
        'fold_check': fold_check,
        'emb_check': emb_check,
        'idx_check': idx_check,
    }
    
    if verbose:
        if all_valid:
            print("=" * 70)
            print("✅ ALL CHECKS PASSED — READY TO TRAIN")
            print("=" * 70)
        else:
            print("=" * 70)
            print("❌ VALIDATION FAILED — FIX BEFORE TRAINING")
            print("=" * 70)
        print()
    
    return all_valid, report
