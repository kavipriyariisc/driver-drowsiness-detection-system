"""
Load Data Utilities — UL-DD Project
====================================
Thin wrappers around ULDDProcessor for notebook convenience.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_fold(
    fold_dir : str,
    fold_idx : int,
) -> Dict[str, np.ndarray]:
    """
    Load a pre-processed fold .npz as a plain dict.

    Returns dict with keys:
        M1 : X_fau_train, X_fau_test, y_train, y_test
        M3 : mm_fau_train, mm_tele_train, mm_y_train,
             mm_fau_test,  mm_tele_test,  mm_y_test
        Scalers: fau_mean, fau_std, tele_mean, tele_std
    """
    path = Path(fold_dir) / f"fold_{fold_idx}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"fold_{fold_idx}.npz not found in {fold_dir}\n"
            "Run notebook 05-uldd-preprocessing.ipynb first."
        )
    return dict(np.load(str(path), allow_pickle=False))


def load_all_folds(fold_dir: str, n_folds: int = 5) -> List[Dict]:
    """Load all fold .npz files and return as a list of dicts."""
    return [load_fold(fold_dir, k) for k in range(n_folds)]


def summarise_fold(fold: Dict) -> None:
    """Print a quick summary of a loaded fold dict."""
    def _shape(k):
        v = fold.get(k)
        return tuple(v.shape) if v is not None else 'N/A'

    print("─" * 50)
    print("M1  (FAU-only, all sessions):")
    print(f"  X_fau_train  : {_shape('X_fau_train')}")
    print(f"  X_fau_test   : {_shape('X_fau_test')}")
    print(f"  y_train      : {_shape('y_train')}")
    print(f"  y_test       : {_shape('y_test')}")
    print("M2/M3  (FAU + Tele, matched sessions):")
    print(f"  mm_fau_train : {_shape('mm_fau_train')}")
    print(f"  mm_tele_train: {_shape('mm_tele_train')}")
    print(f"  mm_y_train   : {_shape('mm_y_train')}")
    print(f"  mm_fau_test  : {_shape('mm_fau_test')}")
    print(f"  mm_tele_test : {_shape('mm_tele_test')}")
    print(f"  mm_y_test    : {_shape('mm_y_test')}")
    print("─" * 50)


def class_distribution(y: np.ndarray, class_names=('Alert', 'LowVigilant', 'Drowsy')) -> str:
    """Return formatted class distribution string."""
    counts = np.bincount(y.astype(int), minlength=3)
    total  = counts.sum()
    parts  = [f"{name}={counts[i]} ({counts[i]/total*100:.1f}%)"
              for i, name in enumerate(class_names)]
    return '  |  '.join(parts)
