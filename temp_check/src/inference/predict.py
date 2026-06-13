"""
Inference Pipeline — UL-DD Multimodal Drowsiness Detection
===========================================================

ULDDPredictor:
    Loads a trained M1 / M2 / M3 Keras model and provides
    windowed inference on numpy arrays or live ring-buffer output.

Usage:
    predictor = ULDDPredictor('models/checkpoints/M3_fold0.keras', model_type='m3')
    probs = predictor.predict_window(fau_window, tele_window)  # → (3,) probabilities
    label = predictor.predict_class(fau_window, tele_window)   # → 0 | 1 | 2
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.architecture import (
    WINDOW, N_FAU, N_TELE, N_RT_FEAT, N_CLASSES,
    CLASS_NAMES, CrossModalAttentionBlock, M3FusionModel,
)

# Register custom objects needed for loading M3
CUSTOM_OBJECTS = {
    'CrossModalAttentionBlock': CrossModalAttentionBlock,
    'M3FusionModel': M3FusionModel,
}


class ULDDPredictor:
    """
    Wraps a trained Keras drowsiness model for inference.

    Supports:
        M1 — single input: fau_window  (240, 30)
        M2 — single input: tele_window (240,  5)
        M3 — dual input  : [fau_window, tele_window]
        M4 — single input: rt_window   (240, 10)
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        model_path  : str,
        model_type  : str   = 'm3',    # 'm1' | 'm2' | 'm3' | 'm4'
        fau_mean    : Optional[np.ndarray] = None,
        fau_std     : Optional[np.ndarray] = None,
        tele_mean   : Optional[np.ndarray] = None,
        tele_std    : Optional[np.ndarray] = None,
    ):
        """
        Args:
            model_path  : Path to .keras or .h5 checkpoint.
            model_type  : Which model architecture ('m1'/'m2'/'m3'/'m4').
            fau_mean/std, tele_mean/std : Per-feature z-score scalers
                (saved in fold .npz as 'fau_mean', 'fau_std', etc.).
                Pass None to skip normalisation (if data pre-normalised).
        """
        self.model_type = model_type.lower()
        self.fau_mean   = fau_mean
        self.fau_std    = fau_std
        self.tele_mean  = tele_mean
        self.tele_std   = tele_std

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        if self.model_type == 'm5':
            from ultralytics import YOLO as _YOLO
            self.model = _YOLO(str(path))
            # Build canonical name→scorer-index map once
            # YOLO assigns names alphabetically: Alert=0, Drowsy=1, LowVigilant=2
            # Scorer expects: Alert=0, LowVigilant=1, Drowsy=2
            _yolo_names = self.model.names           # {0:'Alert', 1:'Drowsy', 2:'LowVigilant'}
            _scorer_order = CLASS_NAMES              # ['Alert','LowVigilant','Drowsy']
            self._m5_remap = [
                next(k for k, v in _yolo_names.items() if v == name)
                for name in _scorer_order
            ]
        else:
            self.model = tf.keras.models.load_model(
                str(path),
                custom_objects=CUSTOM_OBJECTS,
                compile=False,   # skip custom loss fn — inference only
            )
        print(f"✓ Loaded {model_type.upper()} model: {path.name}")

    # ──────────────────────────────────────────────────────────────────────
    # Normalisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _norm_fau(self, x: np.ndarray) -> np.ndarray:
        if self.fau_mean is not None and self.fau_std is not None:
            return ((x - self.fau_mean) / self.fau_std).astype(np.float32)
        return x.astype(np.float32)

    def _norm_tele(self, x: np.ndarray) -> np.ndarray:
        if self.tele_mean is not None and self.tele_std is not None:
            return ((x - self.tele_mean) / self.tele_std).astype(np.float32)
        return x.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Core inference
    # ──────────────────────────────────────────────────────────────────────

    def predict_window(
        self,
        fau_window  : Optional[np.ndarray] = None,   # (240, 30) or (1, 240, 30)
        tele_window : Optional[np.ndarray] = None,   # (240,  5) or (1, 240,  5)
        rt_window   : Optional[np.ndarray] = None,   # (240, 10) or (1, 240, 10)
    ) -> np.ndarray:
        """
        Run inference on one window.

        Returns:
            np.ndarray (3,)  — softmax probabilities for [Alert, LowVigilant, Drowsy]
        """
        def _expand(arr):
            return arr[np.newaxis] if arr.ndim == 2 else arr

        m = self.model_type
        if m == 'm1':
            assert fau_window is not None
            x = self._norm_fau(_expand(fau_window))
            probs = self.model.predict(x, verbose=0)[0]

        elif m == 'm2':
            assert tele_window is not None
            x = self._norm_tele(_expand(tele_window))
            probs = self.model.predict(x, verbose=0)[0]

        elif m == 'm3':
            assert fau_window is not None and tele_window is not None
            xf = self._norm_fau(_expand(fau_window))
            xt = self._norm_tele(_expand(tele_window))
            probs = self.model.predict([xf, xt], verbose=0)[0]

        elif m == 'm4':
            assert rt_window is not None
            x = _expand(rt_window).astype(np.float32)
            probs = self.model.predict(x, verbose=0)[0]

        elif m == 'm5':
            raise ValueError(
                "M5 is an image-based model. Use predict_image(img) instead."
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        return probs  # shape (3,)

    def predict_image(self, img: np.ndarray) -> np.ndarray:
        """
        Run M5 (YOLOv8 classification) inference on a single BGR image.

        Args:
            img : np.ndarray  — BGR face crop (any size; YOLO resizes internally)

        Returns:
            np.ndarray (3,) — probabilities in scorer order [Alert, LowVigilant, Drowsy]
        """
        if self.model_type != 'm5':
            raise ValueError("predict_image is only for M5.")
        results = self.model.predict(img, verbose=False)
        raw_probs = results[0].probs.data.cpu().numpy()  # shape (num_classes,)
        # Reorder to scorer convention: [Alert, LowVigilant, Drowsy]
        probs = np.array([raw_probs[i] for i in self._m5_remap], dtype=np.float32)
        return probs

    def predict_class(self, **kwargs) -> int:
        """Return the predicted class index (0/1/2)."""
        return int(np.argmax(self.predict_window(**kwargs)))

    def predict_label(self, **kwargs) -> str:
        """Return the predicted class name string."""
        return self.CLASS_NAMES[self.predict_class(**kwargs)]

    # ──────────────────────────────────────────────────────────────────────
    # Batch inference on fold data
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_fold(
        self,
        fold_path   : str,
        fold_idx    : int = 0,
    ) -> dict:
        """
        Load a fold .npz and evaluate the model on the test split.

        Returns:
            dict with accuracy, macro_f1, per_class_f1, report
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.helpers import compute_metrics

        fold = dict(np.load(fold_path, allow_pickle=False))
        m    = self.model_type

        if m == 'm1':
            X_te = self._norm_fau(fold['X_fau_test'])
            y_te = fold['y_test']
            probs = self.model.predict(X_te, verbose=0)

        elif m == 'm2':
            X_te = self._norm_tele(fold['mm_tele_test'])
            y_te = fold['mm_y_test']
            probs = self.model.predict(X_te, verbose=0)

        elif m == 'm3':
            Xf = self._norm_fau(fold['mm_fau_test'])
            Xt = self._norm_tele(fold['mm_tele_test'])
            y_te = fold['mm_y_test']
            probs = self.model.predict([Xf, Xt], verbose=0)

        else:
            raise NotImplementedError(f"evaluate_fold not supported for {m}")

        y_pred = np.argmax(probs, axis=1)
        return compute_metrics(y_te, y_pred)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ──────────────────────────────────────────────────────────────────────────────

def load_predictor_from_fold(
    fold_dir    : str,
    fold_idx    : int,
    model_name  : str,   # e.g. 'M1', 'M3'
    checkpoint_dir: str = 'models/checkpoints',
) -> ULDDPredictor:
    """
    Load a predictor with the scalers saved in the fold .npz.

    Looks for:
        {checkpoint_dir}/{model_name}_fold{fold_idx}.keras
        {fold_dir}/fold_{fold_idx}.npz  → fau_mean, fau_std, tele_mean, tele_std
    """
    fold     = dict(np.load(
        str(Path(fold_dir) / f"fold_{fold_idx}.npz"), allow_pickle=False
    ))
    ckpt     = Path(checkpoint_dir) / f"{model_name}_fold{fold_idx}.keras"
    mt       = model_name.lower()  # 'm1', 'm2', 'm3'

    return ULDDPredictor(
        model_path = str(ckpt),
        model_type = mt,
        fau_mean   = fold.get('fau_mean'),
        fau_std    = fold.get('fau_std'),
        tele_mean  = fold.get('tele_mean'),
        tele_std   = fold.get('tele_std'),
    )
