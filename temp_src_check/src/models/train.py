"""
Training Pipeline — UL-DD Multimodal Drowsiness Detection
==========================================================

Provides:
    compile_model      : attach optimiser, loss, and metrics to any model
    make_callbacks     : EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
    get_class_weights  : compute balanced class weights for imbalanced KSS labels
    train_fold         : train M1 / M2 / M3 on one pre-built fold dict
    cross_validate     : iterate all 5 folds, return per-fold + aggregated metrics

Usage (in notebook):
    from models.train import cross_validate
    from models.architecture import build_m1_facial_bilstm

    results = cross_validate(
        model_fn   = build_m1_facial_bilstm,
        model_name = 'M1',
        fold_dir   = 'datasets/processed/ul_dd',
        input_keys = ('X_fau_train', 'X_fau_test'),
        label_keys = ('y_train', 'y_test'),
        save_dir   = 'models/checkpoints',
    )
"""

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.architecture import N_CLASSES, CLASS_NAMES
from utils.helpers import compute_metrics

# ─── Default hyper-parameters ─────────────────────────────────────────────────
DEFAULTS = dict(
    epochs           = 200,   # more headroom; early stopping handles overfitting
    batch_size       = 32,
    learning_rate    = 5e-4,  # finer steps → better convergence on small datasets
    es_patience      = 25,    # EarlyStopping — was 15, now 25 to avoid premature stop
    lr_patience      = 8,     # ReduceLROnPlateau patience
    lr_factor        = 0.5,
    min_lr           = 1e-6,
    label_smoothing  = 0.1,   # reduces overconfidence on noisy KSS labels
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights from label array.
    Handles the Alert-heavy imbalance in UL-DD KSS labels.
    """
    classes = np.arange(N_CLASSES)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _smooth_sparse_cce(label_smoothing: float = 0.1, n_classes: int = N_CLASSES):
    """
    SparseCategoricalCrossentropy with label smoothing.
    SparseCCE doesn't natively support label_smoothing, so we one-hot
    inside the loss fn — compatible with class_weight in model.fit().
    """
    smooth = float(label_smoothing)
    nc     = int(n_classes)

    def _loss(y_true, y_pred):
        y_int  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh   = tf.one_hot(y_int, nc)                          # (N, C)
        y_sm   = y_oh * (1.0 - smooth) + smooth / float(nc)    # smoothed
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_sm, y_pred, from_logits=False
            )
        )
    _loss.__name__ = f'smooth_sparse_cce_{smooth}'
    return _loss


def compile_model(
    model: tf.keras.Model,
    learning_rate   : float = DEFAULTS['learning_rate'],
    label_smoothing : float = DEFAULTS['label_smoothing'],
) -> tf.keras.Model:
    """Attach Adam optimiser, smoothed CCE loss, and accuracy metric."""
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate),
        loss      = _smooth_sparse_cce(label_smoothing, N_CLASSES),
        metrics   = ['accuracy'],
    )
    return model


def make_callbacks(
    checkpoint_path : str,
    es_patience     : int   = DEFAULTS['es_patience'],
    lr_patience     : int   = DEFAULTS['lr_patience'],
    lr_factor       : float = DEFAULTS['lr_factor'],
    min_lr          : float = DEFAULTS['min_lr'],
) -> List[tf.keras.callbacks.Callback]:
    """
    Standard callback stack:
        1. EarlyStopping         — monitor val_accuracy (mode=max)
        2. ReduceLROnPlateau     — halve LR when val_accuracy plateaus
        3. ModelCheckpoint       — save best val_accuracy weights

    NOTE: We monitor val_accuracy (not val_loss) because on subject-independent
    folds, the model quickly becomes overconfident-but-wrong, causing val_loss
    to diverge from epoch 1 while val_accuracy still improves.  Monitoring
    val_loss would always restore epoch-1 weights.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=es_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=lr_factor, mode='max',
            patience=lr_patience, min_lr=min_lr, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor='val_accuracy', mode='max',
            save_best_only=True, verbose=0,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    model          : tf.keras.Model,
    fold           : Dict[str, np.ndarray],
    input_keys     : Tuple[str, str],          # (train_key, test_key) — or list for M3
    label_keys     : Tuple[str, str],          # ('y_train', 'y_test')
    fold_idx       : int,
    model_name     : str        = 'model',
    save_dir       : str        = 'models/checkpoints',
    epochs         : int        = DEFAULTS['epochs'],
    batch_size     : int        = DEFAULTS['batch_size'],
    learning_rate  : float      = DEFAULTS['learning_rate'],
    es_patience    : int        = DEFAULTS['es_patience'],
    lr_patience    : int        = DEFAULTS['lr_patience'],
    validation_split: float     = 0.15,
) -> Dict:
    """
    Train one model on one fold.

    For M3 (dual-input), pass:
        input_keys = (['mm_fau_train', 'mm_tele_train'], ['mm_fau_test', 'mm_tele_test'])

    Returns:
        dict with keys: fold, history, metrics, train_time_s
    """
    y_tr_key, y_te_key = label_keys

    # ── Prepare inputs ──────────────────────────────────────────────────────
    def _get_input(keys):
        if isinstance(keys, (list, tuple)):
            return [fold[k] for k in keys]
        return fold[keys]

    X_train = _get_input(input_keys[0])
    X_test  = _get_input(input_keys[1])
    y_train = fold[y_tr_key]
    y_test  = fold[y_te_key]

    if y_train is None or len(y_train) == 0:
        print(f"  [Fold {fold_idx}] No data — skipping.")
        return {}

    # ── Class weights ────────────────────────────────────────────────────────
    cw = get_class_weights(y_train)
    print(f"  [Fold {fold_idx}] class_weights = {cw}")

    # ── Compile ──────────────────────────────────────────────────────────────
    compile_model(model, learning_rate)

    ckpt = Path(save_dir) / f"{model_name}_fold{fold_idx}.keras"
    cbs  = make_callbacks(str(ckpt), es_patience, lr_patience)

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs           = epochs,
        batch_size       = batch_size,
        validation_split = validation_split,
        class_weight     = cw,
        callbacks        = cbs,
        verbose          = 1,
    )
    train_time = time.time() - t0

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred)

    print(
        f"  [Fold {fold_idx}] "
        f"acc={metrics['accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}  "
        f"time={train_time:.0f}s"
    )
    return {
        'fold'         : fold_idx,
        'history'      : history.history,
        'metrics'      : metrics,
        # Flat copies for easy notebook access
        'accuracy'     : metrics['accuracy'],
        'macro_f1'     : metrics['macro_f1'],
        'y_true'       : y_test.tolist(),
        'y_pred'       : y_pred.tolist(),
        'train_time_s' : train_time,
        'checkpoint'   : str(ckpt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-fold cross-validation runner
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    model_fn    : Callable,
    model_name  : str,
    fold_dir    : str,
    input_keys  : Tuple,
    label_keys  : Tuple[str, str] = ('y_train', 'y_test'),
    save_dir    : str  = 'models/checkpoints',
    n_folds     : int  = 5,
    **train_kwargs,
) -> Dict:
    """
    Run subject-independent 5-fold cross-validation.

    Args:
        model_fn   : callable that returns a fresh tf.keras.Model
        model_name : short label for checkpoints ('M1', 'M2', 'M3')
        fold_dir   : directory containing fold_0.npz … fold_4.npz
        input_keys : (train_key, test_key) strings for M1/M2
                     or ([train_k1,train_k2], [test_k1,test_k2]) for M3
        label_keys : ('y_train', 'y_test') default; ('mm_y_train','mm_y_test') for M3 MM splits

    Returns:
        dict with per-fold results and summary statistics
    """
    fold_dir = Path(fold_dir)
    all_results = []

    print(f"\n{'='*60}")
    print(f"  Cross-Validation: {model_name}  ({n_folds} folds)")
    print(f"{'='*60}")

    for k in range(n_folds):
        npz_path = fold_dir / f"fold_{k}.npz"
        if not npz_path.exists():
            print(f"  [Fold {k}] fold_{k}.npz not found — skipping.")
            continue

        fold = dict(np.load(str(npz_path), allow_pickle=False))
        # Convert numpy str arrays back to lists where needed
        print(f"\n  Fold {k} | test_subjects = {fold.get('test_subjects', '?')}")

        model  = model_fn()
        result = train_fold(
            model       = model,
            fold        = fold,
            input_keys  = input_keys,
            label_keys  = label_keys,
            fold_idx    = k,
            model_name  = model_name,
            save_dir    = save_dir,
            **train_kwargs,
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("No folds completed.")
        return {}

    # ── Aggregate ─────────────────────────────────────────────────────────
    acc  = [r['metrics']['accuracy']  for r in all_results]
    f1   = [r['metrics']['macro_f1']  for r in all_results]

    summary = {
        'model'       : model_name,
        'folds'       : all_results,
        'mean_acc'    : float(np.mean(acc)),
        'std_acc'     : float(np.std(acc)),
        'mean_f1'     : float(np.mean(f1)),
        'std_f1'      : float(np.std(f1)),
    }

    print(f"\n{'─'*60}")
    print(f"  {model_name} Summary:")
    print(f"  Accuracy  : {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"  Macro F1  : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"{'─'*60}\n")

    return summary
