"""
Metrics, Plotting, and Reporting Utilities — UL-DD Project
===========================================================
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)

CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute accuracy, macro F1, per-class F1, and classification report.

    Returns:
        dict with keys: accuracy, macro_f1, per_class_f1, report
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    pc_f1    = f1_score(y_true, y_pred, average=None,
                        labels=[0, 1, 2], zero_division=0)
    report   = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES, zero_division=0
    )
    return {
        'accuracy'     : float(acc),
        'macro_f1'     : float(macro_f1),
        'per_class_f1' : pc_f1.tolist(),
        'report'       : report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true      : np.ndarray,
    y_pred      : np.ndarray,
    title       : str            = 'Confusion Matrix',
    normalise   : bool           = True,
    save_path   : Optional[str]  = None,
    ax          : Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a confusion matrix with seaborn heatmap.

    Args:
        normalise : If True, show row-normalised percentages.
        save_path : If provided, save figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    if normalise:
        cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt, vmax = '.2f', 1.0
    else:
        cm_disp = cm
        fmt, vmax = 'd', None

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_disp,
        annot=True, fmt=fmt,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap='Blues',
        vmin=0, vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Fold results summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_fold_results(
    results_dict : Dict,
    save_path    : Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of accuracy and macro-F1 per fold for one model.

    Args:
        results_dict : output of cross_validate()
    """
    folds   = results_dict.get('folds', [])
    if not folds:
        print("No fold data to plot.")
        return

    fold_ids = [r['fold'] for r in folds]
    accs     = [r['metrics']['accuracy']  for r in folds]
    f1s      = [r['metrics']['macro_f1']  for r in folds]

    x    = np.arange(len(fold_ids))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
    ax.bar(x + w/2, f1s,  w, label='Macro F1', color='coral')
    ax.axhline(np.mean(accs), color='steelblue', linestyle='--', alpha=0.6,
               label=f'Mean Acc={np.mean(accs):.3f}')
    ax.axhline(np.mean(f1s),  color='coral',     linestyle='--', alpha=0.6,
               label=f'Mean F1={np.mean(f1s):.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in fold_ids])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f"{results_dict.get('model', 'Model')} — 5-Fold CV Results")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_list : List[Dict],
    metric       : str          = 'macro_f1',
    save_path    : Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of M0-SVM, M1, M2, M3.

    Args:
        results_list : list of cross_validate() output dicts
        metric       : 'macro_f1' | 'accuracy'
    """
    names  = [r['model'] for r in results_list]
    means  = [r[f'mean_{metric.split("_")[-1]}' if 'f1' in metric else 'mean_acc']
              for r in results_list]
    stds   = [r[f'std_{metric.split("_")[-1]}'  if 'f1' in metric else 'std_acc']
              for r in results_list]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ['#888888', '#4C72B0', '#DD8452', '#55A868'][:len(names)]
    bars    = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                     edgecolor='black', linewidth=0.8)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Model Comparison — Subject-Independent 5-Fold CV')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KSS / label distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_kss_distribution(
    kss_values  : np.ndarray,
    title       : str           = 'KSS Distribution',
    save_path   : Optional[str] = None,
) -> plt.Figure:
    """Bar chart of raw KSS (1-9) counts."""
    fig, ax = plt.subplots(figsize=(8, 4))
    vals, counts = np.unique(kss_values, return_counts=True)
    colors = ['#2196F3' if v <= 3 else '#FF9800' if v <= 6 else '#F44336'
              for v in vals]
    ax.bar(vals, counts, color=colors, edgecolor='black', linewidth=0.7)
    ax.set_xlabel('KSS Score')
    ax.set_ylabel('Count')
    ax.set_xticks(range(1, 10))
    ax.set_title(title)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#2196F3', label='Alert (1-3)'),
        Patch(color='#FF9800', label='Low Vigilant (4-6)'),
        Patch(color='#F44336', label='Drowsy (7-9)'),
    ])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history     : Dict,
    title       : str           = 'Training History',
    save_path   : Optional[str] = None,
) -> plt.Figure:
    """Plot loss + accuracy curves from Keras history dict."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(history['loss'],     label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss'); ax1.legend()

    ax2.plot(history['accuracy'],     label='Train Acc')
    ax2.plot(history['val_accuracy'], label='Val Acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} — Accuracy'); ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Attention weight visualiser  (M3 interpretability)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_weights(
    attn_f2t    : np.ndarray,
    attn_t2f    : np.ndarray,
    sample_idx  : int           = 0,
    save_path   : Optional[str] = None,
) -> plt.Figure:
    """
    Visualise cross-modal attention matrices for one window.

    Args:
        attn_f2t : (batch, T_fau,  T_tele) — FAU→Tele attention
        attn_t2f : (batch, T_tele, T_fau)  — Tele→FAU attention
        sample_idx : which sample in the batch to plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    f2t = attn_f2t[sample_idx]  # (T_fau, T_tele)
    t2f = attn_t2f[sample_idx]  # (T_tele, T_fau)

    sns.heatmap(f2t, ax=ax1, cmap='YlOrRd', cbar=True)
    ax1.set_title('FAU → Telemetry Attention')
    ax1.set_xlabel('Telemetry timestep')
    ax1.set_ylabel('FAU timestep')

    sns.heatmap(t2f, ax=ax2, cmap='YlOrRd', cbar=True)
    ax2.set_title('Telemetry → FAU Attention')
    ax2.set_xlabel('FAU timestep')
    ax2.set_ylabel('Telemetry timestep')

    fig.suptitle('Cross-Modal Attention Weights (M3)', fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    return fig
