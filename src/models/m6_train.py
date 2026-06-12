"""
M6 — Window-aligned Visual + CAN Dataset and Training Loop
==========================================================
Bridges the cached YOLOv8 embeddings (per-session, frame-indexed) with
the existing CAN telemetry windows in `datasets/processed/ul_dd/fold_*.npz`,
then trains the M6 fusion models with subject-independent 5-fold CV.

Window alignment (matches ULDDProcessor in `src/data/preprocess.py`):
    Original video : 60 fps
    Downsampled    : 4 Hz   (factor 15)
    Window length  : 60 s   → 240 timesteps @ 4 Hz  =  3600 frames @ 60 fps
    Window stride  : 15 s   →  60 timesteps @ 4 Hz  =   900 frames @ 60 fps

So window i of a session covers frame indices  [i*900 : i*900 + 3600]
in the original 60-fps video.  We sample T_VIS=16 frames uniformly from
that range and look up their pre-computed YOLOv8 embeddings.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from .m6_fusion import (CLASS_NAMES, M6_Full, M6_Lite, build_m6,
                        count_parameters)

# ─── Project paths ────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "datasets" / "processed" / "ul_dd"
EMB_DIR       = ROOT / "models"   / "embeddings"
CKPT_DIR      = ROOT / "models"   / "checkpoints"
REPORT_DIR    = ROOT / "results"  / "reports"

# ─── Window geometry (must mirror ULDDProcessor) ──────────────────────────────
ORIG_FPS         = 60
TARGET_HZ        = 4
DOWNSAMPLE       = ORIG_FPS // TARGET_HZ          # 15
WINDOW_SEC       = 60
STRIDE_SEC       = 15
T_CAN            = WINDOW_SEC * TARGET_HZ          # 240
WIN_FRAMES_60HZ  = WINDOW_SEC * ORIG_FPS           # 3600
STRIDE_FRAMES    = STRIDE_SEC * ORIG_FPS           #  900

# UL-DD subject layout (mirrors ULDDProcessor.FOLDS exactly)
FOLDS_TEST = [
    list("ABCD"),   # Fold 0
    list("EFGH"),   # Fold 1
    list("IJK"),    # Fold 2
    list("LMNO"),   # Fold 3
    list("PQRS"),   # Fold 4
]
ALL_SUBJECTS = list("ABCDEFGHIJKLMNOPQRS")
AWAKE_ONLY   = frozenset({"C", "F", "L"})    # no Drowsy session
NO_TELEMETRY = frozenset({"A"})              # telemetry CSV missing


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache loader
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingCache:
    """
    Lazy loader for per-session YOLOv8 embedding `.npz` files produced by
    `src/models/m6_extractor.py`.

    Each session is loaded once, kept in RAM, and provides O(1) access to
    the embedding for any *original* 60-fps frame index via a sparse map.
    """

    def __init__(self, emb_dir: Path = EMB_DIR):
        self.emb_dir = emb_dir
        self._cache: Dict[str, dict] = {}

    def has(self, subject: str, session: str) -> bool:
        return (self.emb_dir / f"{subject}_{session}.npz").is_file()

    def available_sessions(self) -> List[str]:
        return [p.stem for p in sorted(self.emb_dir.glob("*.npz"))]

    def load(self, subject: str, session: str) -> dict:
        key = f"{subject}_{session}"
        if key in self._cache:
            return self._cache[key]
        path = self.emb_dir / f"{key}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding cache missing: {path}.  "
                "Run `python -m src.models.m6_extractor` first."
            )
        arr = np.load(str(path), allow_pickle=False)
        frame_idx = arr["frame_idx"].astype(np.int64)
        emb       = arr["embedding"].astype(np.float32)
        # Build a frame_idx → row index lookup for fast sampling
        order     = np.argsort(frame_idx)
        record = {
            "frame_idx": frame_idx[order],
            "embedding": emb[order],
            "embed_dim": emb.shape[1],
        }
        self._cache[key] = record
        return record

    @property
    def embed_dim(self) -> int:
        for rec in self._cache.values():
            return rec["embed_dim"]
        # fall back: peek at any file
        for p in sorted(self.emb_dir.glob("*.npz")):
            with np.load(str(p)) as arr:
                return int(arr["embedding"].shape[1])
        raise RuntimeError("No embedding cache files available.")


def sample_clip_embedding(rec: dict, win_idx: int, t_vis: int = 16) -> np.ndarray:
    """
    Return a (t_vis, embed_dim) clip for window `win_idx` of one session.

    Sampling strategy
    -----------------
        • Compute the original 60-fps frame range covered by the window:
              [win_idx * STRIDE_FRAMES,  win_idx * STRIDE_FRAMES + WIN_FRAMES_60HZ]
        • Find the cached frames whose frame_idx falls in that range
          (these are the frames YOLO actually classified for this session).
        • If at least one matching frame exists → pick `t_vis` of them
          uniformly along the time axis (with replacement if necessary).
        • If NO matching frames exist (rare gap) → fall back to the
          temporally nearest cached frames overall.
    """
    fi  = rec["frame_idx"]
    emb = rec["embedding"]
    lo  = win_idx * STRIDE_FRAMES
    hi  = lo + WIN_FRAMES_60HZ
    sel = np.where((fi >= lo) & (fi < hi))[0]

    if sel.size == 0:
        # Fallback: nearest cached frames around the window centre
        centre = (lo + hi) // 2
        order  = np.argsort(np.abs(fi - centre))[: max(t_vis, 1)]
        sel    = np.sort(order)

    # Uniform sampling (with replacement if too few)
    if sel.size >= t_vis:
        idx = np.linspace(0, sel.size - 1, t_vis).round().astype(np.int64)
    else:
        idx = np.linspace(0, sel.size - 1, t_vis).astype(np.int64)
    rows = sel[idx]
    return emb[rows]                                       # (t_vis, D)


# ─────────────────────────────────────────────────────────────────────────────
# Recover per-session window count + telemetry slices from a fold .npz
# ─────────────────────────────────────────────────────────────────────────────
def _windows_per_session(subject: str, session: str,
                         emb_cache: EmbeddingCache) -> int:
    """
    Estimate how many windows a session contributes to the multimodal
    fold arrays. We use the highest cached frame_idx as a proxy for
    session length (in 60-fps frames) and apply the same sliding-window
    arithmetic as ULDDProcessor.
    """
    rec   = emb_cache.load(subject, session)
    n_60  = int(rec["frame_idx"].max()) + 1            # rough video length
    n_4   = n_60 // DOWNSAMPLE                          # 4-Hz samples
    if n_4 < T_CAN:
        return 0
    return (n_4 - T_CAN) // (STRIDE_SEC * TARGET_HZ) + 1


def _allocate_session_counts(counts: List[int], target_size: int) -> List[int]:
    """Distribute counts to exactly match a target total while preserving order."""
    total = sum(counts)
    if total == target_size or total == 0:
        return counts.copy()

    quotas = [c * target_size / total for c in counts]
    floor_counts = [int(np.floor(q)) for q in quotas]
    remainder = target_size - sum(floor_counts)
    fractions = sorted(
        [(q - floor, idx) for idx, (q, floor) in enumerate(zip(quotas, floor_counts))],
        key=lambda x: (-x[0], x[1])
    )
    for _, idx in fractions[:remainder]:
        floor_counts[idx] += 1

    diff = target_size - sum(floor_counts)
    if diff > 0:
        order = sorted(range(len(counts)),
                       key=lambda i: (-counts[i], i))
        for idx in order:
            if diff == 0:
                break
            floor_counts[idx] += 1
            diff -= 1
    elif diff < 0:
        order = sorted(range(len(floor_counts)),
                       key=lambda i: (floor_counts[i], i))
        for idx in order:
            if diff == 0:
                break
            if floor_counts[idx] > 0:
                floor_counts[idx] -= 1
                diff += 1

    return floor_counts


def build_session_window_index(subjects: List[str],
                               emb_cache: EmbeddingCache,
                               target_size: Optional[int] = None
                               ) -> List[Tuple[str, str, int]]:
    """
    Reconstruct the exact (subject, session, win_idx) index to align with fold data.

    If `target_size` is provided, the session counts are adjusted to exactly
    match the fold sample size. This helps catch data/index alignment bugs
    and keeps the M6 loader consistent with the processed fold arrays.
    """
    index: List[Tuple[str, str, int]] = []
    expected_sessions: List[Tuple[str, str]] = []

    for subj in sorted(subjects):  # Must match preprocess.py ordering exactly
        if subj in NO_TELEMETRY:
            continue
        for sess in ("A", "D"):
            if subj in AWAKE_ONLY and sess == "D":
                continue
            expected_sessions.append((subj, sess))

    if not expected_sessions:
        return []

    valid_sessions: List[Tuple[str, str]] = []
    missing_sessions: List[Tuple[str, str]] = []
    for subj, sess in expected_sessions:
        if emb_cache.has(subj, sess):
            valid_sessions.append((subj, sess))
        else:
            missing_sessions.append((subj, sess))

    if missing_sessions:
        missing_names = [f"{subj}_{sess}" for subj, sess in missing_sessions]
        available_names = emb_cache.available_sessions()
        raise FileNotFoundError(
            f"Missing embedding sessions for index reconstruction: {missing_names}. "
            f"Available embeddings: {available_names}. "
            f"Run `python -m src.models.m6_extractor` to generate the missing files."
        )

    counts: List[int] = []
    for subj, sess in valid_sessions:
        n_win = _windows_per_session(subj, sess, emb_cache)
        counts.append(max(n_win, 0))

    total = sum(counts)
    if target_size is not None:
        if total == 0 and target_size > 0:
            raise ValueError(
                f"Unable to build index for {len(expected_sessions)} sessions: "
                f"no valid window counts were found. Check if the embedding cache is complete."
            )
        if total != target_size:
            if abs(total - target_size) > max(1, int(0.02 * target_size)):
                raise ValueError(
                    f"Index reconstruction failed: estimated total windows={total} "
                    f"does not match fold size={target_size}."
                )
            counts = _allocate_session_counts(counts, target_size)
            total = sum(counts)
            if total != target_size:
                raise RuntimeError(
                    f"Failed to allocate session counts to target size={target_size}."
                )

    for (subj, sess), n_win in zip(valid_sessions, counts):
        for w in range(n_win):
            index.append((subj, sess, w))

    return index


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset that yields aligned (visual_clip, can_window, label)
# ─────────────────────────────────────────────────────────────────────────────
class M6Dataset(Dataset):
    """
    Reads CAN windows + labels from a fold .npz (the `mm_*` arrays).
    
    Reconstructs the (subject, session, window_idx) for each sample
    by using `build_session_window_index`, then loads the correct
    embeddings from the cache.
    """

    def __init__(self,
                 mm_tele: np.ndarray,
                 mm_y: np.ndarray,
                 fold_subjects: List[str],
                 emb_cache: EmbeddingCache,
                 t_vis: int = 16,
                 index: Optional[List[Tuple[str, str, int]]] = None,
                 is_train: bool = False):
        self.mm_tele = mm_tele.astype(np.float32)
        self.mm_y    = mm_y.astype(np.int64)
        self.t_vis   = t_vis
        self.cache   = emb_cache
        self.index   = index
        self.is_train = is_train
        
        # Pre-load all available embeddings for this fold
        self.embeddings_map = {}  # (subject, session) -> full record
        expected_sessions: List[Tuple[str, str]] = []
        for subj in fold_subjects:
            if subj in NO_TELEMETRY:
                continue
            for sess in ("A", "D"):
                if subj in AWAKE_ONLY and sess == "D":
                    continue
                expected_sessions.append((subj, sess))
                if emb_cache.has(subj, sess):
                    try:
                        self.embeddings_map[(subj, sess)] = emb_cache.load(subj, sess)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load embeddings for {subj}_{sess}: {exc}"
                        ) from exc

        missing_sessions = [f"{subj}_{sess}"
                            for subj, sess in expected_sessions
                            if (subj, sess) not in self.embeddings_map]
        if missing_sessions:
            available = emb_cache.available_sessions()
            raise FileNotFoundError(
                f"Missing embeddings for required fold sessions: {missing_sessions}. "
                f"Available embeddings: {available}. "
                f"Run `python -m src.models.m6_extractor` and ensure the cache is complete."
            )

        if self.index is None:
            raise ValueError("M6Dataset requires an explicit index list for alignment.")
        if len(self.index) != len(self.mm_y):
            raise ValueError(
                f"Index length {len(self.index)} does not match labels length {len(self.mm_y)}. "
                f"This indicates a session/window reconstruction bug."
            )

        n_samples = len(self.mm_y)
        print(f"  [info] M6Dataset: {n_samples} samples, {len(self.embeddings_map)} session embeddings available")

    def __len__(self) -> int:
        return len(self.mm_y)

    def __getitem__(self, i: int):
        can_window = self.mm_tele[i]
        label = self.mm_y[i]
        
        # CRITICAL FIX: Fail loudly if embeddings are missing (no silent random data)
        if not self.index or i >= len(self.index):
            raise IndexError(
                f"Sample {i}: No index provided or out of range (index len={len(self.index) if self.index else 0})"
            )
        
        subj, sess, win_idx = self.index[i]
        if (subj, sess) not in self.embeddings_map:
            raise KeyError(
                f"Sample {i}: Missing embeddings for {subj}_{sess}. "
                f"Available: {sorted(self.embeddings_map.keys())}"
            )
        
        rec = self.embeddings_map[(subj, sess)]
        clip = sample_clip_embedding(rec, win_idx, self.t_vis)
        
        # Validate embedding shape
        if clip.shape != (self.t_vis, self.cache.embed_dim):
            raise ValueError(
                f"Sample {i}: Embedding shape mismatch. "
                f"Got {clip.shape}, expected ({self.t_vis}, {self.cache.embed_dim})"
            )
        
        # Optional: Add small Gaussian noise to visual embeddings during training
        # (helps with generalization on small datasets)
        if self.is_train:
            noise = np.random.normal(0, 0.005, clip.shape).astype(np.float32)
            clip = clip + noise
        
        return (
            torch.from_numpy(clip),
            torch.from_numpy(can_window),
            int(label),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
               criterion: nn.Module, optim: Optional[torch.optim.Optimizer]
               ) -> Tuple[float, float]:
    train_mode = optim is not None
    model.train(train_mode)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for vis, can, y in loader:
        vis = vis.to(device, non_blocking=True)
        can = can.to(device, non_blocking=True)
        y   = y.to(device,   non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(vis, can)
            loss   = criterion(logits, y)
            if train_mode:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

        bs            = y.size(0)
        total_loss   += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_n      += bs

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for vis, can, y in loader:
            logits = model(vis.to(device), can.to(device))
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                              zero_division=0))
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "y_true":   y_true,
        "y_pred":   y_pred,
        "report":   report,
    }


def train_one_fold(fold_idx: int,
                   variant: str = "lite_v2",
                   t_vis: int   = 16,
                   epochs: int  = 50,
                   batch_size: int = 16,
                   lr: float    = 1e-3,
                   weight_decay: float = 1e-5,
                   device: Optional[torch.device] = None,
                   seed: int    = 42,
                   processed_dir: Path = PROCESSED_DIR,
                   emb_dir: Path = EMB_DIR,
                   ckpt_dir: Path = CKPT_DIR,
                   validate_data: bool = True,
                   verbose: bool = True) -> dict:
    """Train M6 on one fold and return a result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if verbose:
        print(f"\n-- Fold {fold_idx}  (M6_{variant})  device={device}")

    # 0. DATA VALIDATION (NEW: catch data bugs early)
    if validate_data:
        if verbose:
            print("   Validating data integrity...")
        from .m6_validate import full_validation
        is_valid, report = full_validation(
            fold_idx, processed_dir, emb_dir, verbose=verbose
        )
        if not is_valid:
            fold_issues = report['fold_check']['issues']
            emb_missing = report['emb_check']['missing']
            idx_issues = report['idx_check']['issues']
            raise RuntimeError(
                f"Data validation failed for fold {fold_idx}. "
                f"Fold issues={fold_issues}. "
                f"Missing embeddings={emb_missing}. "
                f"Index issues={idx_issues}."
            )
        if verbose:
            print("   ✓ Data validation passed")

    # 1. Load fold .npz (mm_* arrays)
    fold_path = processed_dir / f"fold_{fold_idx}.npz"
    if not fold_path.exists():
        raise FileNotFoundError(
            f"{fold_path} not found. Run notebook 05 first."
        )
    f = np.load(str(fold_path), allow_pickle=False)

    test_subjects  = list(f["test_subjects"])
    train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]

    # 2. Datasets
    cache = EmbeddingCache(emb_dir)
    
    # Debug: Check what keys are in the fold file
    fold_keys = set(f.files)
    if verbose:
        print(f"   fold keys available: {sorted(fold_keys)}")
    
    # Try to find the right keys (support multiple schema versions)
    tele_train_key = next((k for k in ['mm_tele_train', 'tele_train'] if k in fold_keys), None)
    y_train_key = next((k for k in ['mm_y_train', 'y_train'] if k in fold_keys), None)
    tele_test_key = next((k for k in ['mm_tele_test', 'tele_test'] if k in fold_keys), None)
    y_test_key = next((k for k in ['mm_y_test', 'y_test'] if k in fold_keys), None)
    
    if any(k is None for k in [tele_train_key, y_train_key, tele_test_key, y_test_key]):
        raise ValueError(
            f"Required keys not found in fold_{fold_idx}.npz.\n"
            f"Expected: mm_tele_train, mm_y_train, mm_tele_test, mm_y_test\n"
            f"Found: {sorted(fold_keys)}"
        )
    
    train_idx = build_session_window_index(
        train_subjects, cache, target_size=len(f[tele_train_key])
    )
    test_idx = build_session_window_index(
        test_subjects, cache, target_size=len(f[tele_test_key])
    )
    
    train_ds = M6Dataset(
        f[tele_train_key], f[y_train_key],
        fold_subjects=train_subjects, emb_cache=cache, t_vis=t_vis, index=train_idx, is_train=True,
    )
    test_ds = M6Dataset(
        f[tele_test_key], f[y_test_key],
        fold_subjects=test_subjects, emb_cache=cache, t_vis=t_vis, index=test_idx, is_train=False,
    )
    if verbose:
        print(f"   train={len(train_ds)}  test={len(test_ds)}  "
              f"emb_dim={cache.embed_dim}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 3. Model
    model = build_m6(variant, emb_dim=cache.embed_dim).to(device)
    if verbose:
        print(f"   params={count_parameters(model):,}")

    # 4. Loss / optimiser
    cw         = _class_weights(np.asarray(f[y_train_key])).to(device)
    criterion  = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.0)
    optimiser  = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs
    )

    # 5. Training loop
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_f1, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, device,
                                     criterion, optimiser)
        va_loss, va_acc = _run_epoch(model, test_loader,  device,
                                     criterion, None)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Track best-by-macro-F1
        eval_now = evaluate(model, test_loader, device)
        if eval_now["macro_f1"] > best_f1:
            best_f1    = eval_now["macro_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if verbose:
            print(f"   ep {ep:02d}/{epochs}  "
                  f"tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f}  "
                  f"va_loss={va_loss:.3f} va_acc={va_acc:.3f}  "
                  f"f1={eval_now['macro_f1']:.3f}")

    train_time = time.time() - t0

    # 6. Restore best weights, final evaluation, checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, test_loader, device)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"M6{variant.lower()}_fold{fold_idx}.pt"
    torch.save({
        "state_dict":     model.state_dict(),
        "variant":        variant,
        "fold":           fold_idx,
        "embed_dim":      cache.embed_dim,
        "t_vis":          t_vis,
        "test_subjects":  list(test_subjects),
        "macro_f1":       final["macro_f1"],
        "accuracy":       final["accuracy"],
    }, str(ckpt_path))

    if verbose:
        print(f"\n   best_f1={final['macro_f1']:.4f}  "
              f"acc={final['accuracy']:.4f}  "
              f"time={train_time:.1f}s")
        print(final["report"])

    return {
        "fold":          fold_idx,
        "variant":       variant,
        "test_subjects": list(test_subjects),
        "history":       history,
        "accuracy":      final["accuracy"],
        "macro_f1":      final["macro_f1"],
        "y_true":        final["y_true"],
        "y_pred":        final["y_pred"],
        "report":        final["report"],
        "train_time_s":  train_time,
        "checkpoint":    str(ckpt_path.relative_to(ROOT)),
    }


def cross_validate(variant: str = "full",
                   t_vis: int  = 16,
                   epochs: int = 30,
                   batch_size: int = 32,
                   lr: float    = 3e-4,
                   weight_decay: float = 1e-4,
                   seed: int   = 42,
                   processed_dir: Path = PROCESSED_DIR,
                   emb_dir: Path = EMB_DIR,
                   ckpt_dir: Path = CKPT_DIR,
                   report_dir: Path = REPORT_DIR,
                   verbose: bool = True) -> dict:
    """5-fold subject-independent CV.  Saves results to results/reports/."""
    fold_results = []
    for k in range(len(FOLDS_TEST)):
        res = train_one_fold(
            fold_idx=k, variant=variant, t_vis=t_vis,
            epochs=epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, seed=seed,
            processed_dir=processed_dir, emb_dir=emb_dir,
            ckpt_dir=ckpt_dir, verbose=verbose,
        )
        fold_results.append(res)

    accs = [r["accuracy"] for r in fold_results]
    f1s  = [r["macro_f1"] for r in fold_results]
    summary = {
        "model":    f"M6_{variant}",
        "folds":    fold_results,
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_f1":  float(np.mean(f1s)),
        "std_f1":   float(np.std(f1s)),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"M6_{variant}_results.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.tolist()
                  if isinstance(o, (np.ndarray,)) else str(o))
    if verbose:
        print(f"\n[M6] mean acc={summary['mean_acc']:.4f} "
              f"± {summary['std_acc']:.4f}   "
              f"mean F1={summary['mean_f1']:.4f} "
              f"± {summary['std_f1']:.4f}")
        print(f"     saved → {out_path.relative_to(ROOT)}")
    return summary
