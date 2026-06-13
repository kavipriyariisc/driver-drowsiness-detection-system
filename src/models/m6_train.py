"""
M6 — Window-aligned Visual + CAN Dataset and Training Loop
==========================================================

Design Reference: M6_Design.md (Step 3: Alignment)
Status: Core training loop implemented; data source issue identified (see below)

Core Purpose:
    Bridges cached YOLOv8 embeddings (per-session, frame-indexed) with
    CAN telemetry windows in `datasets/processed/ul_dd/fold_*.npz`,
    then trains M6 fusion models with subject-independent 5-fold CV.

Window-Frame Alignment Strategy (CRITICAL):
    For each telemetry window index i:
    
    1. Compute frame range in original 60-fps video:
       start_frame = i * 900
       end_frame = i * 900 + 3600
       (covers 60 seconds of video)
    
    2. Look up embeddings with frame indices in [start_frame, end_frame]
       (these are 60-fps indices from the original video)
    
    3. Uniformly sample T_VIS=16 embeddings from matched frames
    
    4. Return synchronized (visual_clip, can_window, label)
    
    Result: Visual and telemetry modalities cover SAME time period

Time Reference:
    Original video : 60 fps (reference)
    CAN telemetry  : 4 Hz   (downsampled × 15)
    Frame alignment: 60-fps frame_idx stored in embedding cache
    Window length  : 60 s → 240 CAN samples @ 4Hz → 3600 frames @ 60fps
    Window stride  : 15 s → 60 CAN samples @ 4Hz → 900 frames @ 60fps

Data Source Issue (⚠️ CRITICAL):
    Current: Embeddings extracted from datasets/yolo_frames/ (classification)
    Should be: From actual UL-DD driving videos (temporal sequences)
    Impact: Breaks alignment → accuracy capped at ~41%
    Fix: Modify m6_extractor.py to use actual video source
    
    Until fixed:
      • Training uses random fallback for missing embeddings
      • Multimodal alignment is broken
      • Cannot claim "true fusion" gains
      • Accuracy will plateau ~41% regardless of model

Graceful Error Handling:
    • Missing embeddings (subjects B, I, M): Use random fallback
    • Continues training instead of crashing
    • Prints warnings for transparency
    • Accuracy degradation is documented

Example Window Alignment:
    Window 5 of subject A, Alert session:
      - CAN window 5: indices [300:360] at 4Hz = [4500:5400] frames at 60fps
      - Visual frames: Select all embeddings with frame_idx ∈ [4500, 5400]
      - Sample 16 uniformly from matched frames
      - Return (visual_16x512, can_60x5, label)
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
from torch.utils.data import DataLoader, Dataset, Subset

from .m6_fusion import (CLASS_NAMES, M6_Full, M6_Lite, build_m6,
                        count_parameters)

# ─── Project paths ────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "datasets" / "processed" / "ul_dd"
EMB_DIR       = ROOT / "models"   / "embeddings_uldd"  # NEW: 60 fps from real videos
EMB_DIR_OLD   = ROOT / "models"   / "embeddings"       # OLD: 1 fps from yolo_frames
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
        return (self.emb_dir / f"{subject}_{session}.npz").exists()

    def available_sessions(self) -> List[str]:
        """List all available session files."""
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


def sample_clip_embedding_by_range(rec: dict,
                                   start_4hz: int,
                                   end_4hz: int,
                                   t_vis: int = 16) -> np.ndarray:
    """
    Return a (t_vis, embed_dim) clip using exact 4-Hz window bounds.

    Args:
        rec: embedding record with keys {"frame_idx", "embedding"}
        start_4hz: inclusive start index in 4-Hz domain
        end_4hz: exclusive end index in 4-Hz domain
        t_vis: number of visual tokens to sample
    """
    fi = rec["frame_idx"]
    emb = rec["embedding"]
    lo = int(start_4hz) * DOWNSAMPLE
    hi = int(end_4hz) * DOWNSAMPLE

    sel = np.where((fi >= lo) & (fi < hi))[0]
    if sel.size == 0:
        raise ValueError(
            f"No embeddings in exact frame range [{lo}, {hi})"
        )

    if sel.size >= t_vis:
        idx = np.linspace(0, sel.size - 1, t_vis).round().astype(np.int64)
    else:
        idx = np.linspace(0, sel.size - 1, t_vis).astype(np.int64)
    rows = sel[idx]
    return emb[rows]


def _has_embeddings_in_exact_range(rec: dict,
                                   start_4hz: int,
                                   end_4hz: int) -> bool:
    """Check whether at least one cached embedding exists in exact range."""
    fi = rec["frame_idx"]
    lo = int(start_4hz) * DOWNSAMPLE
    hi = int(end_4hz) * DOWNSAMPLE
    left = np.searchsorted(fi, lo, side="left")
    right = np.searchsorted(fi, hi, side="left")
    return right > left


def _normalize_session_value(raw_session: str) -> Optional[str]:
    """Normalize session metadata token to 'A' or 'D' when possible."""
    s = str(raw_session).strip().upper()
    if s in {"A", "ALERT"}:
        return "A"
    if s in {"D", "DROWSY"}:
        return "D"

    # Handle compound forms like "A_A", "A_D", "SUBJ_A", etc.
    if s.endswith("_A") or s.endswith("-A"):
        return "A"
    if s.endswith("_D") or s.endswith("-D"):
        return "D"
    return None


def _repair_sessions_from_subject_and_winidx(mm_subject: np.ndarray,
                                             mm_session: np.ndarray,
                                             mm_win_idx: np.ndarray) -> np.ndarray:
    """
    Repair potentially corrupted session metadata.

    Strategy:
      1) Use normalized session token when valid ('A'/'D').
      2) Otherwise infer from per-subject win-index reset:
         first block -> 'A', second block -> 'D'.
      3) For awake-only subjects (C/F/L), force 'A'.
    """
    subj_arr = np.asarray(mm_subject).astype(str)
    sess_arr = np.asarray(mm_session).astype(str)
    win_arr = np.asarray(mm_win_idx).astype(np.int64)

    repaired = np.empty(len(subj_arr), dtype="U1")
    state = {}  # subj -> {"prev": int, "block": int}

    for i in range(len(subj_arr)):
        subj = subj_arr[i]
        win = int(win_arr[i])
        norm = _normalize_session_value(sess_arr[i])

        if subj in AWAKE_ONLY:
            repaired[i] = "A"
            continue

        if norm in {"A", "D"}:
            repaired[i] = norm
            continue

        if subj not in state:
            state[subj] = {"prev": win, "block": 0}
        else:
            if win < state[subj]["prev"]:
                state[subj]["block"] += 1
            state[subj]["prev"] = win

        repaired[i] = "A" if state[subj]["block"] == 0 else "D"

    return repaired


# ─────────────────────────────────────────────────────────────────────────────
# Recover per-session window count + telemetry slices from a fold .npz
# ─────────────────────────────────────────────────────────────────────────────
def _windows_per_session(subject: str, session: str,
                         emb_cache: EmbeddingCache) -> int:
    """
    Estimate how many windows a session contributes to the multimodal
    fold arrays.  We use the highest cached frame_idx as a proxy for
    session length (in 60-fps frames) and apply the same sliding-window
    arithmetic as ULDDProcessor.
    """
    rec   = emb_cache.load(subject, session)
    n_60  = int(rec["frame_idx"].max()) + 1            # rough video length
    n_4   = n_60 // DOWNSAMPLE                          # 4-Hz samples
    if n_4 < T_CAN:
        return 0
    return (n_4 - T_CAN) // (STRIDE_SEC * TARGET_HZ) + 1


def build_session_window_index(subjects: List[str],
                               emb_cache: EmbeddingCache
                               ) -> List[Tuple[str, str, int]]:
    """
    Reconstruct the (subject, session, win_idx) ordering used by
    `ULDDProcessor.build_folds()` for the multimodal arrays
    (`mm_fau_*`, `mm_tele_*`, `mm_y_*`).

    Order matches ULDDProcessor:
        outer  : subject in alphabetical order
        middle : session 'A' then 'D'   (Alert then Drowsy)
        inner  : window 0, 1, 2, …
    Only sessions with telemetry (i.e. subject ∉ NO_TELEMETRY) and an
    embedding cache available are included.
    """
    index: List[Tuple[str, str, int]] = []
    for subj in subjects:
        if subj in NO_TELEMETRY:
            continue
        for sess in ("A", "D"):
            if subj in AWAKE_ONLY and sess == "D":
                continue
            if not emb_cache.has(subj, sess):
                continue
            n_win = _windows_per_session(subj, sess, emb_cache)
            for w in range(n_win):
                index.append((subj, sess, w))
    return index


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset that yields aligned (visual_clip, can_window, label)
# ─────────────────────────────────────────────────────────────────────────────
class M6Dataset(Dataset):
    """
    M6 Dataset with PROPER ALIGNMENT.
    
    FIXED APPROACH (vs old random session selection):
    - Each multimodal sample has metadata: (subject, session, win_idx, start_4hz, end_4hz)
    - Use metadata to load the EXACT visual embeddings for that window
    - Use metadata to slice the EXACT telemetry window
    - No random session assignment, no random fallback embeddings
    
    This ensures M6 trains on SYNCHRONIZED visual + telemetry data.
    """

    def __init__(self,
                 mm_tele: np.ndarray,
                 mm_y: np.ndarray,
                 mm_subject: np.ndarray,       # NEW
                 mm_session: np.ndarray,       # NEW
                 mm_win_idx: np.ndarray,       # NEW (not currently used, but here for future)
                 mm_start_4hz: np.ndarray,     # NEW (frame indices @ 4Hz)
                 mm_end_4hz: np.ndarray,       # NEW
                 fold_subjects: List[str],
                 emb_cache: EmbeddingCache,
                 t_vis: int = 16,
                 allow_missing_embeddings: bool = False):  # NEW
        """
        Args:
            mm_tele: Telemetry windows, shape (N, 240, 5) or similar
            mm_y: Labels, shape (N,)
            mm_subject: Subject IDs, shape (N,), dtype 'U1'
            mm_session: Session type ('A' or 'D'), shape (N,), dtype 'U1'
            mm_win_idx: Window index, shape (N,)
            mm_start_4hz: Frame range start (in 4Hz CAN domain), shape (N,)
            mm_end_4hz: Frame range end
            fold_subjects: List of subjects in fold (for pre-loading)
            emb_cache: EmbeddingCache
            t_vis: Temporal window for visual features
            allow_missing_embeddings: If False, raise error on missing embedding
        """
        self.mm_tele = mm_tele.astype(np.float32)
        self.mm_y    = mm_y.astype(np.int64)
        self.mm_subject = mm_subject.astype(str)
        self.mm_session = mm_session.astype(str)
        self.mm_win_idx = mm_win_idx.astype(np.int32)
        self.mm_start_4hz = mm_start_4hz.astype(np.int32)
        self.mm_end_4hz = mm_end_4hz.astype(np.int32)
        
        self.t_vis = t_vis
        self.cache = emb_cache
        self.allow_missing = allow_missing_embeddings
        
        # Pre-load embedding records for exact (subject, session) pairs
        # actually present in this dataset split.
        self.embeddings_map = {}  # (subject, session) -> record dict
        missing_count = 0
        unique_pairs = sorted(set(zip(self.mm_subject.tolist(), self.mm_session.tolist())))
        for subj, sess in unique_pairs:
            if emb_cache.has(subj, sess):
                try:
                    rec = emb_cache.load(subj, sess)
                    self.embeddings_map[(subj, sess)] = rec
                except Exception:
                    missing_count += 1
            else:
                missing_count += 1
        
        n_samples = len(self.mm_y)
        n_available = len(self.embeddings_map)
        print(f"  [info] M6Dataset: {n_samples} samples, {n_available} subject/sessions available")
        if missing_count > 0:
            print(f"        {missing_count} subject/sessions missing embeddings")

    def __len__(self) -> int:
        return len(self.mm_y)

    def __getitem__(self, i: int):
        # Get CAN, label, and METADATA for this sample
        can_window = self.mm_tele[i]        # (240, 5)
        label = self.mm_y[i]
        subject = self.mm_subject[i]
        session = self.mm_session[i]
        
        # Load the EXACT visual embedding for this subject/session
        key = (subject, session)
        if key in self.embeddings_map:
            rec = self.embeddings_map[key]
            win_idx = int(self.mm_win_idx[i])
            start_4hz = int(self.mm_start_4hz[i])
            end_4hz = int(self.mm_end_4hz[i])

            # Primary alignment path: exact metadata range (4Hz -> 60fps)
            # Fallback path: win_idx-based sampling (for backward compatibility)
            clip = sample_clip_embedding_by_range(
                rec,
                start_4hz=start_4hz,
                end_4hz=end_4hz,
                t_vis=self.t_vis,
            )
        else:
            # FIXED: No random fallback
            if not self.allow_missing:
                raise ValueError(
                    f"Sample {i}: Embedding not found for subject={subject}, "
                    f"session={session}. "
                    f"Available: {list(self.embeddings_map.keys())}"
                )
            else:
                # Only if explicitly allowed, use zeros (not random)
                embed_dim = self.cache.embed_dim
                clip = np.zeros((self.t_vis, embed_dim), dtype=np.float32)
        
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


def _get_or_create_metadata(f: dict, key_base: str, n_samples: int, 
                            default_subject: str = 'A',
                            default_session: str = 'A') -> np.ndarray:
    """
    Safely retrieve metadata array from fold dict, or create a default one.
    
    Args:
        f: numpy npz file dict-like object
        key_base: base key name (e.g., 'mm_subject_train')
        n_samples: number of samples to create if key missing
        default_subject: default subject ID if creating
        default_session: default session type if creating
    
    Returns:
        numpy array of shape (n_samples,)
    """
    if key_base in f.files:
        return f[key_base]
    
    # Infer type from key name
    if 'subject' in key_base:
        return np.array([default_subject] * n_samples, dtype='U1')
    elif 'session' in key_base:
        return np.array([default_session] * n_samples, dtype='U1')
    elif 'win_idx' in key_base:
        return np.arange(n_samples, dtype=np.int32)
    elif 'start_4hz' in key_base:
        return np.zeros(n_samples, dtype=np.int32)
    elif 'end_4hz' in key_base:
        return np.full(n_samples, 240, dtype=np.int32)
    else:
        raise ValueError(f"Unknown metadata key: {key_base}")


def train_one_fold(fold_idx: int,
                   variant: str = "full",
                   t_vis: int   = 16,
                   epochs: int  = 30,
                   batch_size: int = 32,
                   lr: float    = 3e-4,
                   weight_decay: float = 1e-4,
                   device: Optional[torch.device] = None,
                   seed: int    = 42,
                   processed_dir: Path = PROCESSED_DIR,
                   emb_dir: Path = EMB_DIR,
                   ckpt_dir: Path = CKPT_DIR,
                   verbose: bool = True) -> dict:
    """Train M6 on one fold and return a result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if verbose:
        print(f"\n-- Fold {fold_idx}  (M6_{variant})  device={device}")

    # 1. Load fold .npz (mm_* arrays)
    fold_path = processed_dir / f"fold_{fold_idx}.npz"
    if not fold_path.exists():
        raise FileNotFoundError(
            f"{fold_path} not found. Run notebook 05 first."
        )
    with np.load(str(fold_path), allow_pickle=False) as fold_npz:
        # Convert immutable NpzFile to mutable dict for in-place filtering.
        f = {k: fold_npz[k] for k in fold_npz.files}

    test_subjects  = list(f["test_subjects"])
    train_subjects = [s for s in ALL_SUBJECTS if s not in test_subjects]

    # 2. Datasets
    cache = EmbeddingCache(emb_dir)
    
    # Debug: Check what keys are in the fold file
    fold_keys = set(f.keys())
    if verbose:
        print(f"   fold keys available: {sorted(fold_keys)}")
    
    # Try to find the right keys (support multiple schema versions)
    tele_train_key = next((k for k in ['mm_tele_train', 'tele_train'] if k in fold_keys), None)
    y_train_key = next((k for k in ['mm_y_train', 'y_train'] if k in fold_keys), None)
    tele_test_key = next((k for k in ['mm_tele_test', 'tele_test'] if k in fold_keys), None)
    y_test_key = next((k for k in ['mm_y_test', 'y_test'] if k in fold_keys), None)
    
    # NEW: Metadata keys (must be present after m6_preprocess.py)
    subj_train_key = next((k for k in ['mm_subject_train'] if k in fold_keys), None)
    sess_train_key = next((k for k in ['mm_session_train'] if k in fold_keys), None)
    win_train_key = next((k for k in ['mm_win_idx_train'] if k in fold_keys), None)
    start_4hz_train_key = next((k for k in ['mm_start_4hz_train'] if k in fold_keys), None)
    end_4hz_train_key = next((k for k in ['mm_end_4hz_train'] if k in fold_keys), None)
    
    subj_test_key = next((k for k in ['mm_subject_test'] if k in fold_keys), None)
    sess_test_key = next((k for k in ['mm_session_test'] if k in fold_keys), None)
    win_test_key = next((k for k in ['mm_win_idx_test'] if k in fold_keys), None)
    start_4hz_test_key = next((k for k in ['mm_start_4hz_test'] if k in fold_keys), None)
    end_4hz_test_key = next((k for k in ['mm_end_4hz_test'] if k in fold_keys), None)
    
    if any(k is None for k in [tele_train_key, y_train_key, tele_test_key, y_test_key]):
        raise ValueError(
            f"Required keys not found in fold_{fold_idx}.npz.\n"
            f"Expected: mm_tele_train, mm_y_train, mm_tele_test, mm_y_test\n"
            f"Found: {sorted(fold_keys)}"
        )
    
    # Check if metadata is present
    has_metadata = all(k is not None for k in [
        subj_train_key, sess_train_key, win_train_key, 
        start_4hz_train_key, end_4hz_train_key,
        subj_test_key, sess_test_key, win_test_key,
        start_4hz_test_key, end_4hz_test_key
    ])
    
    if not has_metadata:
        raise ValueError(
            "Metadata keys are missing in fold file. "
            "Run: python src/data/m6_preprocess.py"
        )

    # Repair/normalize session metadata (some folds may contain corrupted '_' tokens).
    f[sess_train_key] = _repair_sessions_from_subject_and_winidx(
        f[subj_train_key],
        f[sess_train_key],
        f[win_train_key],
    )
    f[sess_test_key] = _repair_sessions_from_subject_and_winidx(
        f[subj_test_key],
        f[sess_test_key],
        f[win_test_key],
    )

    if verbose:
        print(f"   repaired train sessions: {sorted(np.unique(f[sess_train_key]).tolist())}")
        print(f"   repaired test sessions : {sorted(np.unique(f[sess_test_key]).tolist())}")
    
    # PRE-FILTER: Keep only samples with available embeddings
    # This prevents silent failures and fake multimodal data
    def filter_samples_by_embedding_availability(
        mm_subject,
        mm_session,
        mm_win_idx,
        mm_start_4hz,
        mm_end_4hz,
        cache,
    ):
        """Return indices of samples with embeddings in exact visual range."""
        keep_indices = []
        missing_session = 0
        missing_range = 0
        for i in range(len(mm_subject)):
            subject = mm_subject[i]
            session = mm_session[i]
            if not cache.has(subject, session):
                missing_session += 1
                continue
            rec = cache.load(subject, session)
            if _has_embeddings_in_exact_range(
                rec,
                start_4hz=int(mm_start_4hz[i]),
                end_4hz=int(mm_end_4hz[i]),
            ):
                keep_indices.append(i)
            else:
                missing_range += 1
        return np.array(keep_indices, dtype=np.int64), missing_session, missing_range
    
    # Filter training data
    keep_train, miss_train_sess, miss_train_rng = filter_samples_by_embedding_availability(
        f[subj_train_key],
        f[sess_train_key],
        f[win_train_key],
        f[start_4hz_train_key],
        f[end_4hz_train_key],
        cache,
    )
    n_train_before = len(f[y_train_key])
    n_train_after = len(keep_train)
    
    if n_train_after < n_train_before:
        if verbose:
            print(f"   ⚠ Filtering train: kept {n_train_after}/{n_train_before} samples "
                  f"(dropped {n_train_before - n_train_after})")
            print(f"      - missing session cache: {miss_train_sess}")
            print(f"      - no frame in exact range: {miss_train_rng}")
        # Filter all training arrays
        f[tele_train_key] = f[tele_train_key][keep_train]
        f[y_train_key] = f[y_train_key][keep_train]
        f[subj_train_key] = f[subj_train_key][keep_train]
        f[sess_train_key] = f[sess_train_key][keep_train]
        f[win_train_key] = f[win_train_key][keep_train]
        f[start_4hz_train_key] = f[start_4hz_train_key][keep_train]
        f[end_4hz_train_key] = f[end_4hz_train_key][keep_train]
        if 'mm_fau_train' in f:
            f['mm_fau_train'] = f['mm_fau_train'][keep_train]
    
    # Filter test data
    keep_test, miss_test_sess, miss_test_rng = filter_samples_by_embedding_availability(
        f[subj_test_key],
        f[sess_test_key],
        f[win_test_key],
        f[start_4hz_test_key],
        f[end_4hz_test_key],
        cache,
    )
    n_test_before = len(f[y_test_key])
    n_test_after = len(keep_test)
    
    if n_test_after < n_test_before:
        if verbose:
            print(f"   ⚠ Filtering test : kept {n_test_after}/{n_test_before} samples "
                  f"(dropped {n_test_before - n_test_after})")
            print(f"      - missing session cache: {miss_test_sess}")
            print(f"      - no frame in exact range: {miss_test_rng}")
        # Filter all test arrays
        f[tele_test_key] = f[tele_test_key][keep_test]
        f[y_test_key] = f[y_test_key][keep_test]
        f[subj_test_key] = f[subj_test_key][keep_test]
        f[sess_test_key] = f[sess_test_key][keep_test]
        f[win_test_key] = f[win_test_key][keep_test]
        f[start_4hz_test_key] = f[start_4hz_test_key][keep_test]
        f[end_4hz_test_key] = f[end_4hz_test_key][keep_test]
        if 'mm_fau_test' in f:
            f['mm_fau_test'] = f['mm_fau_test'][keep_test]
    
    train_ds = M6Dataset(
        mm_tele=f[tele_train_key],
        mm_y=f[y_train_key],
        mm_subject=f[subj_train_key],
        mm_session=f[sess_train_key],
        mm_win_idx=f[win_train_key],
        mm_start_4hz=f[start_4hz_train_key],
        mm_end_4hz=f[end_4hz_train_key],
        fold_subjects=train_subjects,
        emb_cache=cache,
        t_vis=t_vis,
        allow_missing_embeddings=False,  # STRICT: No missing embeddings
    )
    test_ds = M6Dataset(
        mm_tele=f[tele_test_key],
        mm_y=f[y_test_key],
        mm_subject=f[subj_test_key],
        mm_session=f[sess_test_key],
        mm_win_idx=f[win_test_key],
        mm_start_4hz=f[start_4hz_test_key],
        mm_end_4hz=f[end_4hz_test_key],
        fold_subjects=test_subjects,
        emb_cache=cache,
        t_vis=t_vis,
        allow_missing_embeddings=False,  # STRICT: No missing embeddings
    )
    if verbose:
        print(f"   train={len(train_ds)}  test={len(test_ds)}  "
              f"emb_dim={cache.embed_dim}")

    # Split train set by SUBJECT (not row index) for early stopping.
    train_subject_arr = np.asarray(f[subj_train_key]).astype(str)
    unique_train_subjects = np.unique(train_subject_arr)
    if unique_train_subjects.size < 2:
        raise ValueError(
            "Need at least 2 train subjects for subject-wise train/val split."
        )

    rng = np.random.RandomState(seed + fold_idx)
    perm_subjects = unique_train_subjects[rng.permutation(unique_train_subjects.size)]
    n_val_subjects = max(1, int(np.ceil(0.2 * unique_train_subjects.size)))
    val_subjects = set(perm_subjects[:n_val_subjects].tolist())

    val_mask = np.array([s in val_subjects for s in train_ds.mm_subject], dtype=bool)
    train_mask = ~val_mask

    train_indices = np.where(train_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(
            "Subject-wise split produced empty train or val set. "
            f"val_subjects={sorted(val_subjects)}"
        )
    
    train_ds_actual = Subset(train_ds, train_indices)
    val_ds_actual = Subset(train_ds, val_indices)
    
    if verbose:
        print(f"   val_subjects={sorted(val_subjects)}")
        print(f"   train_split={len(train_ds_actual)}  val={len(val_ds_actual)}  test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds_actual, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    val_loader   = DataLoader(val_ds_actual,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,         batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 3. Model
    model = build_m6(variant, emb_dim=cache.embed_dim).to(device)
    if verbose:
        print(f"   params={count_parameters(model):,}")

    # 4. Loss / optimiser
    cw         = _class_weights(np.asarray(f[y_train_key])).to(device)
    criterion  = nn.CrossEntropyLoss(weight=cw)
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
        va_loss, va_acc = _run_epoch(model, val_loader,  device,
                                     criterion, None)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Track best-by-macro-F1 (using validation set, not test)
        eval_now = evaluate(model, val_loader, device)
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

    # 6. Restore best weights, final evaluation on TEST set ONLY, checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    # CRITICAL: Evaluate ONLY on test_loader, NOT during training
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="M6 Fusion Model Training (Window-Aligned Visual + CAN)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train"],
        help="Training mode (for future eval/infer modes)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["lite", "full"],
        help="M6 model variant: lite (vision-only) or full (multimodal)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If specified, train only this fold (0-4); else 5-fold CV",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per fold",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (AdamW)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output during training",
    )
    parser.add_argument(
        "--fix-alignment",
        action="store_true",
        help="Verify metadata alignment is working (debug mode)",
    )

    args = parser.parse_args()

    if args.fix_alignment:
        # Debug mode: verify metadata is present
        print("\n" + "="*70)
        print("M6 Alignment Verification (Debug Mode)")
        print("="*70)
        
        fold_path = PROCESSED_DIR / "fold_0.npz"
        if not fold_path.exists():
            print(f"✗ Fold not found: {fold_path}")
            exit(1)
        
        fold = np.load(str(fold_path), allow_pickle=False)
        required_keys = [
            "mm_subject_train", "mm_session_train", "mm_win_idx_train",
            "mm_start_4hz_train", "mm_end_4hz_train",
            "mm_subject_test", "mm_session_test", "mm_win_idx_test",
            "mm_start_4hz_test", "mm_end_4hz_test",
        ]
        
        all_present = True
        for key in required_keys:
            if key in fold.files:
                shape = fold[key].shape
                dtype = fold[key].dtype
                print(f"  ✓ {key}: shape={shape} dtype={dtype}")
            else:
                print(f"  ✗ {key}: MISSING")
                all_present = False
        
        if all_present:
            print("\n✅ All metadata present! M6 alignment is ready.")
            print("\nNext: Train with: python -m src.models.m6_train --variant lite --epochs 50")
        else:
            print("\n❌ Some metadata missing. Run: python src/data/m6_preprocess.py")
        exit(0)

    if args.mode == "train":
        if args.fold is not None:
            # Single fold
            print(f"\n[M6] Training single fold: fold={args.fold}, variant={args.variant}")
            res = train_one_fold(
                fold_idx=args.fold,
                variant=args.variant,
                t_vis=16,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                verbose=args.verbose,
            )
            print(f"\nFold {args.fold} completed: F1={res['macro_f1']:.4f}, Acc={res['accuracy']:.4f}")
        else:
            # Full 5-fold CV
            print(f"\n[M6] Running 5-fold cross-validation: variant={args.variant}")
            summary = cross_validate(
                variant=args.variant,
                t_vis=16,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                verbose=args.verbose,
            )
            print(f"\n[M6] Final results:")
            print(f"      Mean F1:  {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
            print(f"      Mean Acc: {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
