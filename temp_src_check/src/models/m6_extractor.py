"""
M6 — YOLOv8 Backbone Embedding Extractor
========================================
Pre-computes 512-d embeddings for every frame in `datasets/yolo_frames/`
using the trained M5 YOLOv8-cls backbone, and caches them on disk so
that M6 training can run on cached features (no GPU required for the
backbone forward pass each epoch).

Output layout
-------------
    models/embeddings/
        A_A.npz     ← embeddings for subject A, Alert  session
        A_D.npz     ← embeddings for subject A, Drowsy session
        ...
        S_D.npz

Each .npz file contains four arrays (length N = #frames in that session):
    frame_idx : (N,) int32       original 60-fps frame index
    cls_idx   : (N,) int8        per-frame KSS-bin class (0/1/2)
    embedding : (N, 512) float32 YOLOv8 penultimate features
    files     : (N,)   <U120     source jpg path (relative to repo root)

The frame_idx field is parsed from the filename `frame_XXXXX.jpg`, so
frames remain time-ordered with respect to the original video — even
though the on-disk folder structure is class-segregated.

Usage
-----
    python -m src.models.m6_extractor                     # all sessions
    python -m src.models.m6_extractor --sessions A_A B_D  # subset
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Project paths --------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
FRAMES_ROOT   = ROOT / "datasets" / "yolo_frames"
DEFAULT_CKPT  = ROOT / "models" / "checkpoints" / "M5_fold0.pt"
OUT_DIR       = ROOT / "models" / "embeddings"

CLASS_MAP = {"Alert": 0, "LowVigilant": 1, "Drowsy": 2}
FRAME_RE  = re.compile(r"frame_(\d+)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone wrapper around the trained YOLOv8-cls model
# ─────────────────────────────────────────────────────────────────────────────
class YoloClsBackbone(torch.nn.Module):
    """
    Strips the final classification head of a YOLOv8-cls model so that the
    forward pass returns the global-pooled feature vector (typically 512-d
    for the `n` size).  Works with `ultralytics`-style checkpoints saved
    via `model.save(...)` after training.
    """

    def __init__(self, ckpt_path: Path):
        super().__init__()
        from ultralytics import YOLO     # local import → optional dep
        self.yolo = YOLO(str(ckpt_path))
        self.yolo.model.model.eval()  # Call eval() on the underlying nn.Module, not the YOLO wrapper

        # Locate the Classify head (usually `model.model.model[-1]`)
        modules = list(self.yolo.model.model)
        self.backbone = torch.nn.Sequential(*modules[:-1])
        self.classify = modules[-1]

        # Probe the global-pool dim by running a dummy tensor
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(x)
            feat = self._pool(feat)
        self.embed_dim = feat.shape[-1]

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        """Global-average-pool the spatial dims → (B, C)."""
        if x.ndim == 4:
            x = x.mean(dim=(2, 3))
        elif x.ndim == 3:
            x = x.mean(dim=2)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B, 3, H, W)
        feat = self.backbone(x)
        feat = self._pool(feat)
        return feat.detach().cpu().float()                  # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Per-session frame dataset
# ─────────────────────────────────────────────────────────────────────────────
class _SessionFrames(Dataset):
    """
    Iterate every jpg under `<frames_root>/<subject>_<session>/{Alert|...}/`
    and yield (image_tensor, frame_idx, cls_idx, rel_path).
    """

    def __init__(self, session_dir: Path, img_size: int = 224):
        self.session_dir = session_dir
        self.img_size    = img_size
        self.samples: List[tuple[Path, int, int]] = []

        for cls_name, cls_idx in CLASS_MAP.items():
            cls_dir = session_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.glob("frame_*.jpg")):
                m = FRAME_RE.search(p.name)
                if m is None:
                    continue
                self.samples.append((p, int(m.group(1)), cls_idx))

        # Sort by frame_idx so embeddings come out in true temporal order
        self.samples.sort(key=lambda s: s[1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, frame_idx, cls_idx = self.samples[i]
        img = Image.open(path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        ten = torch.from_numpy(arr).permute(2, 0, 1)      # (3, H, W)
        return ten, frame_idx, cls_idx, str(path.relative_to(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Extraction routine
# ─────────────────────────────────────────────────────────────────────────────
def list_sessions(frames_root: Path) -> List[Path]:
    """All `<subject>_<session>` directories in alphabetical order."""
    return sorted(p for p in frames_root.iterdir()
                  if p.is_dir() and "_" in p.name)


def extract_session(backbone: YoloClsBackbone, session_dir: Path,
                    out_path: Path, batch_size: int = 64,
                    num_workers: int = 0,
                    device: Optional[torch.device] = None) -> dict:
    """Run backbone over every frame of one session and save the .npz."""
    ds = _SessionFrames(session_dir)
    if len(ds) == 0:
        print(f"  [skip] {session_dir.name}: no frames")
        return {}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    backbone = backbone.to(device)
    backbone.yolo.model.model.eval()  # Call eval() on underlying nn.Module, not the YOLO wrapper

    all_emb: List[np.ndarray] = []
    all_idx: List[int]        = []
    all_cls: List[int]        = []
    all_files: List[str]      = []

    for batch_imgs, batch_idx, batch_cls, batch_files in loader:
        emb = backbone(batch_imgs.to(device))     # (B, D)
        all_emb.append(emb.numpy())
        all_idx.extend(batch_idx.tolist())
        all_cls.extend(batch_cls.tolist())
        all_files.extend(batch_files)

    emb_arr = np.concatenate(all_emb, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        frame_idx=np.asarray(all_idx, dtype=np.int32),
        cls_idx=np.asarray(all_cls,   dtype=np.int8),
        embedding=emb_arr,
        files=np.asarray(all_files,   dtype=f"<U{max(40, max(len(f) for f in all_files))}"),
    )
    print(f"  [ok ] {session_dir.name}  →  {emb_arr.shape}  →  {out_path.name}")
    return {"n": int(emb_arr.shape[0]), "dim": int(emb_arr.shape[1])}


def extract_all(ckpt_path: Path = DEFAULT_CKPT,
                frames_root: Path = FRAMES_ROOT,
                out_dir: Path = OUT_DIR,
                sessions: Optional[List[str]] = None,
                batch_size: int = 64,
                num_workers: int = 0,
                overwrite: bool = False) -> None:
    """Extract embeddings for every (or selected) session under `frames_root`."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"M5 checkpoint not found: {ckpt_path}")

    print(f"[M6] loading backbone from {ckpt_path}")
    backbone = YoloClsBackbone(ckpt_path)
    print(f"[M6] backbone embedding dim = {backbone.embed_dim}")

    todo = list_sessions(frames_root)
    if sessions:
        keep = set(sessions)
        todo = [p for p in todo if p.name in keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[M6] {len(todo)} session(s) to process → {out_dir}")

    for session_dir in todo:
        out_path = out_dir / f"{session_dir.name}.npz"
        if out_path.exists() and not overwrite:
            print(f"  [skip] {session_dir.name}: already extracted "
                  f"(use --overwrite to redo)")
            continue
        extract_session(backbone, session_dir, out_path,
                        batch_size=batch_size, num_workers=num_workers)

    print("[M6] done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help=f"Path to YOLOv8-cls .pt checkpoint "
                        f"(default: {DEFAULT_CKPT.relative_to(ROOT)})")
    p.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    p.add_argument("--out-dir",     type=Path, default=OUT_DIR)
    p.add_argument("--sessions", nargs="+", default=None,
                   help="Subset of sessions to extract, e.g. A_A B_D")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_cli(argv)
    extract_all(
        ckpt_path=args.ckpt, frames_root=args.frames_root,
        out_dir=args.out_dir, sessions=args.sessions,
        batch_size=args.batch_size, num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
