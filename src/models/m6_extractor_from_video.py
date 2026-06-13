"""
M6 — YOLOv8 Embedding Extractor from Original UL-DD Videos
===========================================================
Extracts 512-d embeddings at 60 fps directly from original UL-DD video files
using the trained M5 YOLOv8-cls backbone. This ensures proper temporal alignment
with CAN telemetry (which operates at 4 Hz, 60-second windows).

Key difference from m6_extractor.py:
  OLD: Read from yolo_frames/ (pre-extracted @ 1 fps) → 60x information loss
  NEW: Read from UL-DD videos directly @ 60 fps → Proper temporal alignment

Output layout
-------------
    models/embeddings_uldd/
        A_A.npz     ← embeddings for subject A, Alert  session @ 60 fps
        A_D.npz     ← embeddings for subject A, Drowsy session @ 60 fps
        ...
        S_D.npz

Each .npz file contains four arrays (length N = #frames @ 60 fps in that session):
    frame_idx : (N,) int32       frame index in original video (60 Hz)
    cls_idx   : (N,) int8        per-frame KSS-bin class (0/1/2)
    embedding : (N, 512) float32 YOLOv8 penultimate features
    files     : (N,)   <U120     source file info

Frame indices are stored in 60-fps time domain:
    frame_idx=0 → t=0.000 sec
    frame_idx=60 → t=1.000 sec
    frame_idx=3600 → t=60.0 sec

This maintains perfect alignment with CAN windows:
    CAN window i: [i*240 : i*240+240] @ 4Hz
    Visual window i: [i*3600 : i*3600+3600] @ 60 fps
    → Perfectly synchronized!

Usage
-----
    python -m src.models.m6_extractor_from_video                 # all sessions
    python -m src.models.m6_extractor_from_video --sessions A_A B_D  # subset
    python -m src.models.m6_extractor_from_video --fps 30         # lower fps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Project paths
ROOT          = Path(__file__).resolve().parents[2]
VIDEO_ROOT    = Path(r"C:\Users\raka1005\Documents\IISC\UL-DD\Video_Data\Video_Data\Video_Data")
DEFAULT_CKPT  = ROOT / "models" / "checkpoints" / "M5_fold0.pt"
OUT_DIR       = ROOT / "models" / "embeddings_uldd"

CLASS_MAP = {"A": 0, "D": 1}  # Alert: 0, Drowsy: 1 (based on folder name)
SUBJECTS  = list("ABCDEFGHIJKLMNOPQRS")
NO_VIDEO  = {"B", "I", "M"}  # These subjects have no video files


# ─────────────────────────────────────────────────────────────────────────────
# Backbone wrapper (same as in m6_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────
class YoloClsBackbone(torch.nn.Module):
    """
    Strips the final classification head of a YOLOv8-cls model so that the
    forward pass returns the global-pooled feature vector (typically 512-d).
    """

    def __init__(self, ckpt_path: Path):
        super().__init__()
        from ultralytics import YOLO
        
        self.yolo = YOLO(str(ckpt_path))
        self.yolo.model.model.eval()

        modules = list(self.yolo.model.model)
        self.backbone = torch.nn.Sequential(*modules[:-1])
        self.classify = modules[-1]

        # Probe embedding dimension
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(x)
            feat = self._pool(feat)
        self.embed_dim = feat.shape[-1]

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        """Global-average-pool spatial dimensions → (B, C)."""
        if x.ndim == 4:
            x = x.mean(dim=(2, 3))
        elif x.ndim == 3:
            x = x.mean(dim=2)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image batch
        Returns:
            (B, embed_dim) embeddings
        """
        feat = self.backbone(x)
        feat = self._pool(feat)
        return feat.detach().cpu().float()


# ─────────────────────────────────────────────────────────────────────────────
# Video frame dataset (STREAMING - no full video load)
# ─────────────────────────────────────────────────────────────────────────────
class VideoFrameDataset(Dataset):
    """
    Stream frames from a video file at specified fps.
    
    CRITICAL: Does NOT load all frames to RAM.
    Instead, builds a frame index at init, then decodes on-demand.
    This prevents OOM crashes on long videos in Colab.
    """

    def __init__(
        self,
        video_path: Path,
        fps: float = 4.0,  # Changed: Use 4 fps default (not 60)
        img_size: int = 224,
        session_class: int = 0,  # Alert=0 or Drowsy=1
        max_frames: Optional[int] = None,
    ):
        """
        Args:
            video_path: Path to .mp4 file
            fps: Target frames per second to extract (default 4 for memory efficiency)
            img_size: Size to resize frames to
            session_class: Class label for frames (0=Alert, 1=Drowsy)
            max_frames: Max frames to extract (for testing)
        """
        self.video_path = video_path
        self.fps = fps
        self.img_size = img_size
        self.session_class = session_class
        
        # Build frame index WITHOUT loading frames
        self.frame_indices = self._build_frame_index(max_frames)

    def _build_frame_index(self, max_frames: Optional[int] = None) -> List[int]:
        """
        Scan video and build list of frame indices to extract.
        DOES NOT decode frames - only records which frames to grab.
        
        Stores ORIGINAL 60-fps frame indices so alignment is maintained.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate skip to achieve target fps
        # If video is 30fps and we want 4fps: skip = 30/4 = 7
        skip = max(1, int(round(video_fps / self.fps)))
        
        frame_indices = []
        frame_idx_60hz = 0  # Frame index in 60-fps time domain
        frame_count_extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract at target fps
            if frame_idx_60hz % skip == 0:
                # Store ORIGINAL 60-fps frame index
                # This is critical for M6 alignment!
                frame_indices.append(frame_idx_60hz)
                frame_count_extracted += 1

                if max_frames is not None and frame_count_extracted >= max_frames:
                    break

            frame_idx_60hz += 1

        cap.release()

        if len(frame_indices) == 0:
            raise RuntimeError(f"No frames extracted from {self.video_path}")

        return frame_indices

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        """
        On-demand frame extraction for batch item i.
        Decodes ONLY the requested frame from disk.
        
        Returns:
            (image_tensor, frame_idx_in_60fps_domain)
        """
        target_frame_60hz = self.frame_indices[i]
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        # Seek to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_60hz)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(
                f"Failed to decode frame {target_frame_60hz} from {self.video_path}"
            )

        # Preprocess
        resized = cv2.resize(
            frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = np.asarray(frame_rgb, dtype=np.float32) / 255.0  # (H, W, 3)
        ten = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        return ten, target_frame_60hz  # ← Return ORIGINAL 60-fps index!


# ─────────────────────────────────────────────────────────────────────────────
# Extraction routine
# ─────────────────────────────────────────────────────────────────────────────
def extract_session_from_video(
    backbone: YoloClsBackbone,
    video_path: Path,
    out_path: Path,
    subject: str,
    session: str,
    fps: float = 60.0,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Extract embeddings from one video file and save to .npz.

    Args:
        backbone: YoloClsBackbone instance
        video_path: Path to .mp4 file
        out_path: Path to save .npz
        subject: Subject ID (e.g., 'A')
        session: Session type (e.g., 'A' for Alert)
        fps: Target fps for extraction
        batch_size: Batch size for embedding extraction
        num_workers: DataLoader workers
        device: Torch device

    Returns:
        dict with 'n' (num frames) and 'dim' (embedding dimension)
    """
    if not video_path.exists():
        print(f"  [skip] {video_path.name}: file not found")
        return {}

    try:
        ds = VideoFrameDataset(
            video_path, fps=fps, img_size=224, session_class=CLASS_MAP[session]
        )
    except RuntimeError as e:
        print(f"  [skip] {video_path.name}: {e}")
        return {}

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.yolo.model.model.eval()

    all_emb: List[np.ndarray] = []
    all_idx: List[int] = []
    all_cls: List[int] = []

    for batch_imgs, batch_frame_idx in loader:
        emb = backbone(batch_imgs.to(device))  # (B, D)
        all_emb.append(emb.numpy())
        all_idx.extend(batch_frame_idx.tolist())
        all_cls.extend([CLASS_MAP[session]] * len(batch_frame_idx))

    emb_arr = np.concatenate(all_emb, axis=0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        frame_idx=np.asarray(all_idx, dtype=np.int32),
        cls_idx=np.asarray(all_cls, dtype=np.int8),
        embedding=emb_arr,
    )

    print(
        f"  [ok ] {subject}/{session}  →  {emb_arr.shape[0]:6d} frames @ {fps:.0f} fps  "
        f"→  {out_path.name}"
    )
    return {"n": int(emb_arr.shape[0]), "dim": int(emb_arr.shape[1])}


def extract_all_from_videos(
    ckpt_path: Path = DEFAULT_CKPT,
    video_root: Path = VIDEO_ROOT,
    out_dir: Path = OUT_DIR,
    fps: float = 60.0,
    sessions: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    overwrite: bool = False,
) -> None:
    """
    Extract embeddings for all available sessions from video files.

    Args:
        ckpt_path: Path to M5 checkpoint
        video_root: Root directory of UL-DD videos
        out_dir: Output directory for .npz files
        fps: Target frames per second
        sessions: List of sessions to process (e.g., ['A_A', 'B_D'])
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        overwrite: Overwrite existing .npz files
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"M5 checkpoint not found: {ckpt_path}")

    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    print(f"[M6] Loading backbone from {ckpt_path}")
    backbone = YoloClsBackbone(ckpt_path)
    print(f"[M6] Backbone embedding dim = {backbone.embed_dim}")
    print(f"[M6] Extraction fps = {fps}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[M6] Output directory: {out_dir}\n")

    todo_sessions = []
    for subject in sorted(SUBJECTS):
        if subject in NO_VIDEO:
            continue
        for session_type in ["A", "D"]:
            session_name = f"{subject}_{session_type}"
            if sessions and session_name not in sessions:
                continue

            video_path = video_root / subject / session_type
            if not video_path.exists():
                continue

            # Find IR video
            ir_video = video_path / f"{subject}_IR_{session_type}.mp4"
            if not ir_video.exists():
                continue

            todo_sessions.append((subject, session_type, ir_video))

    print(f"[M6] Found {len(todo_sessions)} session(s) to process\n")

    for subject, session_type, video_path in todo_sessions:
        out_path = out_dir / f"{subject}_{session_type}.npz"

        if out_path.exists() and not overwrite:
            print(f"  [skip] {subject}/{session_type}: already extracted")
            continue

        extract_session_from_video(
            backbone,
            video_path,
            out_path,
            subject=subject,
            session=session_type,
            fps=fps,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    print(f"\n[M6] ✓ Done. Embeddings saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"Path to YOLOv8-cls checkpoint (default: {DEFAULT_CKPT.relative_to(ROOT)})",
    )
    p.add_argument(
        "--video-root",
        type=Path,
        default=VIDEO_ROOT,
        help=f"Root path to UL-DD videos (default: {VIDEO_ROOT})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory for embeddings (default: {OUT_DIR.relative_to(ROOT)})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Target fps for extraction (default: 60.0)",
    )
    p.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Subset of sessions, e.g., A_A B_D C_A",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (default: 0)"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing embeddings"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_cli(argv)
    extract_all_from_videos(
        ckpt_path=args.ckpt,
        video_root=args.video_root,
        out_dir=args.out_dir,
        fps=args.fps,
        sessions=args.sessions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
