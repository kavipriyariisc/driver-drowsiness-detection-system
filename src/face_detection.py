"""
Face Detection & Real-Time Feature Extraction
=============================================
YOLOv8  → face bounding box
MediaPipe FaceMesh → 468 landmarks → 10 drowsiness features

10 real-time features (for M4 BiLSTM, 4 Hz):
    0  EAR_left      — Eye Aspect Ratio, left eye
    1  EAR_right     — Eye Aspect Ratio, right eye
    2  MAR           — Mouth Aspect Ratio (yawn proxy)
    3  PERCLOS       — % eye closure over rolling 2-s window
    4  blink_rate    — blinks per minute (rolling 60-s)
    5  head_pitch    — head pitch angle (nodding)
    6  head_roll     — head roll angle (tilting)
    7  head_yaw      — head yaw angle (turning)
    8  brow_raise    — average brow-to-eye distance (normalised)
    9  jaw_drop      — chin-to-nose distance (yawn / fatigue marker)
"""

import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Graceful imports ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    print("[face_detection] ultralytics not installed — YOLO disabled.")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_OK = True
except ImportError:
    _MP_OK = False
    print("[face_detection] mediapipe not installed — FaceMesh disabled.")

_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_LANDMARKER_PATH = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")


def _ensure_landmarker_model():
    """Download face_landmarker.task if not already present."""
    from pathlib import Path as _Path
    p = _Path(_LANDMARKER_PATH)
    if p.exists():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe] Downloading FaceLandmarker model → {p}")
    import urllib.request
    urllib.request.urlretrieve(_LANDMARKER_MODEL_URL, str(p))
    print("[mediapipe] Download complete.")
    return str(p)


# ─── MediaPipe landmark indices ───────────────────────────────────────────────
# Right eye (from MediaPipe 468-point mesh)
_RE = [33,  160, 158, 133, 153, 144]
# Left eye
_LE = [362, 385, 387, 263, 373, 380]
# Mouth outer corners + top/bottom
_MO = [61, 291, 0, 17]
# Brow landmarks (right / left mid)
_RB_TOP = 105; _RB_BOT = 33
_LB_TOP = 334; _LB_BOT = 263
# Jaw & nose tip
_CHIN = 152; _NOSE = 1


def _ear(lm, idx):
    """Eye Aspect Ratio from 6 landmark indices."""
    p = [np.array([lm[i].x, lm[i].y]) for i in idx]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(lm):
    """Mouth Aspect Ratio from 4 outer landmark indices."""
    p = [np.array([lm[i].x, lm[i].y]) for i in _MO]
    V = np.linalg.norm(p[2] - p[3])   # vertical
    H = np.linalg.norm(p[0] - p[1])   # horizontal
    return V / (H + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# FaceDetector  (YOLOv8)
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """YOLOv8-based face detector."""

    def __init__(self, model_path: str = 'yolov8n.pt', conf: float = 0.45):
        if not _YOLO_OK:
            raise RuntimeError("ultralytics is required for FaceDetector.")
        self.model = YOLO(model_path)
        self.conf  = conf
        print(f"✓ FaceDetector loaded: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns list of (x1, y1, x2, y2, confidence) for each detected face.
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        boxes   = []
        if results and len(results[0].boxes):
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().tolist())
                boxes.append((x1, y1, x2, y2, float(det.conf[0].cpu())))
        return boxes

    def crop_face(self, frame: np.ndarray, box: Tuple, pad: float = 0.15) -> np.ndarray:
        """Crop face region with optional padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box[:4]
        pw = int((x2 - x1) * pad)
        ph = int((y2 - y1) * pad)
        x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw); y2 = min(h, y2 + ph)
        return frame[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor  (MediaPipe FaceMesh → 10 drowsiness features)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts 10 real-time drowsiness features from a face crop
    using MediaPipe FaceMesh (468 landmarks, 3D).
    """

    EAR_CLOSED_THRESH = 0.20   # below → eye considered closed
    PERCLOS_WINDOW_S  = 2.0    # PERCLOS rolling window (seconds)
    BLINK_WINDOW_S    = 60.0   # blink-rate window (seconds)
    TARGET_HZ         = 4      # inference rate

    def __init__(self):
        if not _MP_OK:
            raise RuntimeError("mediapipe is required for FeatureExtractor.")
        model_path = _ensure_landmarker_model()
        base_opts  = _mp_python.BaseOptions(model_asset_path=model_path)
        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=_mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
        # History buffers
        _cap    = int(self.PERCLOS_WINDOW_S * self.TARGET_HZ * 30)  # oversized
        self._ear_hist   : deque = deque(maxlen=_cap)
        self._blink_ts   : deque = deque(maxlen=int(self.BLINK_WINDOW_S * self.TARGET_HZ * 10))
        self._prev_closed: bool  = False
        self._frame_ts         = time.time()

    def extract(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Process one face crop and return 10-feature vector or None.

        Returns:
            np.ndarray (10,) float32, or None if no face detected
        """
        rgb    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        ear_l = _ear(lm, _LE)
        ear_r = _ear(lm, _RE)
        mar   = _mar(lm)
        avg_ear = (ear_l + ear_r) / 2.0

        # ── PERCLOS (% frames eye closed over rolling window) ─────────────
        now   = time.time()
        self._ear_hist.append((now, avg_ear))
        cutoff_p = now - self.PERCLOS_WINDOW_S
        recent   = [e for t, e in self._ear_hist if t >= cutoff_p]
        perclos  = np.mean([1 if e < self.EAR_CLOSED_THRESH else 0
                             for e in recent]) if recent else 0.0

        # ── Blink detection ───────────────────────────────────────────────
        closed = avg_ear < self.EAR_CLOSED_THRESH
        if closed and not self._prev_closed:
            self._blink_ts.append(now)
        self._prev_closed = closed

        cutoff_b   = now - self.BLINK_WINDOW_S
        blinks     = sum(1 for t in self._blink_ts if t >= cutoff_b)
        blink_rate = blinks / (self.BLINK_WINDOW_S / 60.0)   # blinks / min

        # ── Head pose (from MediaPipe 3D landmarks) ───────────────────────
        # Use nose-tip (1), chin (152), left-eye-outer (263), right-eye-outer (33)
        # as a simple rotation estimator via PnP geometry approximation
        pts_3d = np.float32([
            [0, 0, 0],       # nose
            [0, -63, -13],   # chin
            [-43, 32, -26],  # right corner
            [43,  32, -26],  # left corner
        ])
        pts_2d = np.float32([
            [lm[1].x   * face_crop.shape[1], lm[1].y   * face_crop.shape[0]],
            [lm[152].x * face_crop.shape[1], lm[152].y * face_crop.shape[0]],
            [lm[33].x  * face_crop.shape[1], lm[33].y  * face_crop.shape[0]],
            [lm[263].x * face_crop.shape[1], lm[263].y * face_crop.shape[0]],
        ])
        fx = face_crop.shape[1]
        cam = np.float32([[fx, 0, fx/2], [0, fx, face_crop.shape[0]/2], [0, 0, 1]])
        _, rvec, _ = cv2.solvePnP(pts_3d, pts_2d, cam,
                                   np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP)
        rmat, _  = cv2.Rodrigues(rvec)
        angles   = cv2.decomposeProjectionMatrix(
            np.hstack([rmat, np.zeros((3, 1))])
        )[6].flatten()
        head_pitch = float(angles[0])   # nodding
        head_roll  = float(angles[1])   # tilting
        head_yaw   = float(angles[2])   # turning

        # ── Brow raise (normalised by face height) ────────────────────────
        fh = face_crop.shape[0]
        rb = abs(lm[_RB_TOP].y - lm[_RB_BOT].y) * fh
        lb = abs(lm[_LB_TOP].y - lm[_LB_BOT].y) * fh
        brow_raise = ((rb + lb) / 2.0) / (fh + 1e-6)

        # ── Jaw drop (chin-to-nose distance, normalised) ──────────────────
        fw = face_crop.shape[1]
        nose_y  = lm[_NOSE].y  * fh
        chin_y  = lm[_CHIN].y  * fh
        jaw_drop = (chin_y - nose_y) / (fh + 1e-6)

        features = np.array([
            ear_l, ear_r, mar, perclos, blink_rate,
            head_pitch, head_roll, head_yaw, brow_raise, jaw_drop,
        ], dtype=np.float32)

        return features


# ─────────────────────────────────────────────────────────────────────────────
# FacialAnalyzer  (combines YOLO + FaceMesh, emits feature dicts)
# ─────────────────────────────────────────────────────────────────────────────

class FacialAnalyzer:
    """
    High-level wrapper: detects face with YOLO, extracts 10 features
    with MediaPipe FaceMesh, and returns a per-frame feature dict.
    """

    def __init__(self, yolo_model: str = 'yolov8n.pt'):
        self.detector  = FaceDetector(yolo_model)
        self.extractor = FeatureExtractor()

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Process one BGR frame.

        Returns:
            dict with keys:
                faces_detected (int)
                features       (np.ndarray (10,) or None)
                boxes          (list of (x1,y1,x2,y2,conf))
                summary        (dict of named feature values)
        """
        boxes    = self.detector.detect(frame)
        features = None
        summary  = {}

        if boxes:
            box   = boxes[0]           # use largest / first face
            crop  = self.detector.crop_face(frame, box)
            if crop.size > 0:
                features = self.extractor.extract(crop)
                if features is not None:
                    keys = ['ear_left', 'ear_right', 'mar', 'perclos', 'blink_rate',
                            'head_pitch', 'head_roll', 'head_yaw', 'brow_raise', 'jaw_drop']
                    summary = dict(zip(keys, features.tolist()))

        return {
            'faces_detected': len(boxes),
            'features'      : features,
            'boxes'         : boxes,
            'summary'       : summary,
        }

    def draw_overlay(
        self,
        frame    : np.ndarray,
        analysis : Dict,
        label    : str  = '',
        color    : Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw bounding box, feature values, and optional prediction label."""
        out = frame.copy()

        for x1, y1, x2, y2, conf in analysis['boxes']:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"face {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        s = analysis['summary']
        y = 25
        for key, val in s.items():
            cv2.putText(out, f"{key}: {val:.3f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)
            y += 18

        if label:
            c = {'Alert': (0, 200, 0), 'LowVigilant': (0, 165, 255),
                 'Drowsy': (0, 0, 255)}.get(label, (200, 200, 200))
            cv2.putText(out, label, (10, out.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)

        return out
