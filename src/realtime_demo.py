"""
Real-Time Drowsiness Detection Demo — UL-DD Project
=====================================================

Live pipeline:
    Webcam frame → YOLOv8 face detect → MediaPipe 10-feat extract
    UL-DD telemetry CSV replay → 4 Hz ring buffer
    M3 (or M1/M4) inference every STRIDE_S seconds
    Overlay: prediction label + score + CAN values

Run:
    python src/realtime_demo.py

Configure TELEMETRY_CSV and MODEL_PATH constants at the bottom.
"""

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_detection import FacialAnalyzer
from scoring import DrowsinessScorer, AlertGenerator
from data.telemetry_replay import TelemetryReplayer, RingBuffer
from models.architecture import WINDOW, N_FAU, N_TELE, N_RT_FEAT, CLASS_NAMES

# Optional: load trained model
try:
    from inference.predict import ULDDPredictor
    _PREDICTOR_OK = True
except Exception:
    _PREDICTOR_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_HZ = 4          # sample rate into ring buffer
STRIDE_S  = 15.0       # run inference every 15 s (match training stride)
FONT      = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# RealtimeDemoApp
# ─────────────────────────────────────────────────────────────────────────────

class RealtimeDemoApp:
    """
    Combines:
        - YOLOv8 face detection + MediaPipe feature extraction (FacialAnalyzer)
        - UL-DD telemetry CSV replay (TelemetryReplayer + RingBuffer)
        - Optional M3 BiLSTM inference (ULDDPredictor)
        - Score smoothing + alert generation
        - OpenCV display with HUD overlay
    """

    def __init__(
        self,
        telemetry_csv  : Optional[str] = None,
        model_path     : Optional[str] = None,
        model_type     : str           = 'm3',
        yolo_model     : str           = 'yolov8n.pt',
        speed_factor   : float         = 1.0,
    ):
        print("\n" + "="*60)
        print("  UL-DD Real-Time Drowsiness Detection Demo")
        print("="*60)

        # ── Face analysis ─────────────────────────────────────────────────
        self.analyzer  = FacialAnalyzer(yolo_model)
        self.scorer    = DrowsinessScorer(history_len=20)
        self.alerter   = AlertGenerator()

        # ── Telemetry replay (disabled for M5-only test) ─────────────────
        self.replayer: Optional[TelemetryReplayer] = None
        self._latest_tele = dict(pitch=0., roll=0., speed=0., rpm=0., gear=0.)

        # if telemetry_csv and Path(telemetry_csv).exists():
        #     self.replayer = TelemetryReplayer(
        #         telemetry_csv, loop=True, speed_factor=speed_factor
        #     )
        #     print(f"✓ Telemetry replay: {Path(telemetry_csv).name}")
        # else:
        #     print("ℹ  No telemetry CSV — CAN will be simulated (random walk).")

        # ── Ring buffers @ 4 Hz (not used by M5) ─────────────────────────
        self.rt_buf   = RingBuffer(WINDOW, N_RT_FEAT)   # for M4
        self.tele_buf = RingBuffer(WINDOW, N_TELE)      # for M2/M3

        # ── Predictor (optional) ──────────────────────────────────────────
        self.predictor: Optional[ULDDPredictor] = None
        self.model_type = model_type.lower()

        if model_path and Path(model_path).exists() and _PREDICTOR_OK:
            self.predictor = ULDDPredictor(model_path, model_type)
            print(f"✓ Model loaded: {Path(model_path).name}")
        else:
            print("ℹ  No trained model — rule-based scoring only.")

        # ── Inference timing ──────────────────────────────────────────────
        self._last_inference = time.time()
        self._current_label  = 'Alert'
        self._current_score  = 0.0
        self._current_alert  = self.alerter.generate(0, 'Alert', 'none')

        # ── Latest face crop (used by M5 per-frame inference) ─────────────
        self._latest_face_crop: Optional[np.ndarray] = None

        # ── 4 Hz sampling timer ───────────────────────────────────────────
        self._last_sample = time.time()
        self._sample_interval = 1.0 / TARGET_HZ   # 0.25 s
        # M5 runs at 1 Hz (image classifier, no window needed)
        self._m5_stride = 1.0

        print("="*60 + "\n")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _tick_sample(self, features: Optional[np.ndarray]) -> bool:
        """Returns True if a 4-Hz sample interval has elapsed."""
        now = time.time()
        if now - self._last_sample < self._sample_interval:
            return False
        self._last_sample = now

        # Push face features into RT buffer
        if features is not None:
            self.rt_buf.push(features)
        else:
            self.rt_buf.push(np.zeros(N_RT_FEAT, dtype=np.float32))

        # Telemetry drain/simulate disabled for M5-only test
        # if self.replayer:
        #     rows = self.replayer.drain_to_array()
        #     if len(rows):
        #         self.tele_buf.push_batch(rows)
        #         t = rows[-1]
        #         self._latest_tele = dict(
        #             pitch=float(t[0]), roll=float(t[1]),
        #             speed=float(t[2]), rpm=float(t[3]), gear=float(t[4])
        #         )
        # else:
        #     for k, lo, hi, step in [('speed', 30, 90, 2), ('rpm', 800, 3500, 50),
        #                               ('pitch', -5, 5, .3), ('roll', -5, 5, .3),
        #                               ('gear', 1, 6, 0)]:
        #         self._latest_tele[k] = float(np.clip(
        #             self._latest_tele[k] + np.random.uniform(-step, step), lo, hi
        #         ))
        #     self.tele_buf.push(np.array(
        #         [self._latest_tele[k] for k in ['pitch', 'roll', 'speed', 'rpm', 'gear']],
        #         dtype=np.float32
        #     ))
        return True

    def _run_inference(self) -> None:
        """Run model inference if STRIDE_S (or M5's 1 Hz cadence) has elapsed."""
        now = time.time()
        # M5 doesn't need a ring-buffer window — check it separately
        if self.model_type != 'm5':
            if not self.rt_buf.is_ready():
                return
            if now - self._last_inference < STRIDE_S:
                return
            self._last_inference = now

        if self.predictor:
            mt = self.model_type
            try:
                if mt == 'm5':
                    if self._latest_face_crop is None:
                        return
                    # M5 uses a 1 Hz cadence instead of STRIDE_S
                    if now - self._last_inference < self._m5_stride:
                        return
                    self._last_inference = now
                    probs = self.predictor.predict_image(self._latest_face_crop)
                    self._current_score = self.scorer.update(probs)
                elif mt == 'm1':
                    # M1 expects FAU 30-feat — fall back to rule-based if no FAU buffer
                    pass
                elif mt == 'm4':
                    w     = self.rt_buf.get()[np.newaxis]   # (1,240,10)
                    probs = self.predictor.model.predict(w, verbose=0)[0]
                    self._current_score = self.scorer.update(probs)
                elif mt == 'm3' and self.tele_buf.is_ready():
                    wf = self.rt_buf.get()[np.newaxis].astype(np.float32)
                    wt = self.tele_buf.get()[np.newaxis].astype(np.float32)
                    probs = self.predictor.model.predict([wf, wt], verbose=0)[0]
                    self._current_score = self.scorer.update(probs)
                else:
                    return
            except Exception as e:
                print(f"[inference] {e}")
                return
        else:
            # Rule-based fallback: use EAR + speed deviation
            rt = self.rt_buf.get()   # (240, 10)
            avg_ear = rt[:, :2].mean()
            p = np.array([
                max(0, avg_ear - 0.15),          # alert proxy
                max(0, 0.25 - abs(avg_ear - 0.2)),
                max(0, 0.20 - avg_ear),           # drowsy proxy
            ], dtype=np.float32)
            p /= p.sum() + 1e-6
            self._current_score = self.scorer.update(p)

        level, severity = self.scorer.get_level(self._current_score)
        self._current_label  = level
        self._current_alert  = self.alerter.generate(self._current_score, level, severity)

    # ─────────────────────────────────────────────────────────────────────
    # HUD drawing
    # ─────────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, analysis: dict) -> np.ndarray:
        out   = frame.copy()
        color = self._current_alert['color_bgr']

        # Face boxes
        for x1, y1, x2, y2, conf in analysis['boxes']:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Feature panel (left side)
        s = analysis['summary']
        y = 22
        for key, val in s.items():
            cv2.putText(out, f"{key}: {val:.3f}", (10, y), FONT, 0.40,
                        (200, 230, 200), 1)
            y += 16

        # Telemetry panel disabled for M5-only test
        # t = self._latest_tele
        # src = "Replay" if self.replayer else "Simulated"
        # tele_lines = [
        #     f"[CAN | {src}]",
        #     f"Speed: {t['speed']:.1f} m/s",
        #     f"RPM  : {t['rpm']:.0f}",
        #     f"Gear : {t['gear']:.0f}",
        #     f"Pitch: {t['pitch']:.1f}°",
        #     f"Roll : {t['roll']:.1f}°",
        # ]
        # tw = out.shape[1]
        # for i, line in enumerate(tele_lines):
        #     cv2.putText(out, line, (tw - 230, 22 + i * 18), FONT, 0.42,
        #                 (200, 200, 50), 1)

        # Prediction panel (bottom)
        bh = out.shape[0]
        cv2.rectangle(out, (0, bh - 60), (out.shape[1], bh), (20, 20, 20), -1)
        cv2.putText(out, self._current_label, (10, bh - 32),
                    FONT, 1.1, color, 2)
        cv2.putText(out, f"Score: {self._current_score:.1f}/100  Trend: {self.scorer.trend}",
                    (10, bh - 10), FONT, 0.50, color, 1)
        cv2.putText(out, self._current_alert['message'],
                    (200, bh - 10), FONT, 0.40, color, 1)

        return out

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self, webcam_index: int = 0, output_path: Optional[str] = None):
        """
        Start webcam + telemetry replay loop.
        Press 'q' to quit.
        """
        # if self.replayer:
        #     self.replayer.start()

        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            print("✗ Cannot open webcam.")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)
            )
            print(f"💾 Recording → {output_path}")

        print("📷 Live demo running (press 'q' to quit)…\n")
        t_prev = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                analysis = self.analyzer.analyze_frame(frame)

                # Store latest face crop for M5 per-frame inference
                if analysis['boxes'] and self.model_type == 'm5':
                    x1, y1, x2, y2, _ = analysis['boxes'][0]
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        self._latest_face_crop = crop

                # 4 Hz sampling
                self._tick_sample(analysis['features'])

                # Inference (every STRIDE_S seconds)
                self._run_inference()

                # Overlay
                out = self._draw_hud(frame, analysis)

                # FPS
                now  = time.time()
                fps  = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                cv2.putText(out, f"FPS: {fps:.1f}", (w - 90, 16),
                            FONT, 0.45, (150, 150, 150), 1)

                if writer:
                    writer.write(out)

                cv2.imshow("DDD — Camera + CAN (UL-DD Replay)", out)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == 27:   # 'q' or ESC
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            # if self.replayer:
            #     self.replayer.stop()
            print("Demo stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — configure paths here
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Configure ─────────────────────────────────────────────────────────────
    ULDD_ROOT = Path(r"C:/Users/raka1005/Documents/IISC/UL-DD")

    # Telemetry CSV disabled for M5-only test
    # TELEMETRY_CSV = (
    #     ULDD_ROOT / "CSV_Files" / "CSV_Files" / "CSV_Files"
    #     / "B" / "A" / "B_Telemetry_A.csv"
    # )

    # Trained model checkpoint
    MODEL_PATH  = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'M5_fold0.pt'
    MODEL_TYPE  = 'm5'    # 'm1' | 'm2' | 'm3' | 'm4' | 'm5'
    # ─────────────────────────────────────────────────────────────────────────

    app = RealtimeDemoApp(
        telemetry_csv = None,   # disabled for M5-only test
        model_path    = str(MODEL_PATH),
        model_type    = MODEL_TYPE,
        yolo_model    = 'yolov8n.pt',
        speed_factor  = 1.0,
    )
    app.run(webcam_index=0, output_path='demo_output.mp4')
