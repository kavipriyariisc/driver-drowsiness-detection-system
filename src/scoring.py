"""
Drowsiness Scoring & Alert System — UL-DD Project
==================================================

DrowsinessScorer:
    Converts model output (3-class probabilities) into a
    0-100 drowsiness score with progressive alert levels.

AlertGenerator:
    Generates alert messages and severity levels.
"""

import numpy as np
from typing import Dict, Tuple

CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']

# Class weights for the 0-100 composite score
# Alert=0  Low Vigilant=50  Drowsy=100
CLASS_SCORE_MAP = {0: 0.0, 1: 50.0, 2: 100.0}


class DrowsinessScorer:
    """
    Converts 3-class softmax probabilities into a smooth 0-100 score
    and tracks a rolling history for trend analysis.

    Score = Σ (p_i × CLASS_SCORE_MAP[i])  for i in {0,1,2}
    """

    LEVELS = [
        (0,   25,  'Alert',       'none'),
        (25,  50,  'Mild',        'info'),
        (50,  70,  'Moderate',    'warning'),
        (70,  85,  'Severe',      'critical'),
        (85, 101,  'Very Severe', 'critical'),
    ]

    def __init__(self, history_len: int = 20):
        """
        Args:
            history_len : Rolling window size for smoothing (in inference steps).
        """
        self.history_len   = history_len
        self._score_history: list = []

    # ── Core scoring ──────────────────────────────────────────────────────────

    def probs_to_score(self, probs: np.ndarray) -> float:
        """
        Convert (3,) softmax probability vector to 0-100 score.

        Args:
            probs : np.ndarray of shape (3,) — [p_alert, p_low, p_drowsy]

        Returns:
            float in [0, 100]
        """
        score = (probs[0] * CLASS_SCORE_MAP[0]
                 + probs[1] * CLASS_SCORE_MAP[1]
                 + probs[2] * CLASS_SCORE_MAP[2])
        return float(np.clip(score, 0, 100))

    def update(self, probs: np.ndarray) -> float:
        """
        Push new probabilities; returns the smoothed score.
        """
        raw = self.probs_to_score(probs)
        self._score_history.append(raw)
        if len(self._score_history) > self.history_len:
            self._score_history.pop(0)
        return self.smoothed_score

    @property
    def smoothed_score(self) -> float:
        """Exponentially weighted average of score history."""
        if not self._score_history:
            return 0.0
        weights = np.exp(np.linspace(-1, 0, len(self._score_history)))
        weights /= weights.sum()
        return float(np.dot(weights, self._score_history))

    @property
    def trend(self) -> str:
        """'rising' | 'falling' | 'stable'"""
        if len(self._score_history) < 4:
            return 'stable'
        slope = np.polyfit(range(len(self._score_history)),
                           self._score_history, 1)[0]
        if slope > 1.5:
            return 'rising ↑'
        if slope < -1.5:
            return 'falling ↓'
        return 'stable →'

    def get_level(self, score: float) -> Tuple[str, str]:
        """
        Map score to (level_name, severity) pair.

        Returns:
            ('Moderate', 'warning') etc.
        """
        for lo, hi, name, severity in self.LEVELS:
            if lo <= score < hi:
                return name, severity
        return 'Very Severe', 'critical'

    def reset(self) -> None:
        self._score_history.clear()


class AlertGenerator:
    """
    Generates human-readable alert messages from drowsiness score.
    """

    MESSAGES = {
        'none'    : '✓  Alert — no action required.',
        'info'    : '⚠  Mild drowsiness detected. Consider a short break.',
        'warning' : '⚠⚠ Moderate drowsiness! Take a break soon.',
        'critical': '🔴 SEVERE DROWSINESS — STOP DRIVING IMMEDIATELY!',
    }
    COLORS = {          # BGR for OpenCV overlay
        'none'    : (0, 200, 0),
        'info'    : (0, 200, 200),
        'warning' : (0, 165, 255),
        'critical': (0, 0, 255),
    }

    def generate(self, score: float, level: str, severity: str) -> Dict:
        """
        Build alert dict from scorer output.

        Returns:
            dict: score, level, severity, message, color_bgr
        """
        return {
            'score'    : score,
            'level'    : level,
            'severity' : severity,
            'message'  : self.MESSAGES.get(severity, ''),
            'color_bgr': self.COLORS.get(severity, (200, 200, 200)),
        }
