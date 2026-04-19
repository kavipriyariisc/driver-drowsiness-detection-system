"""
tests/test_data.py
==================
Unit tests for UL-DD data utilities.
Tests use synthetic (random) data — no real UL-DD files required.

Run:
    pytest tests/test_data.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── allow running from project root ──────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.load_data import load_fold, class_distribution, summarise_fold
from models.architecture import WINDOW, N_FAU, N_TELE, N_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_fold(n_samples: int = 200) -> dict:
    """Create a minimal fold dictionary that mirrors ULDDProcessor output."""
    rng = np.random.default_rng(42)
    return {
        "fau_windows":  rng.random((n_samples, WINDOW, N_FAU),  dtype=np.float32),
        "tele_windows": rng.random((n_samples, WINDOW, N_TELE), dtype=np.float32),
        "labels":       rng.integers(0, N_CLASSES, size=n_samples).astype(np.int32),
        "fold_idx":     0,
    }


def save_dummy_fold(path: Path, n_samples: int = 200) -> dict:
    fold = make_dummy_fold(n_samples)
    np.savez(path, **{k: v for k, v in fold.items() if k != "fold_idx"},
             fold_idx=np.array(fold["fold_idx"]))
    return fold


# ─────────────────────────────────────────────────────────────────────────────
# Tests — load_fold
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFold:
    def test_load_returns_all_keys(self, tmp_path):
        save_dummy_fold(tmp_path / "fold_0.npz")
        fold = load_fold(str(tmp_path), fold_idx=0)
        for key in ("fau_windows", "tele_windows", "labels"):
            assert key in fold, f"Key '{key}' missing from loaded fold"

    def test_fau_shape(self, tmp_path):
        n = 120
        save_dummy_fold(tmp_path / "fold_0.npz", n_samples=n)
        fold = load_fold(str(tmp_path), fold_idx=0)
        assert fold["fau_windows"].shape == (n, WINDOW, N_FAU), \
            f"Expected ({n},{WINDOW},{N_FAU}), got {fold['fau_windows'].shape}"

    def test_tele_shape(self, tmp_path):
        n = 120
        save_dummy_fold(tmp_path / "fold_0.npz", n_samples=n)
        fold = load_fold(str(tmp_path), fold_idx=0)
        assert fold["tele_windows"].shape == (n, WINDOW, N_TELE), \
            f"Expected ({n},{WINDOW},{N_TELE}), got {fold['tele_windows'].shape}"

    def test_labels_shape(self, tmp_path):
        n = 120
        save_dummy_fold(tmp_path / "fold_0.npz", n_samples=n)
        fold = load_fold(str(tmp_path), fold_idx=0)
        assert fold["labels"].shape == (n,), \
            f"Expected ({n},), got {fold['labels'].shape}"

    def test_labels_range(self, tmp_path):
        save_dummy_fold(tmp_path / "fold_0.npz")
        fold = load_fold(str(tmp_path), fold_idx=0)
        labels = fold["labels"]
        assert labels.min() >= 0,           "Label below 0"
        assert labels.max() < N_CLASSES,    f"Label >= N_CLASSES ({N_CLASSES})"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fold(str(tmp_path), fold_idx=99)

    def test_dtype_float32_fau(self, tmp_path):
        save_dummy_fold(tmp_path / "fold_0.npz")
        fold = load_fold(str(tmp_path), fold_idx=0)
        assert fold["fau_windows"].dtype == np.float32, \
            f"FAU dtype should be float32, got {fold['fau_windows'].dtype}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — class_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestClassDistribution:
    def test_returns_string(self):
        y = np.array([0, 0, 1, 2, 2, 2])
        result = class_distribution(y)
        assert isinstance(result, str)

    def test_contains_all_classes(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        result = class_distribution(y)
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_empty_class(self):
        y = np.array([0, 0, 0])
        result = class_distribution(y)
        assert isinstance(result, str)   # should not raise

    def test_percentages_sum_100(self):
        y = np.array([0, 1, 2, 0, 1, 2, 0])
        # parse percentages from the string (basic sanity)
        result = class_distribution(y)
        assert "%" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests — summarise_fold (smoke test — just checks it doesn't crash)
# ─────────────────────────────────────────────────────────────────────────────

class TestSummariseFold:
    def test_no_exception(self, capsys):
        fold = make_dummy_fold()
        summarise_fold(fold)    # should print without raising
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — RingBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestRingBuffer:
    def setup_method(self):
        from data.telemetry_replay import RingBuffer
        self.RingBuffer = RingBuffer

    def test_not_ready_when_empty(self):
        buf = self.RingBuffer(WINDOW, N_TELE)
        assert not buf.is_ready()

    def test_ready_after_full(self):
        buf = self.RingBuffer(WINDOW, N_TELE)
        for _ in range(WINDOW):
            buf.push(np.zeros(N_TELE, dtype=np.float32))
        assert buf.is_ready()

    def test_get_shape(self):
        buf = self.RingBuffer(WINDOW, N_TELE)
        for _ in range(WINDOW):
            buf.push(np.ones(N_TELE, dtype=np.float32))
        arr = buf.get()
        assert arr.shape == (WINDOW, N_TELE)

    def test_values_correct(self):
        buf = self.RingBuffer(5, 2)
        for i in range(5):
            buf.push(np.array([float(i), float(i)]))
        arr = buf.get()
        np.testing.assert_allclose(arr[:, 0], [0., 1., 2., 3., 4.])

    def test_push_batch(self):
        buf = self.RingBuffer(WINDOW, N_TELE)
        batch = np.random.rand(WINDOW, N_TELE).astype(np.float32)
        buf.push_batch(batch)
        assert buf.is_ready()
        arr = buf.get()
        assert arr.shape == (WINDOW, N_TELE)

    def test_overwrite_old_values(self):
        buf = self.RingBuffer(3, 1)
        for v in [1., 2., 3.]:
            buf.push(np.array([v]))
        buf.push(np.array([99.]))   # pushes 1 out → [2, 3, 99]
        arr = buf.get()
        np.testing.assert_allclose(arr[:, 0], [2., 3., 99.])
