"""
tests/test_models.py
====================
Unit tests for M1, M2, M3, M4 model architectures.
All tests use random tensors — no trained weights or data files required.

Run:
    pytest tests/test_models.py -v
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tensorflow as tf
from models.architecture import (
    build_m1_facial_bilstm,
    build_m2_telemetry_lstm,
    M3FusionModel,
    build_m4_realtime_bilstm,
    build_model,
    WINDOW, N_FAU, N_TELE, N_CLASSES, N_RT_FEAT,
)
from models.train import compile_model, make_callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BATCH = 8

def rand_fau()  -> np.ndarray: return np.random.rand(BATCH, WINDOW, N_FAU).astype(np.float32)
def rand_tele() -> np.ndarray: return np.random.rand(BATCH, WINDOW, N_TELE).astype(np.float32)
def rand_rt()   -> np.ndarray: return np.random.rand(BATCH, WINDOW, N_RT_FEAT).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# M1 — Facial BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestM1:
    @pytest.fixture(scope="class")
    def model(self):
        return build_m1_facial_bilstm()

    def test_output_shape(self, model):
        out = model(rand_fau(), training=False)
        assert out.shape == (BATCH, N_CLASSES), f"M1 output: {out.shape}"

    def test_output_is_probability(self, model):
        out = model(rand_fau(), training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=-1), 1., atol=1e-5,
                                   err_msg="M1 output rows must sum to 1")

    def test_model_name(self, model):
        assert "M1" in model.name, f"Unexpected name: {model.name}"

    def test_parameter_count(self, model):
        params = model.count_params()
        assert params > 100_000, f"M1 seems too small: {params}"
        assert params < 2_000_000, f"M1 seems too large: {params}"

    def test_compile_no_error(self, model):
        compile_model(model)

    def test_single_step_train(self, model):
        compile_model(model)
        y = np.random.randint(0, N_CLASSES, BATCH).astype(np.int32)
        history = model.fit(rand_fau(), y, epochs=1, verbose=0)
        assert "loss" in history.history

    def test_build_model_factory(self):
        m = build_model('m1')
        assert m is not None


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Telemetry LSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestM2:
    @pytest.fixture(scope="class")
    def model(self):
        return build_m2_telemetry_lstm()

    def test_output_shape(self, model):
        out = model(rand_tele(), training=False)
        assert out.shape == (BATCH, N_CLASSES), f"M2 output: {out.shape}"

    def test_output_is_probability(self, model):
        out = model(rand_tele(), training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=-1), 1., atol=1e-5,
                                   err_msg="M2 output rows must sum to 1")

    def test_model_name(self, model):
        assert "M2" in model.name

    def test_parameter_count(self, model):
        params = model.count_params()
        assert params > 10_000
        assert params < 500_000

    def test_compile_no_error(self, model):
        compile_model(model)

    def test_single_step_train(self, model):
        compile_model(model)
        y = np.random.randint(0, N_CLASSES, BATCH).astype(np.int32)
        history = model.fit(rand_tele(), y, epochs=1, verbose=0)
        assert "loss" in history.history

    def test_build_model_factory(self):
        m = build_model('m2')
        assert m is not None


# ─────────────────────────────────────────────────────────────────────────────
# M3 — Cross-Modal Fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestM3:
    @pytest.fixture(scope="class")
    def model(self):
        return M3FusionModel()

    def test_output_shape(self, model):
        out = model([rand_fau(), rand_tele()], training=False)
        assert out.shape == (BATCH, N_CLASSES), f"M3 output: {out.shape}"

    def test_output_is_probability(self, model):
        out = model([rand_fau(), rand_tele()], training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=-1), 1., atol=1e-5,
                                   err_msg="M3 output rows must sum to 1")

    def test_model_name(self, model):
        assert "M3" in model.name

    def test_parameter_count(self, model):
        # need a forward pass to build
        model([rand_fau(), rand_tele()], training=False)
        params = model.count_params()
        assert params > 200_000
        assert params < 5_000_000

    def test_compile_no_error(self, model):
        compile_model(model)

    def test_single_step_train(self, model):
        compile_model(model)
        y = np.random.randint(0, N_CLASSES, BATCH).astype(np.int32)
        history = model.fit([rand_fau(), rand_tele()], y, epochs=1, verbose=0)
        assert "loss" in history.history

    def test_build_model_factory(self):
        m = build_model('m3')
        assert m is not None

    def test_attention_sub_layers_exist(self, model):
        # force build
        model([rand_fau(), rand_tele()], training=False)
        layer_names = [l.name for l in model.layers]
        assert any("attn" in n for n in layer_names), \
            f"No attention layers found: {layer_names}"

    def test_get_config_roundtrip(self, model):
        cfg = model.get_config()
        assert "d_attn" in cfg
        assert "fau_units" in cfg


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Real-Time BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestM4:
    @pytest.fixture(scope="class")
    def model(self):
        return build_m4_realtime_bilstm()

    def test_output_shape(self, model):
        out = model(rand_rt(), training=False)
        assert out.shape == (BATCH, N_CLASSES), f"M4 output: {out.shape}"

    def test_output_is_probability(self, model):
        out = model(rand_rt(), training=False).numpy()
        np.testing.assert_allclose(out.sum(axis=-1), 1., atol=1e-5,
                                   err_msg="M4 output rows must sum to 1")

    def test_model_name(self, model):
        assert "M4" in model.name

    def test_parameter_count(self, model):
        params = model.count_params()
        assert params > 20_000
        assert params < 500_000

    def test_compile_no_error(self, model):
        compile_model(model)

    def test_single_step_train(self, model):
        compile_model(model)
        y = np.random.randint(0, N_CLASSES, BATCH).astype(np.int32)
        history = model.fit(rand_rt(), y, epochs=1, verbose=0)
        assert "loss" in history.history

    def test_build_model_factory(self):
        m = build_model('m4')
        assert m is not None


# ─────────────────────────────────────────────────────────────────────────────
# build_model factory — error case
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModelFactory:
    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("m99")

    def test_case_insensitive(self):
        for name in ("M1", "M2", "M3", "M4"):
            m = build_model(name)
            assert m is not None


# ─────────────────────────────────────────────────────────────────────────────
# make_callbacks — smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbacks:
    def test_callbacks_are_list(self, tmp_path):
        cbs = make_callbacks(str(tmp_path / "best.keras"))
        assert isinstance(cbs, list)
        assert len(cbs) >= 2

    def test_includes_early_stopping(self, tmp_path):
        from tensorflow.keras.callbacks import EarlyStopping
        cbs = make_callbacks(str(tmp_path / "best.keras"))
        types = [type(c).__name__ for c in cbs]
        assert "EarlyStopping" in types, f"Expected EarlyStopping in {types}"

    def test_includes_checkpoint(self, tmp_path):
        cbs = make_callbacks(str(tmp_path / "best.keras"))
        types = [type(c).__name__ for c in cbs]
        assert "ModelCheckpoint" in types, f"Expected ModelCheckpoint in {types}"
