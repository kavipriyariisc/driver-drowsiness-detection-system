"""
Model Architectures — UL-DD Multimodal Drowsiness Detection
============================================================

M1 : build_m1_facial_bilstm   — FAU (30-feat) BiLSTM    [camera only]
M2 : build_m2_telemetry_lstm  — CAN (5-feat)  LSTM      [vehicle only]
M3 : M3FusionModel            — Cross-modal attention   [camera + CAN] ★ thesis novelty
M4 : build_m4_realtime_bilstm — YOLO 10-feat BiLSTM     [live demo]

Label schema  (matches UL-DD KSS binning):
    0 → Alert          (KSS 1-3)
    1 → Low Vigilant   (KSS 4-6)
    2 → Drowsy         (KSS 7-9)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2

# ─── Global constants ─────────────────────────────────────────────────────────
WINDOW    = 240   # 60 s × 4 Hz
N_FAU     = 30    # Facial Action Units (UL-DD Extracted_Features CSVs)
N_TELE    = 5     # pitch, roll, speed, rpm, gear (UL-DD CSV_Files)
N_CLASSES = 3     # Alert / Low Vigilant / Drowsy
N_RT_FEAT = 10    # Real-time features for M4 (YOLO + MediaPipe pipeline)
CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']


# ─────────────────────────────────────────────────────────────────────────────
# M1 — Facial BiLSTM  (camera-only baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_m1_facial_bilstm(
    window: int      = WINDOW,
    n_fau: int       = N_FAU,
    n_classes: int   = N_CLASSES,
    lstm_units: tuple = (128, 64),
    dropout: float   = 0.45,
    l2_reg: float    = 5e-4,
) -> Model:
    """
    M1: Bidirectional LSTM on 30 FAU time-series features.

    Architecture:
        Input (batch, 240, 30)
        → BiLSTM(128, return_sequences=True) → Dropout
        → BiLSTM(64)                          → Dropout
        → Dense(64, relu) → BatchNorm
        → Dense(3, softmax)

    Parameters: ~420 K
    """
    reg = l2(l2_reg)
    inp = Input(shape=(window, n_fau), name='fau_input')

    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True,
                    kernel_regularizer=reg, recurrent_regularizer=reg),
        name='bilstm_1'
    )(inp)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units[1], return_sequences=False,
                    kernel_regularizer=reg, recurrent_regularizer=reg),
        name='bilstm_2'
    )(x)
    x = layers.Dropout(dropout)(x)

    x   = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x   = layers.BatchNormalization()(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    return Model(inp, out, name='M1_FacialBiLSTM')


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Telemetry LSTM  (CAN-signal-only baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_m2_telemetry_lstm(
    window: int      = WINDOW,
    n_tele: int      = N_TELE,
    n_classes: int   = N_CLASSES,
    lstm_units: tuple = (64, 32),
    dropout: float   = 0.45,
    l2_reg: float    = 5e-4,
) -> Model:
    """
    M2: Unidirectional LSTM on 5 CAN/telemetry features.

    Unidirectional (not BiLSTM) — telemetry is causal: past steering
    causes future vehicle state, so forward-only context is appropriate
    and more realistic for online deployment.

    Architecture:
        Input (batch, 240, 5)
        → LSTM(64, return_sequences=True) → Dropout
        → LSTM(32)                         → Dropout
        → Dense(32, relu) → BatchNorm
        → Dense(3, softmax)

    Parameters: ~45 K
    """
    reg = l2(l2_reg)
    inp = Input(shape=(window, n_tele), name='tele_input')

    x = layers.LSTM(
        lstm_units[0], return_sequences=True,
        kernel_regularizer=reg, recurrent_regularizer=reg, name='lstm_1'
    )(inp)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(
        lstm_units[1], return_sequences=False,
        kernel_regularizer=reg, recurrent_regularizer=reg, name='lstm_2'
    )(x)
    x = layers.Dropout(dropout)(x)

    x   = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x   = layers.BatchNormalization()(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    return Model(inp, out, name='M2_TelemetryLSTM')


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal attention block  (shared utility for M3)
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionBlock(layers.Layer):
    """
    Scaled dot-product attention where query comes from one modality
    and key/value come from another.

    Both tensors are projected to d_model before attention so that
    input dimensions need not match.

    Outputs:
        context : (batch, T_query, d_model)  — attended representation
        weights : (batch, T_query, T_kv)     — attention map (for viz)
    """

    def __init__(self, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.Wq   = layers.Dense(d_model, use_bias=False)
        self.Wk   = layers.Dense(d_model, use_bias=False)
        self.Wv   = layers.Dense(d_model, use_bias=False)
        self.proj = layers.Dense(d_model)

    def call(self, query, key_value, training=None):
        Q = self.Wq(query)        # (B, T_q,  d)
        K = self.Wk(key_value)   # (B, T_kv, d)
        V = self.Wv(key_value)   # (B, T_kv, d)

        scale   = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        scores  = tf.matmul(Q, K, transpose_b=True) / scale   # (B, T_q, T_kv)
        weights = tf.nn.softmax(scores, axis=-1)               # (B, T_q, T_kv)
        context = tf.matmul(weights, V)                        # (B, T_q, d)
        return self.proj(context), weights

    def get_config(self):
        cfg = super().get_config()
        cfg['d_model'] = self.d_model
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# M3 — Cross-Modal Attention Fusion  (camera + CAN)  ★ Thesis Contribution
# ─────────────────────────────────────────────────────────────────────────────

class M3FusionModel(Model):
    """
    M3: Bidirectional cross-modal attention fusion of FAU + Telemetry.

    Architecture:
        FAU  branch → BiLSTM(128, seq)  → H_fau  (B, 240, 256)
        Tele branch → LSTM(64,   seq)   → H_tele (B, 240,  64)

        FAU→Tele : ctx_fau  = CrossAttn(query=H_fau,  kv=H_tele)  (B, 240, 64)
        Tele→FAU : ctx_tele = CrossAttn(query=H_tele, kv=H_fau)   (B, 240, 64)

        GAP each:  p_fau_ctx (64) | p_tele_ctx (64) | p_fau (256) | p_tele (64)
        Concat → Dense(128) → BN → Dropout → Dense(3, softmax)

    Parameters: ~750 K
    Novelty over UL-DD paper: temporal modelling + learned cross-modal
    attention instead of SVM/RF early fusion.
    """

    def __init__(
        self,
        window: int    = WINDOW,
        n_fau: int     = N_FAU,
        n_tele: int    = N_TELE,
        n_classes: int = N_CLASSES,
        d_attn: int    = 64,
        fau_units: int = 128,
        tele_units: int = 64,
        dropout: float = 0.45,
        l2_reg: float  = 5e-4,
        **kwargs,
    ):
        kwargs.setdefault('name', 'M3_FusionAttention')
        super().__init__(**kwargs)
        reg = l2(l2_reg)

        # FAU branch
        self.fau_bilstm = layers.Bidirectional(
            layers.LSTM(fau_units, return_sequences=True,
                        kernel_regularizer=reg, recurrent_regularizer=reg),
            name='fau_bilstm'
        )
        self.fau_drop = layers.Dropout(dropout)

        # Telemetry branch
        self.tele_lstm = layers.LSTM(
            tele_units, return_sequences=True,
            kernel_regularizer=reg, recurrent_regularizer=reg,
            name='tele_lstm'
        )
        self.tele_drop = layers.Dropout(dropout)

        # Cross-modal attention (bidirectional)
        self.attn_f2t = CrossModalAttentionBlock(d_attn, name='attn_fau_to_tele')
        self.attn_t2f = CrossModalAttentionBlock(d_attn, name='attn_tele_to_fau')

        # Pooling + head
        self.gap      = layers.GlobalAveragePooling1D()
        self.concat   = layers.Concatenate()
        self.dense1   = layers.Dense(128, activation='relu', kernel_regularizer=reg)
        self.bn       = layers.BatchNormalization()
        self.drop_cls = layers.Dropout(dropout)
        self.out      = layers.Dense(n_classes, activation='softmax', name='output')

        # Save config for serialisation
        self._cfg = dict(
            window=window, n_fau=n_fau, n_tele=n_tele,
            n_classes=n_classes, d_attn=d_attn,
            fau_units=fau_units, tele_units=tele_units,
            dropout=dropout, l2_reg=l2_reg,
        )

    def call(self, inputs, training=None):
        fau_in, tele_in = inputs

        H_fau  = self.fau_drop(
            self.fau_bilstm(fau_in, training=training), training=training
        )                                                    # (B, 240, 256)
        H_tele = self.tele_drop(
            self.tele_lstm(tele_in, training=training), training=training
        )                                                    # (B, 240,  64)

        ctx_fau,  self._attn_f2t = self.attn_f2t(H_fau,  H_tele, training=training)
        ctx_tele, self._attn_t2f = self.attn_t2f(H_tele, H_fau,  training=training)

        pooled = self.concat([
            self.gap(ctx_fau),    # (B,  64)
            self.gap(ctx_tele),   # (B,  64)
            self.gap(H_fau),      # (B, 256)
            self.gap(H_tele),     # (B,  64)
        ])                        # (B, 448)

        x = self.dense1(pooled, training=training)
        x = self.bn(x,          training=training)
        x = self.drop_cls(x,    training=training)
        return self.out(x)

    def get_attention_weights(self, fau_in, tele_in):
        """Return (attn_fau→tele, attn_tele→fau) arrays for interpretability."""
        _ = self.call([fau_in, tele_in], training=False)
        return self._attn_f2t, self._attn_t2f

    def get_config(self):
        cfg = super().get_config()
        cfg.update(self._cfg)
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# M4 — YOLO Real-Time BiLSTM  (live demo, no pre-extracted CSV needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_m4_realtime_bilstm(
    window: int    = WINDOW,
    n_rt_feat: int = N_RT_FEAT,
    n_classes: int = N_CLASSES,
    dropout: float = 0.30,
) -> Model:
    """
    M4: Lightweight BiLSTM on 10 real-time features extracted live by
    YOLOv8 (face detection) + MediaPipe FaceMesh (landmarks → features).

    10 features per 4-Hz frame:
        EAR_left, EAR_right, MAR, PERCLOS, blink_rate,
        head_pitch, head_roll, head_yaw, brow_raise, jaw_drop

    Architecture:
        Input (batch, 240, 10)
        → BiLSTM(64, seq) → Dropout → BiLSTM(32) → Dropout
        → Dense(32, relu) → Dense(3, softmax)

    Parameters: ~82 K  (edge-deployable)
    """
    inp = Input(shape=(window, n_rt_feat), name='rt_feature_input')

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name='bilstm_1'
    )(inp)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False), name='bilstm_2'
    )(x)
    x = layers.Dropout(dropout)(x)

    x   = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    return Model(inp, out, name='M4_RealTimeBiLSTM')


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, **kwargs) -> Model:
    """
    Build a model by short name.

    Args:
        name : 'm1' | 'm2' | 'm3' | 'm4'
        **kwargs : forwarded to the specific builder / class

    Returns:
        tf.keras.Model
    """
    name = name.lower()
    if name == 'm1':
        return build_m1_facial_bilstm(**kwargs)
    if name == 'm2':
        return build_m2_telemetry_lstm(**kwargs)
    if name == 'm3':
        return M3FusionModel(**kwargs)
    if name == 'm4':
        return build_m4_realtime_bilstm(**kwargs)
    raise ValueError(
        f"Unknown model: {name!r}.  Choose from: m1 | m2 | m3 | m4"
    )
