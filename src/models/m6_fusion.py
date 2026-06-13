"""
M6 — Temporal YOLOv8 Visual + CAN Telemetry Fusion (PyTorch)
============================================================

Design Reference: M6_Design.md
Status: M6-C Fusion model implemented; M6-A baseline in m6_vision_only.py

Improves over M5 (per-frame YOLOv8-cls, ~50-54% on frames) by:

  1. **Temporal Modeling**: Encodes T=16 frames per 60-second window
     through BiLSTM or Transformer to capture temporal dependencies
     in driver behavior.

  2. **Multimodal Fusion**: Combines visual embeddings with 5-channel
     CAN telemetry (pitch, roll, speed, RPM, gear) @ 4 Hz for 240 timesteps.

  3. **Cross-Modal Attention** (M6_Full only): Bidirectional attention
     allows visual and telemetry branches to inform each other.

Window Alignment (Critical for fusion correctness):
    For each telemetry window covering [start_sec, end_sec]:
      • Window index: win_idx
      • Frame range (60fps): [win_idx * 900, win_idx * 900 + 3600]
      • Extract frames falling in this range
      • Uniformly sample T_VIS=16 embeddings
      • Align with CAN window covering same time period
    → Only synchronized data should be fused

Two Variants:

  **M6_Lite** (M6 Baseline from Design)
     BiLSTM(visual:128 bidir) → (B, 256)
     ⊕
     BiLSTM(CAN:64 bidir) → (B, 128)
     ↓
     concat + Dense(128) + ReLU + Dropout → Dense(3)
     
     Params: ~380K + frozen backbone
     Training: Fast, easy baseline

  **M6_Full** (M6-C Fusion from Design — Thesis Novelty)
     Visual (B, T, 512):
       → Linear(512→128) + PosPE
       → Transformer(2 layers, 128-dim, 4 heads)
       → H_v: (B, T, 128)
     
     CAN (B, 240, 5):
       → BiLSTM(64 bidir)
       → Linear(128→128)
       → H_c: (B, 240, 128)
     
     Bidirectional Cross-Attention:
       ctx_v = CrossAttn(query=H_v, kv=H_c)  # Visual attends to CAN
       ctx_c = CrossAttn(query=H_c, kv=H_v)  # CAN attends to visual
     
     Fusion Head:
       concat[H_v, H_c, ctx_v, ctx_c] → (B, 512)
       → Dense(128) + ReLU + Dropout → Dense(3)
     
     Params: ~1M + frozen backbone
     Training: ~5-10 min per fold (GPU)

Visual Embeddings:
    Source: M5 YOLOv8-cls backbone (512-dim, frozen)
    Extraction: See src/models/m6_extractor.py
    Cache: models/embeddings/{subject}_{session}.npz
    
    ⚠️  IMPORTANT: Embeddings must be extracted from ACTUAL UL-DD driving
    videos, not from yolo_frames/ classification dataset. Current implementation
    uses yolo_frames which breaks multimodal alignment → accuracy capped at ~41%.
    
    Fix: Modify m6_extractor.py to extract from UL-DD videos instead.

Label Schema (matches M1/M2/M3/M5):
    0 → Alert        (KSS 1-3)
    1 → Low Vigilant (KSS 4-6)
    2 → Drowsy       (KSS 7-9)

Expected Accuracy (with proper data alignment):
    M6-A (Vision-only, no CAN):     ~50-60%
    M6-Lite (Simple multimodal):    ~65-75%
    M6-Full (Full fusion):          ~70-78%
    
    Current (wrong data source):    ~41% (alignment broken)
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ─── Defaults aligned with the rest of the project ────────────────────────────
T_VIS_DEFAULT   = 16    # frames sampled per window
EMB_DIM_DEFAULT = 512   # YOLOv8-cls backbone embedding size
T_CAN_DEFAULT   = 240   # 60 s × 4 Hz telemetry timesteps
N_CAN_DEFAULT   = 5     # pitch, roll, speed, rpm, gear
N_CLASSES       = 3
CLASS_NAMES     = ["Alert", "LowVigilant", "Drowsy"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared CAN telemetry encoder (BiLSTM)
# ─────────────────────────────────────────────────────────────────────────────
class _CANEncoder(nn.Module):
    """BiLSTM encoder for CAN telemetry → returns (B, T_can, 2*hidden)."""

    def __init__(self, n_in: int = N_CAN_DEFAULT, hidden: int = 64,
                 num_layers: int = 1, dropout: float = 0.30):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=n_in, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # (B, T_can, n_in)
        out, _ = self.bilstm(x)
        return self.drop(out)                               # (B, T_can, 2H)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal attention block
# ─────────────────────────────────────────────────────────────────────────────
class _CrossAttnBlock(nn.Module):
    """
    Standard scaled-dot-product multi-head cross-attention.

    Query comes from one modality, key/value from the other.
    Inputs are first projected to `d_model` so source dims may differ.
    """

    def __init__(self, d_q: int, d_kv: int, d_model: int = 128,
                 n_heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.proj_q  = nn.Linear(d_q,  d_model)
        self.proj_kv = nn.Linear(d_kv, d_model)
        self.attn    = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, q_seq: torch.Tensor,
                kv_seq: torch.Tensor) -> torch.Tensor:
        Q  = self.proj_q(q_seq)        # (B, T_q,  d_model)
        KV = self.proj_kv(kv_seq)      # (B, T_kv, d_model)
        ctx, _ = self.attn(Q, KV, KV, need_weights=False)
        return self.norm(ctx + Q)      # residual + LN  → (B, T_q, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# M6_Lite  — simple BiLSTM ⊕ BiLSTM concat
# ─────────────────────────────────────────────────────────────────────────────
class M6_Lite(nn.Module):
    """
    Lightweight fusion baseline.

        Visual (B, T_vis, 512)  → BiLSTM(128 bidir) → mean-pool → (B, 256)
        CAN    (B, T_can,   5)  → BiLSTM( 64 bidir) → mean-pool → (B, 128)
        concat → Dense(128) → Dropout → Dense(3)

    Intended as the first sanity-check model for the M6 idea.
    """

    def __init__(self, emb_dim: int = EMB_DIM_DEFAULT,
                 n_can: int = N_CAN_DEFAULT,
                 vis_hidden: int = 128, can_hidden: int = 64,
                 dropout: float = 0.40, n_classes: int = N_CLASSES):
        super().__init__()
        self.vis_bilstm = nn.LSTM(
            input_size=emb_dim, hidden_size=vis_hidden,
            batch_first=True, bidirectional=True,
        )
        self.can_enc = _CANEncoder(n_in=n_can, hidden=can_hidden,
                                   dropout=dropout)
        self.drop    = nn.Dropout(dropout)
        d_v          = 2 * vis_hidden       # 256
        d_c          = 2 * can_hidden       # 128
        self.head    = nn.Sequential(
            nn.Linear(d_v + d_c, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, vis: torch.Tensor,
                can: torch.Tensor) -> torch.Tensor:
        # vis: (B, T_vis, 512)   can: (B, T_can, 5)
        v_seq, _ = self.vis_bilstm(vis)
        v_pool   = self.drop(v_seq.mean(dim=1))         # (B, 256)
        c_seq    = self.can_enc(can)
        c_pool   = c_seq.mean(dim=1)                    # (B, 128)
        return self.head(torch.cat([v_pool, c_pool], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# M6_Full  — Transformer(visual) + BiLSTM(can) + bidirectional cross-attention
# ─────────────────────────────────────────────────────────────────────────────
class M6_Full(nn.Module):
    """
    Full M6 model — the thesis-novelty configuration.

        Visual (B, T_vis, 512)
            → Linear(512 → d_model)
            → +PositionalEmbedding
            → TransformerEncoder × n_layers           ─► H_v  (B, T_vis, d)

        CAN    (B, T_can, 5)
            → BiLSTM(can_hidden bidir) → Linear(2H → d) ─► H_c  (B, T_can, d)

        Cross-modal attention (bidirectional):
            ctx_v = CrossAttn(query=H_v, kv=H_c)       (B, T_vis, d)
            ctx_c = CrossAttn(query=H_c, kv=H_v)       (B, T_can, d)

        Mean-pool both → concat (4 × d) → Dense(128) → Dropout → Dense(3)
    """

    def __init__(self,
                 emb_dim: int = EMB_DIM_DEFAULT,
                 n_can: int   = N_CAN_DEFAULT,
                 t_vis: int   = T_VIS_DEFAULT,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 can_hidden: int = 64,
                 dropout: float = 0.30,
                 n_classes: int = N_CLASSES):
        super().__init__()

        # Visual branch -------------------------------------------------------
        self.vis_proj   = nn.Linear(emb_dim, d_model)
        self.vis_pos    = nn.Parameter(torch.zeros(1, t_vis, d_model))
        nn.init.trunc_normal_(self.vis_pos, std=0.02)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.vis_tx     = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # CAN branch ----------------------------------------------------------
        self.can_enc    = _CANEncoder(n_in=n_can, hidden=can_hidden,
                                      dropout=dropout)
        self.can_proj   = nn.Linear(2 * can_hidden, d_model)

        # Bidirectional cross-modal attention ---------------------------------
        self.attn_v2c   = _CrossAttnBlock(d_q=d_model, d_kv=d_model,
                                          d_model=d_model, n_heads=n_heads,
                                          dropout=dropout)
        self.attn_c2v   = _CrossAttnBlock(d_q=d_model, d_kv=d_model,
                                          d_model=d_model, n_heads=n_heads,
                                          dropout=dropout)

        # Classifier head -----------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, vis: torch.Tensor,
                can: torch.Tensor) -> torch.Tensor:
        # Visual branch
        H_v = self.vis_proj(vis) + self.vis_pos[:, : vis.size(1), :]
        H_v = self.vis_tx(H_v)                                    # (B, T_vis, d)

        # CAN branch
        H_c = self.can_proj(self.can_enc(can))                    # (B, T_can, d)

        # Cross-modal attention (bidirectional)
        ctx_v = self.attn_v2c(H_v, H_c)                           # (B, T_vis, d)
        ctx_c = self.attn_c2v(H_c, H_v)                           # (B, T_can, d)

        # Pool everything → concat
        pooled = torch.cat([
            H_v.mean(dim=1),    # (B, d)  raw visual context
            H_c.mean(dim=1),    # (B, d)  raw CAN context
            ctx_v.mean(dim=1),  # (B, d)  visual attended to CAN
            ctx_c.mean(dim=1),  # (B, d)  CAN attended to visual
        ], dim=1)                                                  # (B, 4d)

        return self.head(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_m6(name: str = "full", **kwargs) -> nn.Module:
    """Build an M6 variant by short name: 'lite' | 'full'."""
    name = name.lower()
    if name in ("lite", "m6_lite", "m6-lite"):
        return M6_Lite(**kwargs)
    if name in ("full", "m6", "m6_full", "m6-full"):
        return M6_Full(**kwargs)
    raise ValueError(f"Unknown M6 variant: {name!r}. Use 'lite' or 'full'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters of a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
