"""
M6-A — Vision-Only Temporal Model for Driver Drowsiness Detection
==================================================================

Implements M6-A from M6_Design.md — a temporal vision-only baseline
that uses YOLOv8 embeddings with BiLSTM + Attention Pooling.

This model serves as:
  1. Proof that temporal visual features help drowsiness detection
  2. Fair comparison baseline for M6-C (multimodal) fusion
  3. Ablation study: Vision vs. Vision+Telemetry

Architecture:
    YOLOv8 Embeddings (16, 512)
        ↓
    BiLSTM(128 bidir) → (16, 256)
        ↓
    Attention Pooling → (256,)
        ↓
    Dense(128) + ReLU + Dropout(0.3)
        ↓
    Dense(3) → Softmax
    
Expected Performance (with proper data alignment):
    - M6-A (Vision-only): ~50-60% accuracy
    - M6-C (Vision+Telemetry): ~70-78% accuracy
    - Gain from multimodal fusion: +10-20%

Usage:
    from src.models.m6_vision_only import M6VisionOnly, build_m6a
    
    model = build_m6a(t_vis=16, dropout=0.3)
    logits = model(embeddings)  # (B, 3)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

# Defaults (must match M6_Design.md and m6_fusion.py)
T_VIS_DEFAULT   = 16    # frames per window
EMB_DIM_DEFAULT = 512   # YOLOv8 embedding dimension
VIS_HIDDEN_DEFAULT = 128  # BiLSTM hidden size
N_CLASSES       = 3
CLASS_NAMES     = ["Alert", "LowVigilant", "Drowsy"]


# ─────────────────────────────────────────────────────────────────────────────
# Attention Pooling (Multi-head Attention)
# ─────────────────────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    Multi-head attention-based pooling.
    
    Instead of simple mean/max pooling, learn which frames are most
    important for the decision via scaled dot-product attention.
    
    Input:  (B, T, D) sequence
    Output: (B, D) weighted sum
    """
    
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Learnable query vector (context vector)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=n_heads,
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequence
        Returns:
            context: (B, D) attended context
        """
        B = x.size(0)
        
        # Expand query to batch size
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        
        # Self-attention of sequence with learnable query
        context, _ = self.attention(query, x, x, need_weights=False)
        
        # Residual + LayerNorm
        return self.norm(context.squeeze(1))  # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# M6VisionOnly Model
# ─────────────────────────────────────────────────────────────────────────────
class M6VisionOnly(nn.Module):
    """
    Vision-only temporal baseline (M6-A from M6_Design.md).
    
    Process infrared frame embeddings through BiLSTM and attention pooling,
    then classify drowsiness state. No telemetry involved.
    
    This model demonstrates:
      1. That temporal modeling of visual embeddings is effective
      2. Provides a fair baseline for multimodal comparison
      3. Validates the YOLOv8 embedding quality
    
    Architecture:
        BiLSTM(emb_dim → vis_hidden=128 bidir) → (B, T, 256)
            ↓
        Attention Pooling → (B, 256)
            ↓
        Dense(256 → 128) + ReLU + Dropout
            ↓
        Dense(128 → 3)
    """
    
    def __init__(self,
                 emb_dim: int = EMB_DIM_DEFAULT,
                 t_vis: int = T_VIS_DEFAULT,
                 vis_hidden: int = VIS_HIDDEN_DEFAULT,
                 n_heads: int = 4,
                 dropout: float = 0.3,
                 n_classes: int = N_CLASSES):
        """
        Args:
            emb_dim: Dimension of input embeddings (512 for YOLOv8)
            t_vis: Number of frames per window (16)
            vis_hidden: Hidden size of BiLSTM (128)
            n_heads: Number of attention heads (4)
            dropout: Dropout rate (0.3)
            n_classes: Number of output classes (3)
        """
        super().__init__()
        
        # Temporal encoder (BiLSTM)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=vis_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )
        
        # Attention pooling
        d_bilstm = 2 * vis_hidden  # 256 (bidirectional)
        self.attention_pool = AttentionPooling(
            hidden_dim=d_bilstm,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_bilstm, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vis: (B, T_vis, emb_dim) frame embeddings
        Returns:
            logits: (B, n_classes) prediction logits
        """
        # BiLSTM temporal encoding
        lstm_out, _ = self.bilstm(vis)  # (B, T, 2*vis_hidden)
        
        # Attention-based pooling
        pooled = self.attention_pool(lstm_out)  # (B, 2*vis_hidden)
        
        # Classification
        logits = self.head(pooled)  # (B, n_classes)
        
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────
def build_m6a(t_vis: int = T_VIS_DEFAULT,
              emb_dim: int = EMB_DIM_DEFAULT,
              vis_hidden: int = VIS_HIDDEN_DEFAULT,
              n_heads: int = 4,
              dropout: float = 0.3,
              n_classes: int = N_CLASSES) -> M6VisionOnly:
    """
    Factory function to build M6-A (Vision-Only) model.
    
    Args:
        t_vis: Frames per window (16)
        emb_dim: Embedding dimension (512)
        vis_hidden: BiLSTM hidden size (128)
        n_heads: Attention heads (4)
        dropout: Dropout rate (0.3)
        n_classes: Output classes (3)
    
    Returns:
        M6VisionOnly: Instantiated model
    """
    return M6VisionOnly(
        emb_dim=emb_dim,
        t_vis=t_vis,
        vis_hidden=vis_hidden,
        n_heads=n_heads,
        dropout=dropout,
        n_classes=n_classes
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Test / Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    
    # Create model
    model = build_m6a()
    print(f"M6-A Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    t_vis = 16
    emb_dim = 512
    
    x = torch.randn(batch_size, t_vis, emb_dim)
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: ({batch_size}, 3)")
    assert logits.shape == (batch_size, 3), "Output shape mismatch!"
    
    print("\n✅ M6-A Vision-Only model test passed!")
