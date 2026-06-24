# M6 Model Architecture Comparison

## M6_lite (Vision-Only)
```
Input: Video embeddings @ 60fps (256-dim)
  ↓
LSTM layer
  ↓
Dense layers
  ↓
Output: Drowsiness prediction
```

**Does NOT use:**
- ✗ CAN signals (speed, throttle, steering angle)
- ✗ Facial Action Units (eye/mouth movements)

**Result @ 1fps:** 41%
**Result @ 60fps:** 42.29% (no improvement)

---

## M6_full (Multimodal Fusion) ⭐ CORRECT APPROACH
```
┌─────────────────────────────────────────────┐
│  Video embeddings @ 60fps (256-dim)         │
│  ↓                                          │
│  LSTM (temporal modeling)                   │
│  ↓                                          │
├─────────────────────────────────────────────┤
│  Facial Action Units (normalized)           │
│  ↓                                          │
│  Dense layers                               │
│  ↓                                          │
├─────────────────────────────────────────────┤
│  CAN Telemetry signals (speed/throttle)     │ ← ALIGNED to 60fps video!
│  ↓                                          │
│  Temporal alignment layer                   │
│  ↓                                          │
├─────────────────────────────────────────────┤
│  Concatenate all modalities                 │
│  ↓                                          │
│  Multi-head attention / Fusion layers       │
│  ↓                                          │
│  Dense classifier                           │
│  ↓                                          │
│  Output: Drowsiness prediction              │
└─────────────────────────────────────────────┘
```

**Uses ALL modalities:**
- ✓ Video embeddings (vision)
- ✓ Facial Action Units (facial)
- ✓ CAN signals (behavior/vehicle state)

**Why the 60fps fix matters for M6_full:**

Before (1fps):
- Video frames sampled @ 1fps = 60 frames per 60-sec session
- CAN signals @ 100Hz = 6000 samples per 60-sec session
- Alignment: POOR (60x mismatch) ❌

After (60fps):
- Video frames extracted @ 60fps = 3600 frames per 60-sec session
- CAN signals @ 100Hz = 6000 samples per 60-sec session
- Alignment: GOOD (1.67x difference, interpolatable) ✓

**Expected improvement for M6_full:**
- Previous: 41% (@ 1fps misaligned)
- Now: Should be 50-65% (@ 60fps aligned)

---

## Key Insight

M6_lite improvement (41% → 42%) is small because:
- It only uses video, not temporal context from CAN
- Temporal alignment fix helps MOSTLY with multimodal fusion

M6_full should show BIG improvement because:
- It combines video + facial + CAN signals
- The 60fps alignment lets the fusion layer learn real correlations
- Example: "When speed drops + video blurs + AU action → drowsy"

This is the THESIS contribution!
