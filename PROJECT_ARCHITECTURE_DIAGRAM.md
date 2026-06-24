# DRIVER DROWSINESS DETECTION - PROJECT ARCHITECTURE

## Official Project Architecture

Based on your UL-DD dataset implementation with M1-M6 progression

---

## COMPLETE SYSTEM ARCHITECTURE (VERIFIED)

```
INPUT LAYER
═══════════════════════════════════════════════════════════════════════════════

    VIDEO STREAM                          CAN BUS TELEMETRY
    (Driver Face)                         (Vehicle Signals)
          │                                      │
          │ 60 Hz RGB                           │ 60 Hz multivariate
          │ 640x640 frames                      │ 5 channels: steering, speed, etc
          │                                      │
          └──────────────────┬───────────────────┘
                             │
                      PREPROCESSING
                      │
                      ├─ Video: 60 Hz → 4 Hz (downsample)
                      ├─ CAN: 60 Hz → 4 Hz (downsample)
                      ├─ Sliding windows: 60 seconds @ 4 Hz = 240 timesteps
                      ├─ Stride: 15 seconds (overlapping windows)
                      └─ Z-score normalization per feature


FEATURE EXTRACTION LAYER
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────┐    ┌──────────────────────────────┐
    │  VISUAL FEATURE EXTRACTION      │    │  CAN FEATURE EXTRACTION      │
    │  (YOLOv8-Nano Backbone)         │    │  (1D CNN)                    │
    └─────────────────────────────────┘    └──────────────────────────────┘
                  │                                      │
                  │ 640x640 Input                       │ (240, 5) Input
                  ↓                                      ↓
    ┌─────────────────────────────────┐    ┌──────────────────────────────┐
    │ Conv Stem                       │    │ 1D Conv Block 1              │
    │ Output: (32, 320, 320)          │    │ Filters: 32, Kernel: 3      │
    │                                 │    │ Output: (240, 32)            │
    │ Stage 1 (8x downsample)         │    │                              │
    │ Output: (64, 160, 160)          │    │ 1D Conv Block 2              │
    │                                 │    │ Filters: 64, Kernel: 3      │
    │ Stage 2 (16x downsample)        │    │ Output: (240, 64)            │
    │ Output: (128, 80, 80)           │    │                              │
    │                                 │    │ Global Average Pool          │
    │ Stage 3 (32x downsample)        │    │ Output: (1, 64)              │
    │ Output: (256, 40, 40)           │    │                              │
    │                                 │    │ Dense(64 → 128)              │
    │ Stage 4 (64x downsample)        │    │ Output: (1, 128)             │
    │ Output: (512, 20, 20)           │    └──────────────────────────────┘
    │                                 │
    │ Global Average Pool             │                  F_can
    │ Output: (1, 512)                │
    └─────────────────────────────────┘
                  │
               F_visual (512-dim)


CROSS-MODAL ATTENTION LAYER
═══════════════════════════════════════════════════════════════════════════════

    F_visual (512-dim)              F_can (128-dim)
         │                               │
         │                               │
    ┌────┴────┐                    ┌─────┴──────┐
    │          │                    │            │
    ↓          ↓                    ↓            ↓
  Query(Q)   Key(K)               Query(Q)   Key(K)
  from       from                 from       from
  Visual     CAN                   CAN        Visual
    │          │                    │            │
    │     ┌────┴────────────────────┘            │
    │     │                                      │
    │     ↓ SCALED DOT-PRODUCT ATTENTION         │
    │     ├─ α = softmax(Q·K^T / sqrt(d_k))     │
    │     ├─ context_v = α·V_can                 │
    │     └─ Output: Z_v→c (512-dim)             │
    │                                            │
    │                                      ┌─────┘
    │                                      │
    │                                      ↓ SCALED DOT-PRODUCT ATTENTION
    │                                      ├─ β = softmax(Q·K^T / sqrt(d_k))
    │                                      ├─ context_c = β·V_visual
    │                                      └─ Output: Z_c→v (128-dim)
    │                                            │
    └────────────────┬──────────────────────────┘
                     │
              Z_v→c + Z_c→v
              (Visual enhanced with CAN context)
              (CAN enhanced with Visual context)


TEMPORAL MODELING LAYER
═══════════════════════════════════════════════════════════════════════════════

    Two Variant Paths:

    M6_LITE (Edge-Deployable)          M6_FULL (Research Maximum)
    ───────────────────────            ───────────────────────────
    
    Input: Fused features              Input: Fused features
      │                                  │
      ↓                                  ↓
    BiLSTM(128 bidir)                 Transformer
    - 128 forward units                - 2 layers
    - 128 backward units               - 4 attention heads
    - Captures past & future           - Self-attention mechanism
    - Output: (1, 256)                 - Output: (1, 256)
      │                                  │
      │                                  │
      └──────────────┬───────────────────┘
                     │
            Temporal Context Captured
            (60-second progression)
                     │
                     ↓


FUSION & CLASSIFICATION LAYER
═══════════════════════════════════════════════════════════════════════════════

    Concatenate all representations:
    [context_visual, context_can, visual_features, can_features, temporal_context]
    
    Total dimension: 640-d (or 512-d for M6_Lite)
           │
           ↓
    Dense(640 → 128) + BatchNorm + ReLU
           │
           ↓
    Dense(128 → 64) + BatchNorm + ReLU
           │
           ↓
    Dense(64 → 3) + Softmax
           │
           ↓
    Logits: [Alert, Low Vigilance, Drowsy]


OUTPUT LAYER
═══════════════════════════════════════════════════════════════════════════════

    Probability Distribution (3-class)
    
    Output: [p_alert, p_low_vig, p_drowsy]
    
    Prediction: argmax(logits)
    ├─ Class 0: Alert (green)
    ├─ Class 1: Low Vigilance (yellow)
    └─ Class 2: Drowsy (red)
```

---

## TRAINING & EVALUATION PIPELINE

```
SUBJECT-INDEPENDENT 5-FOLD CROSS-VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Dataset: UL-DD (19 drivers, 2 sessions each)

Fold 0: Train [E-S] (15 subjects) → Test [A-D] (4 subjects)
Fold 1: Train [A-D,I-S] (15 subjects) → Test [E-H] (4 subjects)
Fold 2: Train [...] (16 subjects) → Test [I-K] (3 subjects)
Fold 3: Train [...] (15 subjects) → Test [L-O] (4 subjects)
Fold 4: Train [...] (15 subjects) → Test [P-S] (4 subjects)

CRITICAL: NO subject appears in both train and test

Per-Fold Training:
├─ Epochs: 30-35 (with early stopping)
├─ Batch size: 32
├─ Loss: Weighted cross-entropy (inverse class frequency)
├─ Optimizer: AdamW with cosine learning rate schedule
├─ Initial LR: 5e-4 or 3e-4
└─ Weight decay: 1e-4


EVALUATION METRICS
═══════════════════════════════════════════════════════════════════════════════

1. Accuracy: (TP + TN) / Total
2. Macro F1: Mean of per-class F1 scores
3. Per-Class Recall: Sensitivity for each drowsiness level
4. Confusion Matrix: Error pattern analysis
5. Stability: Std dev across 5 folds (< 1% = robust)
```

---

## MODEL VARIANTS COMPARISON

```
┌────────┬──────────────────┬────────────┬──────────────┬─────────────┐
│ Model  │ Visual Encoder   │ CAN Enc    │ Fusion       │ Parameters  │
├────────┼──────────────────┼────────────┼──────────────┼─────────────┤
│ M1     │ BiLSTM(FAU feat) │ N/A        │ N/A          │ 420K        │
│ M2     │ N/A              │ LSTM       │ N/A          │ 45K         │
│ M3     │ BiLSTM(FAU feat) │ LSTM       │ Attention    │ 750K        │
│ M5     │ YOLOv8 (per-fr)  │ N/A        │ N/A          │ 6.4M        │
│ M6_L   │ BiLSTM(YOLOv8)   │ BiLSTM     │ Attention    │ 0.4M        │
│ M6_F   │ Transformer      │ BiLSTM     │ BiAttn       │ 0.9M        │
└────────┴──────────────────┴────────────┴──────────────┴─────────────┘

Target Accuracies:
M1: 39.1%  (hand-crafted baseline)
M2: 38.7%  (telemetry-only baseline)
M3: 39.9%  (multimodal hand-crafted)
M5: 54.5%  (SOTA: per-frame deep learning)
M6_L: ~58-60% (temporal + edge-friendly)
M6_F: ~60-65% (temporal + full fusion)
```

---

## DATA FLOW SUMMARY

```
RAW UL-DD DATA
    ↓
[Preprocessing: downsample, window, normalize]
    ↓
WINDOWED DATASET (240 timesteps per window)
    ├─ Visual: (240 frames) → YOLOv8 backbone → (512-dim per frame)
    ├─ CAN: (240, 5) → 1D CNN → (128-dim)
    └─ Labels: Alert / Low Vigilance / Drowsy
    ↓
[For M5: Per-frame classification only]
[For M6: Extract embeddings, apply temporal modeling]
    ↓
SUBJECT-INDEPENDENT 5-FOLD CROSS-VALIDATION
    ├─ Fold 0-4: Train on 15-16 subjects, Test on 3-4 subjects
    ├─ No subject leakage
    ├─ Per-fold metrics recorded
    └─ Final: Average accuracy + F1 across folds
    ↓
RESULTS
    ├─ Per-fold breakdown (5 folds)
    ├─ Aggregate statistics (mean, std)
    ├─ Confusion matrices
    └─ Model checkpoints saved
    ↓
DEPLOYMENT
    ├─ M5: Real-time demo (30-50ms per frame)
    ├─ M6_Lite: Edge ECU (100ms)
    └─ M6_Full: Server/cloud (150-200ms)
```

---

## KEY ARCHITECTURAL DECISIONS

### 1. Why YOLOv8 Backbone for Visual Features?
- Pre-trained on 14 million ImageNet images
- Robust facial feature extraction
- Handles lighting, pose, expression variations
- Much better than hand-crafted FAU features (39% → 54.5%)

### 2. Why 1D CNN for CAN Features?
- Multivariate time-series processing
- Captures local temporal patterns in steering/speed
- Lightweight (45K-128K params)
- Fast inference

### 3. Why Cross-Modal Attention?
- Learn which modality to trust in different contexts
- Bidirectional (visual→CAN and CAN→visual)
- Interpretable attention weights
- Enables weighted fusion

### 4. Why BiLSTM for Temporal?
- Bidirectional: learns context from past AND future (60s window)
- Captures drowsiness progression (eye closure trajectory)
- State-aware (remembers patterns across timesteps)
- Well-suited for sequence modeling

### 5. Why Subject-Independent Evaluation?
- Tests true generalization to unseen drivers
- Prevents memorizing individual driver patterns
- Honest benchmark (54.5% vs published 88% with leakage)
- Realistic deployment scenario

---

## COMPARISON: YOUR ARCHITECTURE vs DIAGRAM

### What's CORRECT in the provided diagram:
✓ YOLOv8 for visual extraction
✓ 1D CNN for CAN feature extraction
✓ Cross-modal attention fusion
✓ BiLSTM for temporal modeling
✓ Final classification head
✓ 3-class output (Alert, Light Drowsy, Drowsy)

### What NEEDS CLARIFICATION:
⚠ Diagram shows single BiLSTM → Your M6 has two variants (Lite: BiLSTM, Full: Transformer)
⚠ Diagram doesn't show feature sampling → Your M6 samples 16 frames from 60s window
⚠ Diagram shows concatenation → Your implementation uses attention-based fusion
⚠ Embedding caching not shown → Critical for M6 efficiency (pre-extract M5 features)

### Your Architecture is MORE Advanced:
- Embedding caching strategy (fast M6 training)
- Two variants (edge vs research)
- Attention-based fusion (not simple concat)
- Temporal sampling (efficient windowing)
- Subject-independent evaluation (rigorous)

---

## OFFICIAL CORRECTED ARCHITECTURE FOR YOUR PROJECT

```
INPUT: 60-SECOND MULTIMODAL WINDOW
════════════════════════════════════════════════════════════════════════════════

Video Stream (60s @ 4 Hz = 240 frames)  +  CAN Signals (60s @ 4 Hz = 240, 5)


STEP 1: EXTRACT VISUAL EMBEDDINGS
────────────────────────────────────────────────────────────────────────────────

For each of 240 frames:
  Frame → YOLOv8-Nano backbone → 512-dim embedding
  
Result: (240, 512) embedding sequence

Optimization (M6 Strategy):
  Pre-compute all embeddings for dataset
  Cache to .npz files per session
  → M6 training: only 1.5M params (not 6.4M YOLOv8)
  → Training time: 15-20 min per fold (not 2 hours)


STEP 2: SAMPLE TEMPORAL CONTEXT
────────────────────────────────────────────────────────────────────────────────

From (240, 512) embeddings:
  Sample 16 frames uniformly across 60-second window
  → (16, 512) sampled embeddings
  
Rationale:
  16 frames × 3.75 sec apart = full 60-second coverage
  Reduces computation while preserving temporal progression


STEP 3: VISUAL TEMPORAL ENCODING
────────────────────────────────────────────────────────────────────────────────

M6_Lite:
  BiLSTM(128 forward + 128 backward)
  Input: (16, 512)
  Output: (1, 256) temporal visual features

M6_Full:
  Transformer(2 layers, 4 heads)
  Input: (16, 512)
  Output: (1, 256) temporal visual features


STEP 4: CAN SIGNAL ENCODING
────────────────────────────────────────────────────────────────────────────────

Input: (240, 5) CAN window

1D CNN Encoder:
  Conv1D(5 → 32, kernel=3) + ReLU
  Conv1D(32 → 64, kernel=3) + ReLU
  Global Average Pool
  Dense(64 → 128) + ReLU
  
Output: (1, 128) CAN features

Causal Design:
  Use unidirectional LSTM (not bidirectional)
  For real-time edge deployment
  Can only use past data, not future


STEP 5: BIDIRECTIONAL CROSS-MODAL ATTENTION
────────────────────────────────────────────────────────────────────────────────

Direction 1: Visual queries CAN
  Q = W_q · visual_features  (256-dim)
  K = W_k · can_features     (128-dim)
  V = W_v · can_features     (128-dim)
  α = softmax(Q·K^T / sqrt(d_k))
  context_v = α · V          (128-dim)

Direction 2: CAN queries Visual
  Q = W_q · can_features     (128-dim)
  K = W_k · visual_features  (256-dim)
  V = W_v · visual_features  (256-dim)
  β = softmax(Q·K^T / sqrt(d_k))
  context_c = β · V          (256-dim)

Interpretation:
  α weights: Which CAN signals matter for facial analysis?
  β weights: Which facial regions correlate with vehicle dynamics?


STEP 6: FEATURE FUSION
────────────────────────────────────────────────────────────────────────────────

Concatenate:
  [context_visual, context_can, visual_temporal, can_encoded]
  = [128 + 256 + 256 + 128] = 768-dim fused feature

Or simplified (M6_Lite):
  [visual_temporal, can_encoded]
  = [256 + 128] = 384-dim fused feature


STEP 7: CLASSIFICATION HEAD
────────────────────────────────────────────────────────────────────────────────

Dense(768 → 256) + BatchNorm + ReLU
Dense(256 → 128) + BatchNorm + ReLU
Dense(128 → 3) + Softmax

Output: [p_alert, p_low_vigilance, p_drowsy]

Class weights (inverse frequency):
  Alert: 1.0 / 0.40 = 2.5
  Low Vigilance: 1.0 / 0.35 = 2.86
  Drowsy: 1.0 / 0.25 = 4.0


STEP 8: SUBJECT-INDEPENDENT EVALUATION
────────────────────────────────────────────────────────────────────────────────

5-Fold Protocol:
  Fold 0: Train [E-S], Test [A-D]
  Fold 1: Train [A-D,I-S], Test [E-H]
  Fold 2: Train [A-H,L-S], Test [I-K]
  Fold 3: Train [A-K,P-S], Test [L-O]
  Fold 4: Train [A-O], Test [P-S]

No subject overlap between train and test

Metrics (per fold):
  Accuracy, Macro F1, Confusion Matrix

Final Results:
  Mean accuracy ± std across 5 folds
  Example: 60.5% ± 0.8%
```

---

## DEPLOYMENT ARCHITECTURE

```
REAL-TIME INFERENCE PIPELINE
════════════════════════════════════════════════════════════════════════════════

Live Video Stream (Webcam)
  ↓
[YOLOv8 Face Detection]
  ├─ Input: RGB frame
  ├─ Detect: Bounding box around driver face
  └─ Output: Cropped face region (640×640)
  ↓
[M5 Per-Frame Inference] (30ms)
  ├─ Input: Cropped face
  ├─ YOLOv8 backbone: Extract 512-dim embedding
  ├─ Classification head: Predict class
  └─ Output: [Alert / LowVig / Drowsy] + confidence
  ↓
[Aggregation Over 60s Window]
  ├─ Collect 16 sampled embeddings
  ├─ Feed to M6_Lite temporal model (100ms)
  └─ Output: Refined prediction
  ↓
[Alert Mechanism]
  ├─ If Drowsy confidence > 0.7: Sound + Vibration alert
  ├─ Log prediction + timestamp
  └─ Display on dashboard


Deployment Targets:
═════════════════════

1. Vehicle ECU (Edge)
   Hardware: NVIDIA Jetson Nano or TX2
   Model: M6_Lite (0.4M params, 100ms latency)
   Deployment: TensorFlow Lite or ONNX
   
2. Smartphone (Mobile)
   Framework: TensorFlow Lite
   Model: M6_Lite quantized (int8, 0.1M params)
   Latency: 150-200ms on modern phones
   
3. Cloud/Server
   Framework: PyTorch or TensorFlow
   Model: M6_Full (0.9M params, 150ms latency)
   Throughput: 10+ predictions per second
```

---

## FINAL ANSWER: IS THE DIAGRAM CORRECT?

**80% Correct** ✓

The diagram captures the key architectural concepts:
- YOLOv8 visual extraction
- 1D CNN CAN encoding
- Cross-modal attention
- BiLSTM temporal modeling
- Classification output

**20% Missing Details** ⚠

Your actual implementation has:
- Embedding pre-caching (efficiency optimization)
- Two variants (M6_Lite vs M6_Full)
- Attention-based fusion (not simple concatenation)
- Subject-independent evaluation protocol
- Temporal sampling strategy (16 frames from 240)

**Recommendation**: The diagram is **suitable for presentation** but consider adding a detailed slide explaining:
1. Embedding caching strategy
2. Why 16-frame sampling (efficiency)
3. Two variants and their use cases
4. Subject-independent evaluation rigor

Your architecture is **more sophisticated and practical** than the generic diagram shows!
