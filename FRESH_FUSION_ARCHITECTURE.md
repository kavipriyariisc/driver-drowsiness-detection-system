# Fresh Fusion Architecture Visualization

## System Overview Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    UL-DD DATASET (19 Subjects, Preprocessed)                     │
│                                                                                  │
│  ├─ Fold 0: (X_fau_train, mm_tele_train, y_train) + (X_fau_test, mm_tele_test) │
│  ├─ Fold 1: ...                                                                 │
│  └─ Fold 4: ...                                                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │    Subject-wise Splitting (70/15/15)  │
                    │   No subject overlap between splits    │
                    └───────────────────┬───────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
        ┌───────▼────────┐      ┌───────▼────────┐      ┌───────▼────────┐
        │ TRAIN SPLIT    │      │ VAL SPLIT      │      │ TEST SPLIT     │
        │ 70% subjects   │      │ 15% subjects   │      │ 15% subjects   │
        │ ~4000 samples  │      │ ~700 samples   │      │ ~1000 samples  │
        └───────┬────────┘      └───────┬────────┘      └───────┬────────┘
                │                       │                       │
                └───────────────────────┼───────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │    Data Preprocessing & Alignment     │
                    │                                       │
                    │  ├─ Extract FAU (T=240, F=30)        │
                    │  ├─ Extract Telemetry (T=240, F=5)   │
                    │  ├─ Normalize using train stats       │
                    │  └─ Engineer telemetry (F=5 → F=25)  │
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
        ┌───────▼──────────────┐ ┌─────▼──────────────┐ ┌─────▼───────────┐
        │  CAMERA MODALITY     │ │ FACIAL MODALITY    │ │ TELEMETRY MOD   │
        │                      │ │                    │ │                 │
        │  [CameraTemporalNet] │ │ [FacialAttention]  │ │ [TelemetryTrans]│
        │                      │ │                    │ │                 │
        │ Input: (B,8,3,224²)  │ │ Input: (B,T,30)    │ │ Input: (B,T,25) │
        │ Output: (B,3) logits │ │ Output: (B,3)      │ │ Output: (B,3)   │
        │ Embed: (B,128)       │ │ Embed: (B,192)     │ │ Embed: (B,96)   │
        └───────┬──────────────┘ └─────┬──────────────┘ └─────┬───────────┘
                │                      │                      │
                │      ┌───────────────┴──────────────┐       │
                │      │                              │       │
                └──────┼─ Extract all embeddings ────┼───────┘
                       │                              │
                    ┌──▼────────────────────────────▼──┐
                    │  FUSION MODALITY                 │
                    │                                 │
                    │ [CrossModalFusionNet]           │
                    │                                 │
                    │ ├─ Project to common dim (128) │
                    │ ├─ Compute confidence scores   │
                    │ ├─ Softmax learned weights     │
                    │ ├─ Weighted fusion             │
                    │ └─ Concatenate + classify      │
                    │                                 │
                    │ Input: 3 embeddings            │
                    │ Output: (B,3) logits + weights │
                    └──┬────────────────────────────┬──┘
                       │                            │
              ┌────────▼──────────┐    ┌───────────▼─────────┐
              │  PREDICTIONS      │    │  MODALITY WEIGHTS   │
              │                   │    │                     │
              │  ├─ Class 0 logit │    │ ├─ Visual weight    │
              │  ├─ Class 1 logit │    │ ├─ Facial weight    │
              │  └─ Class 2 logit │    │ └─ Telemetry wt     │
              └────────┬──────────┘    └───────────┬─────────┘
                       │                           │
              ┌────────▼───────────────────────────▼────────┐
              │         EVALUATION & METRICS               │
              │                                            │
              │  ├─ Accuracy: ?%                          │
              │  ├─ Macro-F1: ?                           │
              │  ├─ Per-class precision/recall/F1         │
              │  ├─ Confusion matrix                      │
              │  └─ Comparison with baseline              │
              │                                            │
              │  📊 Results saved to:                     │
              │    - fresh_fusion_summary.csv             │
              │    - fresh_fusion_results.json            │
              │    - fusion_modality_weights.json         │
              └────────────────────────────────────────────┘
```

---

## Individual Model Architectures

### Model A: CameraTemporalNet

```
                    Frames
                    (B, 8, 3, 224, 224)
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Process Each Frame With      │
        │   MobileNetV3-Small Backbone   │
        └────────────────────────────────┘
                         │
                         ▼
              Frame Features (B×8, 1152)
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Project to 128-dim Embedding │
        │   per frame                    │
        └────────────────────────────────┘
                         │
                         ▼
           Frame Embeddings (B, 8, 128)
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Temporal Attention Pooling   │
        │   Learn importance weights     │
        │   for each frame               │
        └────────────────────────────────┘
                         │
                         ▼
           Visual Embedding (B, 128)
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Classifier Head              │
        │   Dense 128 → ReLU → Dropout  │
        │   Dense 128 → ReLU → Dropout  │
        │   Dense 64  → ReLU → Dropout  │
        │   Dense 3   (Logits)          │
        └────────────────────────────────┘
                         │
                         ▼
                    Logits (B, 3)
                    
Output Methods:
- forward(frames) → logits
- encode_visual(frames) → 128-dim embedding
```

---

### Model B: FacialAttentionNet

```
              Facial Action Units
              (B, T=240, F=30)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Input Projection         │
        │   30 features → 64 features│
        └────────────────────────────┘
                     │
                     ▼
         Projected FAU (B, 240, 64)
                     │
                     ▼
        ┌────────────────────────────┐
        │   BiLSTM Encoder           │
        │   - 2 layers               │
        │   - hidden_size = 96       │
        │   - bidirectional          │
        │   Output: 192-dim (2×96)   │
        └────────────────────────────┘
                     │
                     ▼
         LSTM Outputs (B, 240, 192)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Attention Pooling        │
        │   - Learn importance per   │
        │     timestep               │
        │   - Weighted sum           │
        └────────────────────────────┘
                     │
                     ▼
        Facial Embedding (B, 192)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Classifier Head          │
        │   Dense 192 → ReLU        │
        │   Dense 128 → ReLU        │
        │   Dense 64  → ReLU        │
        │   Dense 3   (Logits)      │
        └────────────────────────────┘
                     │
                     ▼
                Logits (B, 3)
                
Output Methods:
- forward(facial) → logits
- encode_facial(facial) → 192-dim embedding
```

---

### Model C: TelemetryTransformerNet

```
                Raw Telemetry
              (B, T=240, F=5)
              - Speed, RPM, Gear
              - Pitch, Roll
                     │
                     ▼
        ┌────────────────────────────┐
        │   Feature Engineering      │
        │   For each signal:         │
        │   1. Raw value             │
        │   2. 1st derivative (Δx)   │
        │   3. 2nd derivative (ΔΔx)  │
        │   4. Rolling mean (win=5)  │
        │   5. Rolling std (win=5)   │
        │   5 signals → 25 features  │
        └────────────────────────────┘
                     │
                     ▼
       Engineered Features (B, 240, 25)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Input Projection         │
        │   25 features → 96 features│
        └────────────────────────────┘
                     │
                     ▼
         Projected (B, 240, 96)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Positional Encoding      │
        │   Add sinusoidal encoding  │
        │   to preserve position info│
        └────────────────────────────┘
                     │
                     ▼
        With Positions (B, 240, 96)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Transformer Encoder      │
        │   - d_model = 96           │
        │   - nhead = 4              │
        │   - num_layers = 2         │
        │   - Self-attention         │
        │   - Feed-forward           │
        └────────────────────────────┘
                     │
                     ▼
        Transformer Out (B, 240, 96)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Attention Pooling        │
        │   - Learn importance per   │
        │     timestep               │
        │   - Weighted sum           │
        └────────────────────────────┘
                     │
                     ▼
      Telemetry Embedding (B, 96)
                     │
                     ▼
        ┌────────────────────────────┐
        │   Classifier Head          │
        │   Dense 96 → ReLU         │
        │   Dense 64 → ReLU         │
        │   Dense 3  (Logits)       │
        └────────────────────────────┘
                     │
                     ▼
                Logits (B, 3)
                
Output Methods:
- forward(telemetry) → logits
- encode_telemetry(telemetry) → 96-dim embedding
```

---

### Model D: CrossModalFusionNet (Learned Fusion)

```
    Visual (128)      Facial (192)      Telemetry (96)
         │                 │                 │
         ▼                 ▼                 ▼
    ┌────────┐         ┌────────┐       ┌────────┐
    │Project │         │Project │       │Project │
    │128→128 │         │192→128 │       │96→128  │
    └────┬───┘         └────┬───┘       └────┬───┘
         │                 │                 │
         ▼                 ▼                 ▼
    128-dim           128-dim           128-dim
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Compute Confidence Scores      │
        │   For each modality:             │
        │   score = Linear(128 → 1)        │
        │                                  │
        │   visual_score (B, 1)            │
        │   facial_score (B, 1)            │
        │   telemetry_score (B, 1)         │
        └──────────────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Softmax Weighting              │
        │   weights = softmax([scores])    │
        │                                  │
        │   w_visual    (B,)               │
        │   w_facial    (B,)               │
        │   w_telemetry (B,)               │
        │   All sum to 1.0                 │
        └──────────────────┬───────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    w_visual*         w_facial*          w_telemetry*
    visual_proj       facial_proj        telemetry_proj
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Fusion Embedding               │
        │   Sum of weighted modalities     │
        │   fusion_emb = Σ(w * emb)       │
        │   Shape: (B, 128)                │
        └──────────────────┬───────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    visual_proj       facial_proj        telemetry_proj
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    fusion_embedding
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
         Final Concat: [visual | facial | telemetry | fusion]
                    (512-dim representation)
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   Deep Classifier                │
        │   Dense 512 → ReLU → Dropout(0.5)│
        │   Dense 256 → ReLU → Dropout(0.4)│
        │   Dense 128 → ReLU → Dropout(0.3)│
        │   Dense 64                       │
        │   Dense 3   (Logits)             │
        └──────────────┬───────────────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          Logits (B, 3)    Weights dict
                           {visual, facial,
                            telemetry}

Output Method:
forward(frames, facial, telemetry) → dict
{
    'logits': (B, 3),
    'weights': {
        'visual': (B,),
        'facial': (B,),
        'telemetry': (B,),
    },
    'embeddings': {
        'visual': (B, 128),
        'facial': (B, 128),
        'telemetry': (B, 128),
        'fusion': (B, 128),
    }
}
```

---

## Training Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    INDIVIDUAL MODEL TRAINING                       │
│                    (Parallel or Sequential)                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Train CameraTemporalNet                                      │ │
│  │ - Load camera-specific dataloaders                          │ │
│  │ - Create CameraTemporalNet model                            │ │
│  │ - Setup CameraTrainer                                       │ │
│  │ - Train with validation monitoring                          │ │
│  │ - Save best checkpoint                                      │ │
│  │ - Evaluate on test set                                      │ │
│  │ Output: camera_temporal_best.pt                             │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Train FacialAttentionNet                                     │ │
│  │ - Load facial-specific dataloaders                          │ │
│  │ - Create FacialAttentionNet model                           │ │
│  │ - Setup FacialTrainer                                       │ │
│  │ - Train with validation monitoring                          │ │
│  │ - Save best checkpoint                                      │ │
│  │ - Evaluate on test set                                      │ │
│  │ Output: facial_attention_best.pt                            │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Train TelemetryTransformerNet                                │ │
│  │ - Load telemetry-specific dataloaders                       │ │
│  │ - Auto-engineer features (5 → 25)                          │ │
│  │ - Create TelemetryTransformerNet model                      │ │
│  │ - Setup TelemetryTrainer                                    │ │
│  │ - Train with validation monitoring                          │ │
│  │ - Save best checkpoint                                      │ │
│  │ - Evaluate on test set                                      │ │
│  │ Output: telemetry_transformer_best.pt                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            │
                  All 3 models trained
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                    FUSION MODEL TRAINING                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ├─ Load pretrained individual models                            │
│  ├─ Create multimodal dataloaders                               │
│  ├─ Create CrossModalFusionNet                                  │
│  │  - Optionally freeze encoder weights                         │
│  ├─ Setup FusionTrainer                                         │
│  ├─ Train with validation monitoring                            │
│  ├─ Extract and log modality weights                            │
│  ├─ Save best checkpoint                                        │
│  └─ Evaluate on test set                                        │
│                                                                  │
│  Output: cross_modal_fusion_best.pt + weights                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            │
                 Fusion model trained
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                  COMPREHENSIVE EVALUATION                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  For each model:                                                  │
│  ├─ Load best checkpoint                                         │
│  ├─ Run on test set                                              │
│  ├─ Compute metrics:                                             │
│  │  - Accuracy                                                   │
│  │  - Macro-F1                                                   │
│  │  - Balanced Accuracy                                          │
│  │  - Per-class Precision/Recall/F1                             │
│  │  - Confusion Matrix                                          │
│  └─ Save results to JSON                                         │
│                                                                    │
│  Create comparison table:                                         │
│  ├─ Model | Accuracy | Macro-F1 | Balanced Acc | Class Metrics  │
│  ├─ Camera|          |          |             |                  │
│  ├─ Facial|          |          |             |                  │
│  ├─ Telem |          |          |             |                  │
│  └─ Fusion|          |          |             |                  │
│                                                                    │
│  Save outputs:                                                    │
│  ├─ fresh_fusion_summary.csv                                     │
│  ├─ fresh_fusion_results.json                                    │
│  └─ fusion_modality_weights.json                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example (Single Sample)

```
Sample from Dataset:
{
    'fau': Tensor(240, 30),        # FAU features over 60 sec
    'tele': Tensor(240, 5),        # Raw telemetry over 60 sec
    'label': 1,                     # Low vigilance
    'subject': 'E',                # Subject ID
    'window_id': 42,               # Window number
}
        │
        ├─────────────────────┬─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
    CAMERA PATH          FACIAL PATH          TELEMETRY PATH
    (Synthetic)          (Real)               (Real → Engineered)
        │                     │                     │
        ▼                     ▼                     ▼
    Tensor               Tensor                Engineered
    (8,3,224,224)        (240,30)              Tensor
                                               (240,25)
        │                     │                     │
        ▼                     ▼                     ▼
    Camera Model         Facial Model         Telemetry Model
        │                     │                     │
        ▼                     ▼                     ▼
    Logits               Logits                Logits
    (3,)                 (3,)                  (3,)
        │                     │                     │
        ├─────────────────────┼─────────────────────┤
        │                     │                     │
        │  Embedding vectors extracted:             │
        ├─ Visual (128)                            │
        ├─ Facial (192)                            │
        └─ Telemetry (96)                          │
                │
                ▼
          Fusion Model
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
    Logits        Weights Dict
    (3,)          {visual, facial, telemetry}
```

---

## Evaluation Flow

```
For Each Model:

Load Best Checkpoint
        │
        ▼
Test Dataset Loop:
├─ Forward pass
├─ Get predictions
├─ Get ground truth
└─ Store for metrics

Compute Metrics:
├─ Accuracy
├─ Macro-F1
├─ Per-class metrics
├─ Confusion matrix
└─ Classification report

Save Results:
├─ JSON with all metrics
├─ Embeddings NPY file
└─ Training history JSON

Create Comparison:
├─ Build table across models
├─ Save as CSV
├─ Log to console
└─ Identify best model

Extract Modality Weights (Fusion only):
├─ Average weights per modality
├─ Per-class breakdown optional
└─ Save to JSON
```

---

## Class Labels & Meanings

```
┌─────────────────────────────────────────────────────┐
│ CLASS DISTRIBUTION ACROSS SPLITS                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Class 0 (Alert):                                   │
│ └─ Fully awake, vigilant                          │
│    ~30-40% of samples                             │
│                                                     │
│ Class 1 (Low Vigilance):                           │
│ └─ Reduced alertness, early drowsiness            │
│    ~35-45% of samples                             │
│                                                     │
│ Class 2 (Very Drowsy):                            │
│ └─ Severe drowsiness, high-risk                   │
│    ~15-25% of samples                             │
│                                                     │
│ Imbalance Handling:                               │
│ └─ Class weights: 1 / class_count                 │
│    (automatic in training)                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

**Last Updated:** June 18, 2026  
**Architecture Version:** 1.0  
**Status:** Complete & Verified ✓
