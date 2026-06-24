# UL-DD M6 Redesign

## Objective

Improve the current M6 model for Driver Drowsiness Detection on the UL-DD dataset by fixing synchronization issues between visual and telemetry modalities and introducing proper temporal multimodal fusion.

The goal is not to replace the existing project but to build a stronger and scientifically valid extension of M5.

---

# Existing Models

| Model | Input                                |
| ----- | ------------------------------------ |
| M1    | Facial Action Units (FAU)            |
| M2    | Telemetry Signals                    |
| M3    | FAU + Telemetry Fusion               |
| M5    | YOLOv8 Infrared Image Classification |
| M6    | YOLO Embeddings + Telemetry Fusion   |

---

# Current M6 Problem

The existing M6 implementation does not correctly align visual embeddings with telemetry windows.

Current behavior:

```text
Telemetry Window
        +
Visual Embeddings
from another session
```

This breaks multimodal learning.

As a result:

* Camera features and telemetry features are unrelated.
* Fusion becomes noisy.
* M6 performance becomes unreliable.

---

# Proposed M6 Solution

## Core Idea

For every telemetry window:

```text
Telemetry Window
        +
Visual Frames
from SAME subject
from SAME session
from SAME time range
```

Only synchronized data should be fused.

---

# Dataset Settings

Keep the existing preprocessing pipeline.

```text
Sampling Rate = 4 Hz
Window Length = 60 seconds
Stride = 15 seconds
Classes = 3
```

Classes:

```text
0 = Alert
1 = Low Vigilant
2 = Drowsy
```

Evaluation:

```text
5-Fold Subject Independent Cross Validation
```

---

# Step 1: Visual Feature Extraction

Use the trained M5 model.

Remove the classification head.

Extract:

```text
512-Dimensional Feature Vector
```

for every infrared frame.

Store:

```text
Subject
Session
Timestamp
Frame Index
Embedding
```

Example:

```text
A_A.npz
A_D.npz
B_A.npz
B_D.npz
...
```

---

# Step 2: Window Metadata

Generate metadata for every telemetry window.

Store:

```text
subject
session
window_id
start_time
end_time
label
```

Example:

```text
Subject B
Session D
Window 15

Start = 300 sec
End = 360 sec

Label = Drowsy
```

---

# Step 3: Alignment

For each telemetry window:

```text
[start_time, end_time]
```

Select only visual embeddings belonging to that interval.

Example:

```python
frames = embeddings[
    (timestamp >= start_time)
    &
    (timestamp < end_time)
]
```

Then uniformly sample:

```text
T_VIS = 16
```

embeddings.

Output:

```text
(16,512)
```

visual sequence.

---

# M6-A: Temporal Vision Baseline

## Purpose

Strong temporal visual model without telemetry.

Architecture:

```text
YOLO Embeddings
      ↓
BiLSTM(128)
      ↓
Attention Pooling
      ↓
Dense(128)
      ↓
Dropout(0.3)
      ↓
Softmax(3)
```

Input:

```text
(16,512)
```

Output:

```text
Alert
Low Vigilant
Drowsy
```

---

# M6-B: Transformer Vision Model

## Purpose

Stronger temporal modeling.

Architecture:

```text
YOLO Embeddings
      ↓
Linear Projection
      ↓
Positional Encoding
      ↓
Transformer Encoder × 2
      ↓
Attention Pooling
      ↓
Dense
      ↓
Softmax
```

---

# M6-C: Final Multimodal Fusion Model

## Visual Branch

```text
YOLO Embeddings
      ↓
Transformer Encoder
      ↓
Visual Representation
```

---

## Telemetry Branch

Input:

```text
(240,5)
```

Features:

```text
Pitch
Roll
Speed
RPM
Gear
```

Architecture:

```text
Telemetry
      ↓
BiLSTM(64)
      ↓
Telemetry Representation
```

---

## Fusion Layer

```text
Visual Representation
          ↕
Cross Modal Attention
          ↕
Telemetry Representation
```

---

## Classification Head

```text
Concatenate
      ↓
Dense(128)
      ↓
Dropout(0.3)
      ↓
Softmax(3)
```

---

# Training Configuration

## Optimizer

```text
AdamW
```

Parameters:

```text
Learning Rate = 3e-4
Weight Decay = 1e-4
```

---

## Regularization

```text
Dropout = 0.3
Gradient Clipping = 1.0
```

---

## Loss Function

Default:

```text
Weighted Cross Entropy
```

Optional:

```text
Focal Loss
Gamma = 2
```

---

## Early Stopping

Monitor:

```text
Validation Macro-F1
```

Patience:

```text
10 Epochs
```

---

# Validation Strategy

Do NOT use test folds during training.

Use:

```text
Training Subjects
      ↓
GroupShuffleSplit
      ↓
Train Set
Validation Set
```

Validation must be subject-based.

---

# Fair Comparison

Current comparison:

```text
M1/M2/M3 → Window Level
M5 → Frame Level
```

Not fully fair.

Create:

## M5 Window Baseline

For each 60-second window:

```text
Collect all M5 frame predictions
Average probabilities
Generate final prediction
```

Then compare:

```text
M5 Window
vs
M6 Window
```

on identical windows.

---

# Recommended Ablations

## Window Length

```text
30 seconds
60 seconds
```

---

## Stride

```text
15 seconds
30 seconds
```

---

## Visual Sequence Length

```text
8
16
32
```

---

## Temporal Model

```text
BiLSTM
Transformer
```

---

## Modality Comparison

```text
Vision Only
Vision + Telemetry
```

---

# Expected Thesis Flow

## M1

FAU Baseline

---

## M2

Telemetry Baseline

---

## M3

FAU + Telemetry Fusion

---

## M5

Infrared Vision Baseline

---

## M6-A

Temporal Vision Model

---

## M6-C

Temporal Multimodal Fusion Model

---

# Success Criteria

The redesigned M6 should:

1. Use synchronized camera and telemetry data.
2. Maintain subject-independent evaluation.
3. Compare fairly with M5.
4. Improve robustness over existing baselines.
5. Provide a clear thesis contribution.

---

# Final Thesis Contribution

Temporal Multimodal Driver Drowsiness Detection using Infrared Visual Features and Vehicle Telemetry on the UL-DD Dataset.

The contribution is achieved through:

* Temporal modeling of visual embeddings.
* Correct visual-telemetry synchronization.
* Cross-modal attention fusion.
* Subject-independent evaluation.
* Comparison against FAU, telemetry, and vision-only baselines.
