# Technology Stack Explained — Driver Drowsiness Detection

**Project**: UL-DD Multimodal Drowsiness Detection (MTech Thesis)  
**Stack**: TensorFlow/Keras + PyTorch + YOLOv8 + MediaPipe + OpenCV  
**Purpose**: Real-time and batch inference for driver drowsiness classification

---

## 🎯 Quick Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Data Ingestion & Preprocessing                                  │
│ ─────────────────────────────────────────────────────────────── │
│ • NumPy/Pandas      — Load CSVs, statistical operations         │
│ • OpenCV            — Image I/O, color space conversion         │
│ • Pillow            — Image loading/resizing                    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Real-Time Face Detection & Feature Extraction                   │
│ ─────────────────────────────────────────────────────────────── │
│ • YOLOv8 (Ultralytics)    — Detect face bounding box            │
│ • MediaPipe FaceMesh      — Extract 468 facial landmarks        │
│ • OpenCV                  — Draw annotations, geometry ops      │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Model Training (Dual Framework)                                  │
│ ─────────────────────────────────────────────────────────────── │
│ • TensorFlow/Keras    — M1, M2, M3, M4 (temporal models)       │
│ • PyTorch            — M6 (future: temporal + fusion)          │
│ • Scikit-learn       — Metrics (F1, confusion matrix)          │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Inference & Demo                                                 │
│ ─────────────────────────────────────────────────────────────── │
│ • TensorFlow/Keras    — Load M1-M4 checkpoints (.keras)        │
│ • Ultralytics         — Load M5 YOLOv8 (.pt checkpoint)        │
│ • OpenCV              — Webcam capture, real-time rendering    │
│ • Matplotlib/Seaborn  — Visualize results, confusion matrices  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 DETAILED BREAKDOWN BY FRAMEWORK

---

## 1. **TensorFlow/Keras** (Models M1–M4, ~85% of training code)

### What is it?
- **TensorFlow**: Google's deep learning framework (compute engine)
- **Keras**: High-level API on top of TensorFlow (user-friendly model definition)
- **Version**: ≥2.14.0 (latest stable with GPU auto-detection)

### What it's used for in the project:

#### ✅ **Define Neural Network Architectures (M1-M4)**

**Example: M1 Facial BiLSTM**
```python
from tensorflow.keras import layers, Model, Input

# Define input layer
input_fau = Input(shape=(240, 30))  # (timesteps, FAU features)

# Bidirectional LSTM for temporal encoding
x = layers.BiLSTM(128, return_sequences=True)(input_fau)
x = layers.Dropout(0.45)(x)
x = layers.BiLSTM(64)(x)  # return_sequences=False → collapses time
x = layers.Dropout(0.45)(x)

# Classification head
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
output = layers.Dense(3, activation='softmax')(x)  # 3 classes

model = Model(inputs=input_fau, outputs=output)
```

**Why?** BiLSTMs capture temporal patterns in facial expressions:
- **Bidirectional**: Can see past AND future context
- **LSTM cells**: Remember long-term dependencies (60-second window)
- **Dropout & BatchNorm**: Prevent overfitting, stabilize training

#### ✅ **Train Models with Loss Functions**

```python
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(),  # For multi-class (3 classes)
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Train with class weighting (handle imbalance: Alert 40%, Drowsy 25%)
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    class_weight={0: 1.0, 1: 1.0, 2: 1.6},  # Upweight Drowsy
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)
```

**Why?**
- **Categorical Crossentropy**: Standard loss for 3-class classification
- **Adam optimizer**: Adaptive learning rates, fast convergence
- **Class weights**: Downsample Alert, upweight Drowsy (imbalanced dataset)
- **Early stopping**: Stop training when validation loss plateaus (prevent overfitting)

#### ✅ **Custom Layers (M3 Cross-Modal Attention)**

```python
from tensorflow.keras import layers

class CrossModalAttentionBlock(layers.Layer):
    """Learn attention between two modalities (FAU ↔ Telemetry)"""
    
    def __init__(self, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.query_proj = layers.Dense(d_model)
        self.key_proj = layers.Dense(d_model)
        self.value_proj = layers.Dense(d_model)
        self.output_proj = layers.Dense(d_model)
    
    def call(self, query, key_value):
        """
        query:     (batch, time_q, d_q)      — First modality
        key_value: (batch, time_kv, d_kv)    — Second modality
        
        Returns:   (batch, time_q, d_model)  — Attended features
        """
        Q = self.query_proj(query)           # (batch, time_q, d_model)
        K = self.key_proj(key_value)         # (batch, time_kv, d_model)
        V = self.value_proj(key_value)       # (batch, time_kv, d_model)
        
        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.d_model, tf.float32)
        )  # (batch, time_q, time_kv)
        
        weights = tf.nn.softmax(scores, axis=-1)  # Attention weights
        context = tf.matmul(weights, V)            # (batch, time_q, d_model)
        output = self.output_proj(context)
        
        return output
```

**Why?** Custom layers provide:
- **Interpretability**: See which telemetry features influence facial predictions
- **Flexibility**: Design modality-specific attention
- **Modularity**: Reuse in M3 and future models

#### ✅ **Load Pre-trained Models for Inference**

```python
from tensorflow.keras.models import load_model

# Load M1 model
m1_model = load_model('models/checkpoints/M1_fold0.keras')

# For M3 (with custom layer), register custom objects
from models.architecture import CrossModalAttentionBlock
m3_model = load_model(
    'models/checkpoints/M3_fold0.keras',
    custom_objects={'CrossModalAttentionBlock': CrossModalAttentionBlock}
)

# Inference
fau_window = np.random.randn(1, 240, 30)  # Single window, batch size 1
probabilities = m1_model.predict(fau_window)  # (1, 3)
predicted_class = np.argmax(probabilities)    # 0 or 1 or 2
```

---

### 📊 TensorFlow/Keras Models in This Project

| Model | Input | Architecture | Code |
|-------|-------|--------------|------|
| **M1** | FAU (240, 30) | BiLSTM(128→64) | `src/models/architecture.py:build_m1_facial_bilstm()` |
| **M2** | Telemetry (240, 5) | LSTM(64→32) | `src/models/architecture.py:build_m2_telemetry_lstm()` |
| **M3** | FAU + Tele | Cross-modal attention | `src/models/architecture.py:M3FusionModel` |
| **M4** | 10 RT features | BiLSTM(64→32) | `src/models/architecture.py:build_m4_realtime_bilstm()` |

---

---

## 2. **PyTorch** (Model M6, ~15% of code, future direction)

### What is it?
- **PyTorch**: Facebook's deep learning framework (more pythonic than TensorFlow)
- **Version**: ≥2.0.0 (latest stable with GPU support)
- **Use case**: M6 temporal + multimodal fusion (newer models often use PyTorch)

### What it's used for:

#### ✅ **M6 Temporal Fusion Architecture**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class M6Lite(nn.Module):
    """Lightweight temporal + multimodal fusion"""
    
    def __init__(self, visual_dim=512, tele_dim=5, hidden=256, num_classes=3):
        super().__init__()
        
        # Visual branch: temporal encoding over 16 frozen embeddings
        self.visual_lstm = nn.LSTM(
            input_size=visual_dim,   # (16, 512) embeddings from frozen M5
            hidden_size=hidden,      # Learn temporal patterns
            bidirectional=True,
            batch_first=True
        )
        
        # Telemetry branch: causal encoding over 240 CAN samples
        self.tele_lstm = nn.LSTM(
            input_size=tele_dim,     # 5 CAN channels
            hidden_size=hidden,
            bidirectional=False,     # Unidirectional = causal (realistic for edge)
            batch_first=True
        )
        
        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden * 4, 128),  # 2×hidden from bi-LSTM + 2×hidden from uni-LSTM
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, visual_emb, tele_window):
        """
        visual_emb: (batch, 16, 512)  — sampled M5 embeddings
        tele_window: (batch, 240, 5)  — CAN signals over 60s
        
        Returns: logits (batch, 3)
        """
        # Visual: temporal encoding
        _, (h_vis_fwd, h_vis_bwd) = self.visual_lstm(visual_emb)  # (batch, hidden)
        h_vis = torch.cat([h_vis_fwd, h_vis_bwd], dim=-1)  # (batch, 2×hidden)
        
        # Telemetry: causal encoding
        _, (h_tele, _) = self.tele_lstm(tele_window)  # (batch, hidden)
        h_tele = h_tele.squeeze(0)
        
        # Concatenate and classify
        fused = torch.cat([h_vis, h_tele], dim=-1)  # (batch, 4×hidden)
        logits = self.fusion_head(fused)
        
        return logits
```

**Why PyTorch for M6?**
- **Dynamic computation graphs**: Easier to debug temporal models
- **Cleaner syntax**: More pythonic than Keras
- **Active research community**: Most new architectures (Transformers, attention) published in PyTorch first
- **Cross-framework compatibility**: Can load M5 YOLO model (also PyTorch)

#### ✅ **Extract M5 Embeddings for M6 Training**

```python
import torch
from ultralytics import YOLO

# Load frozen M5 YOLOv8 backbone
yolo_model = YOLO('models/checkpoints/M5_fold0.pt')

class M5FeatureExtractor:
    """Cache M5 embeddings for M6 training (avoid recomputation)"""
    
    def __init__(self, yolo_model):
        self.model = yolo_model
        # Extract intermediate layer (before classification head)
        self.backbone = yolo_model.model.model[:10]  # Get up to global pool
    
    def extract(self, frame):
        """
        frame: (3, H, W) or (3, 640, 640)
        
        Returns: (512,) embedding
        """
        with torch.no_grad():
            frame = torch.from_numpy(frame).unsqueeze(0).float()  # (1, 3, H, W)
            embedding = self.backbone(frame)  # (1, 512, 20, 20) → (1, 512)
            return embedding.squeeze().cpu().numpy()

# Usage: Pre-compute embeddings for all frames (cache to disk)
extractor = M5FeatureExtractor(yolo_model)
for frame in video_frames:
    emb = extractor.extract(frame)
    # Save to disk for M6 training (avoids recomputation)
```

---

---

## 3. **Ultralytics YOLOv8** (M5 Model, Real-Time Detection)

### What is it?
- **YOLO**: "You Only Look Once" — real-time object detection
- **YOLOv8**: Latest Ultralytics architecture (2023+)
- **YOLOv8-cls**: Classification variant (not detection), perfect for per-frame drowsiness
- **Version**: ≥8.0.0

### What it's used for:

#### ✅ **Per-Frame Drowsiness Classification (M5)**

```python
from ultralytics import YOLO

# Load pretrained YOLOv8-nano classifier
model = YOLO('yolov8n-cls.pt')  # Pretrained on ImageNet

# Train on UL-DD dataset
results = model.train(
    data='datasets/yolo_cls/fold_0',  # Directory structure:
                                       # fold_0/train/{Alert,LowVig,Drowsy}/
                                       # fold_0/val/{Alert,LowVig,Drowsy}/
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,  # GPU 0
    patience=20,
    augment=True,
)

# Inference: single frame or batch
frame = cv2.imread('face_crop.jpg')  # (H, W, 3) BGR
results = model.predict(frame)  # → Results object

# Parse results
for r in results:
    class_idx = r.probs.top1    # 0, 1, or 2
    confidence = r.probs.top1conf.item()
    class_name = r.names[class_idx]  # 'Alert', 'LowVigilant', 'Drowsy'
    print(f"Prediction: {class_name} ({confidence:.2%})")
```

#### ✅ **Real-Time Face Detection (Demo)**

```python
from ultralytics import YOLO

# YOLOv8 detection model (not classification)
face_detector = YOLO('yolov8n.pt')  # Pretrained on COCO (includes faces)

# Webcam capture loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in frame
    results = face_detector(frame)
    
    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        
        # Extract face crop
        face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Classify drowsiness
        class_results = drowsiness_classifier.predict(face_crop)
        class_idx = class_results[0].probs.top1
        confidence = class_results[0].probs.top1conf.item()
        
        # Draw on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_idx} ({confidence:.0%})", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Real-Time Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### ✅ **M5 Architecture Details**

```
YOLOv8-Nano Backbone:
  Input: (3, 640, 640) RGB image
  
  → Stem: Conv + Downsampling → (32, 320, 320)
  → Stage 1: Conv blocks → (64, 160, 160)
  → Stage 2: Conv blocks → (128, 80, 80)
  → Stage 3: Conv blocks → (256, 40, 40)
  → Stage 4: Conv blocks → (512, 20, 20)
  
  Global Average Pooling → (512,)
  Classification Head: Dense(512 → 3) → logits

Parameters: ~6.4M (but pretrained on ImageNet → few-shot adaptation)
Inference time: ~30-50ms on edge GPU
```

**Why YOLOv8 for M5?**
- **Transfer learning**: Pretrained on 14M ImageNet images
- **Real-time**: 30ms inference = 33 FPS (suitable for cars)
- **Simple**: One-liner training via Ultralytics CLI
- **Mobile-friendly**: Can quantize to int8 for edge deployment
- **High accuracy**: ~54.5% on subject-independent UL-DD (vs 39% for M1)

---

---

## 4. **MediaPipe** (Real-Time Facial Landmarks, M4)

### What is it?
- **MediaPipe**: Google's framework for building multimodal ML pipelines
- **FaceMesh**: Pre-trained model that detects 468 facial landmarks
- **Version**: ≥0.10.0

### What it's used for:

#### ✅ **Extract 468 Facial Landmarks**

```python
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2
import numpy as np

# Load FaceMesh landmark detector
BaseOptions = mp_python.BaseOptions
FaceLandmarker = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode = mp_vision.VisionRunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,  # Single image, not video stream
)

landmarker = FaceLandmarker.create_from_options(options)

# Process frame
frame = cv2.imread('face_crop.jpg')
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
detection_result = landmarker.detect(image)

# Extract landmarks
landmarks = detection_result.face_landmarks[0]  # 468 points
for landmark in landmarks:
    x, y, z = landmark.x, landmark.y, landmark.z  # Normalized [0, 1]
    print(f"Landmark: ({x}, {y}, {z})")
```

#### ✅ **Compute 10 Real-Time Drowsiness Features (M4 Input)**

```python
import math
from collections import deque

class RTFeaturesExtractor:
    """Extract 10 real-time features for M4 BiLSTM"""
    
    def __init__(self, buffer_size=240):
        self.landmarks_buffer = deque(maxlen=buffer_size)
        self.blink_buffer = deque(maxlen=60)  # 60 frames = 15 seconds @ 4 Hz
    
    def compute_eye_aspect_ratio(self, eye_landmarks):
        """
        Eye Aspect Ratio (EAR) — proxy for eye openness
        
        Landmarks: [eye_outer, eye_top, eye_inner, eye_bottom, ...]
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
        
        Typical: EAR > 0.2 = eye open, EAR < 0.15 = eye closed
        """
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0
    
    def compute_mouth_aspect_ratio(self, mouth_landmarks):
        """
        Mouth Aspect Ratio (MAR) — proxy for yawning/fatigue
        
        Similar to EAR but for mouth corners
        High MAR = yawning = drowsiness indicator
        """
        A = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[7])
        B = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
        C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])
        return (A + B) / (2.0 * C) if C > 0 else 0.0
    
    def compute_perclos(self, eye_ear_buffer, threshold=0.15):
        """
        PERCLOS — Percentage of Eye Closure
        
        Fraction of frames where EAR < threshold over rolling window (2s)
        High PERCLOS = eye mostly closed = drowsy
        
        Typical threshold: 0.15 (80% eye closure)
        """
        if len(eye_ear_buffer) == 0:
            return 0.0
        closed_count = sum(1 for ear in eye_ear_buffer if ear < threshold)
        return closed_count / len(eye_ear_buffer)
    
    def compute_blink_rate(self, blink_buffer):
        """
        Blink rate — blinks per minute
        
        Detect blink: Sharp drop in EAR (eye close → open)
        Low blink rate = fatigue (drowsy drivers blink less)
        """
        if len(blink_buffer) < 60:
            return 0.0
        blink_count = sum(blink_buffer)
        return blink_count / (60 / 4.0) * 60  # Convert to blinks/min
    
    def compute_head_pose(self, landmarks):
        """
        Head pose angles (pitch, roll, yaw)
        
        Use 6-point face model (corners + nose tip) for angle estimation
        
        Typical: Drowsy drivers show forward head nod (pitch > 15°)
        """
        # Simplified: use z-coordinates of landmarks for depth
        # In production: Use full 3D head pose estimation (PnP)
        
        nose_tip = landmarks[1]      # Nose tip landmark index
        chin = landmarks[152]        # Chin landmark index
        left_ear = landmarks[234]    # Left ear landmark
        right_ear = landmarks[454]   # Right ear landmark
        
        pitch = math.degrees(math.atan2(chin.z - nose_tip.z, chin.y - nose_tip.y))
        roll = math.degrees(math.atan2(left_ear.y - right_ear.y, right_ear.x - left_ear.x))
        yaw = math.degrees(math.atan2(left_ear.x - right_ear.x, right_ear.z - left_ear.z))
        
        return pitch, roll, yaw
    
    def extract_10_features(self, landmarks):
        """
        Extract 10 real-time features for M4
        
        Returns: (10,) feature vector
        """
        self.landmarks_buffer.append(landmarks)
        
        if len(self.landmarks_buffer) < 30:  # Not enough history
            return np.zeros(10)
        
        # Feature 0-1: Eye Aspect Ratios
        left_eye_landmarks = landmarks[33:133]  # Left eye region
        right_eye_landmarks = landmarks[263:363]  # Right eye region
        
        ear_left = self.compute_eye_aspect_ratio(left_eye_landmarks)
        ear_right = self.compute_eye_aspect_ratio(right_eye_landmarks)
        
        # Feature 2: Mouth Aspect Ratio
        mouth_landmarks = landmarks[61:101]
        mar = self.compute_mouth_aspect_ratio(mouth_landmarks)
        
        # Feature 3: PERCLOS (% eye closure)
        eye_ear_buffer = deque(
            [self.compute_eye_aspect_ratio(frame[33:133]) 
             for frame in list(self.landmarks_buffer)[-30:]],
            maxlen=30
        )
        perclos = self.compute_perclos(eye_ear_buffer)
        
        # Features 4-6: Head Pose Angles
        pitch, roll, yaw = self.compute_head_pose(landmarks)
        
        # Features 7-8: Brow & Jaw
        brow_raise = landmarks[107].y - landmarks[66].y  # Normalized distance
        jaw_drop = landmarks[177].y - landmarks[152].y   # Normalized distance
        
        # Feature 9: Blink rate (requires temporal history)
        blink_rate = self.compute_blink_rate(self.blink_buffer)
        
        return np.array([
            ear_left,       # 0
            ear_right,      # 1
            mar,            # 2
            perclos,        # 3
            pitch,          # 4
            roll,           # 5
            yaw,            # 6
            brow_raise,     # 7
            jaw_drop,       # 8
            blink_rate      # 9
        ])
```

---

---

## 5. **OpenCV** (Video I/O, Image Processing)

### What is it?
- **OpenCV**: Open-source computer vision library
- **Version**: ≥4.8.0

### What it's used for:

#### ✅ **Webcam Capture**

```python
import cv2

cap = cv2.VideoCapture(0)  # Open default camera

while True:
    ret, frame = cap.read()  # (H, W, 3) BGR format
    if not ret:
        break
    
    # frame is now a numpy array
    # Use for drowsiness detection...
    
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### ✅ **Image Processing & Drawing Annotations**

```python
import cv2
import numpy as np

# Read image
frame = cv2.imread('webcam_frame.jpg')  # (H, W, 3) BGR

# Color conversion
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # For face detection
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # For color-based detection

# Draw bounding box
cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Draw text
cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.0, color=(0, 0, 255), thickness=2)

# Draw circle (for landmark visualization)
cv2.circle(frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)

# Resize
resized = cv2.resize(frame, (640, 480))

# Display
cv2.imshow('Processed', frame)
cv2.waitKey(1)

# Save
cv2.imwrite('output.jpg', frame)
```

#### ✅ **Real-Time Demo Pipeline (realtime_demo.py)**

```python
import cv2
from src.face_detection import YOLOFaceDetector, RTFeaturesExtractor
from src.inference.predict import ULDDPredictor

# Initialize components
face_detector = YOLOFaceDetector('models/checkpoints/yolov8n.pt')
feature_extractor = RTFeaturesExtractor()
m5_predictor = ULDDPredictor('models/checkpoints/M5_fold0.pt', model_type='m5')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces using YOLOv8
    detections = face_detector.detect(frame)  # YOLOv8
    
    for x1, y1, x2, y2 in detections:
        # Extract face crop
        face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # M5 inference on face crop
        probs = m5_predictor.predict_image(face_crop)
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        
        # Draw results on frame
        color = (0, 255, 0) if class_idx == 0 else (0, 165, 255) if class_idx == 1 else (0, 0, 255)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        label = ['Alert', 'LowVigilant', 'Drowsy'][class_idx]
        cv2.putText(frame, f"{label} ({confidence:.0%})",
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    
    cv2.imshow('Real-Time Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

---

## 6. **NumPy & Pandas** (Data Processing)

### What it's used for:

#### ✅ **NumPy: Numerical Operations**

```python
import numpy as np

# Load raw UL-DD data (from CSV)
fau_data = np.loadtxt('datasets/raw/A_FAU_A.csv', delimiter=',', skiprows=1)
# Shape: (N_samples, 30) — 30 FAU features

# Downsampling: 60 Hz → 4 Hz (keep every 15th sample)
fau_downsampled = fau_data[::15, :]
# Shape: (N_samples // 15, 30)

# Sliding windows: Extract 60-second windows (240 timesteps)
window_size = 240
stride = 60

windows = []
for i in range(0, len(fau_downsampled) - window_size, stride):
    window = fau_downsampled[i:i+window_size, :]  # (240, 30)
    windows.append(window)

X = np.array(windows)  # (N_windows, 240, 30)

# Z-score normalization (per-feature)
mean = X.mean(axis=0)        # (240, 30) → (30,) if using global mean
std = X.std(axis=0)
X_normalized = (X - mean) / (std + 1e-8)

# Save to .npz (compressed numpy format)
np.savez_compressed(
    'datasets/processed/ul_dd/fold_0.npz',
    X_fau=X_fau_train,
    X_tele=X_tele_train,
    y=y_train,
    fau_mean=mean,
    fau_std=std
)
```

#### ✅ **Pandas: Structured Data**

```python
import pandas as pd

# Load UL-DD Labels
labels_df = pd.read_csv('datasets/raw/Labels.csv')
# Columns: [Subject, Session, Time, KSS]

# Load telemetry
tele_df = pd.read_csv('datasets/raw/A_Telemetry_A.csv')
# Columns: [Time, Pitch, Roll, Speed, RPM, Gear]

# Bin KSS into 3 classes
def kss_to_class(kss_value):
    if kss_value <= 3:
        return 0  # Alert
    elif kss_value <= 6:
        return 1  # Low Vigilant
    else:
        return 2  # Drowsy

labels_df['Class'] = labels_df['KSS'].apply(kss_to_class)

# Group by subject & session for windowing
for subject in labels_df['Subject'].unique():
    subject_data = labels_df[labels_df['Subject'] == subject]
    print(f"Subject {subject}: {len(subject_data)} samples")

# Merge FAU + telemetry on timestamp
merged = pd.merge(
    fau_df, 
    tele_df, 
    on='Time', 
    how='inner'  # Keep only timestamps present in both
)
```

---

---

## 7. **Scikit-Learn** (Metrics & Evaluation)

### What it's used for:

```python
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)

# After M5 inference on test set
y_true = [0, 1, 2, 0, 1, 1, 2, 2, 0]  # Ground truth
y_pred = [0, 1, 2, 0, 2, 1, 2, 1, 0]  # M5 predictions

# Overall accuracy
acc = accuracy_score(y_true, y_pred)  # 7/9 = 77.8%

# Per-class F1 scores (macro = average)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_per_class = f1_score(y_true, y_pred, average=None)
# [F1_Alert, F1_LowVig, F1_Drowsy]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
# [[TP_0, FP_0, FP_0],
#  [FN_1, TP_1, FP_1],
#  [FN_2, FN_2, TP_2]]

# Detailed classification report
print(classification_report(y_true, y_pred, 
                          target_names=['Alert', 'LowVig', 'Drowsy']))
# Output:
#               precision    recall  f1-score   support
#        Alert       0.75      1.00      0.86         3
#      LowVig       1.00      0.67      0.80         3
#       Drowsy      0.67      0.67      0.67         3
```

---

---

## 8. **Matplotlib & Seaborn** (Visualization)

### What it's used for:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Alert', 'LowVig', 'Drowsy'],
            yticklabels=['Alert', 'LowVig', 'Drowsy'])
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('M5 Confusion Matrix (Fold 0)')
plt.tight_layout()
plt.savefig('results/reports/M5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('M1 Training Curve')
ax1.legend()
ax1.grid()

ax2.plot(history.history['accuracy'], label='Training Acc')
ax2.plot(history.history['val_accuracy'], label='Validation Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('M1 Accuracy')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.savefig('results/reports/M1_training_history.png', dpi=150)
plt.close()

# Per-fold accuracy comparison
folds = [0, 1, 2, 3, 4]
m1_accs = [38.2, 39.5, 40.1, 38.9, 38.6]
m5_accs = [54.8, 54.2, 54.9, 54.1, 54.5]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(folds))
width = 0.35

ax.bar(x - width/2, m1_accs, width, label='M1 (39.1%)', alpha=0.8)
ax.bar(x + width/2, m5_accs, width, label='M5 (54.5%)', alpha=0.8)

ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Comparison: M1 vs M5')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in folds])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/reports/model_comparison.png', dpi=150)
plt.close()
```

---

---

## 📊 Tech Stack Summary by Component

```
┌─────────────────────────────────────────────────────────────────────────┐
│ COMPONENT                  │ FRAMEWORK(S)           │ PURPOSE             │
├─────────────────────────────────────────────────────────────────────────┤
│ Data Loading & I/O         │ NumPy, Pandas, OpenCV  │ Read CSVs, images   │
│ Data Preprocessing         │ NumPy, Pandas          │ Normalize, window    │
│ Training (M1–M4)           │ TensorFlow/Keras       │ Build & train models │
│ Training (M6)              │ PyTorch                │ Temporal fusion      │
│ Face Detection             │ YOLOv8 (Ultralytics)  │ Detect face regions  │
│ Face Landmarks             │ MediaPipe              │ 468 facial points    │
│ Per-Frame Classification   │ YOLOv8-cls             │ M5 model             │
│ Inference (M1–M4)          │ TensorFlow/Keras       │ Load & predict       │
│ Inference (M5)             │ Ultralytics YOLO       │ Load & predict       │
│ Evaluation Metrics         │ Scikit-learn           │ Accuracy, F1, CM     │
│ Visualization              │ Matplotlib, Seaborn    │ Plots, heatmaps      │
│ Real-Time Video Loop       │ OpenCV                 │ Webcam capture       │
│ Notebook Environment       │ Jupyter, IPython       │ Interactive analysis │
└─────────────────────────────────────────────────────────────────────────┘
```

---

---

## 🔄 Data Flow Through Stack

```
Raw UL-DD Dataset
  ├── Extracted_Features/ (30 FAU @ 60 Hz, per-subject CSV)
  ├── CSV_Files/ (5 CAN channels @ 60 Hz, per-subject CSV)
  └── Labels.csv (KSS ratings every 4 min)
       ↓
[Pandas/NumPy] — Load CSVs, parse structures
       ↓
Preprocessing (ULDDProcessor)
  • Downsample 60 Hz → 4 Hz [NumPy: keep every 15th sample]
  • Sliding windows: 60s @ 4Hz = 240 timesteps [NumPy: stride operations]
  • Z-score normalize [NumPy: (X - mean) / std]
  • Subject-independent 5-fold split [Pandas: groupby + filtering]
       ↓
[NumPy] Save to .npz files (compressed)
  fold_0.npz: X_fau (N, 240, 30), X_tele (N, 240, 5), y (N,)
       ↓
Training Phase (Framework-specific)
  ├─ M1–M4: [TensorFlow/Keras]
  │    • Load .npz → tf.data.Dataset
  │    • Build model: layers.BiLSTM(), layers.Dense(), etc.
  │    • model.fit() with callbacks (early stopping, checkpointing)
  │    • Save .keras checkpoints
  │
  └─ M6: [PyTorch]
       • Load .npz → torch.utils.data.DataLoader
       • Build model: nn.LSTM(), nn.Linear(), etc.
       • Training loop: forward → loss → backward → optimizer step
       • Save .pt checkpoints
       ↓
Inference / Real-Time Demo
  ├─ [OpenCV] Capture webcam frame (H, W, 3)
  ├─ [YOLOv8] Detect face bounding box
  ├─ [OpenCV] Extract face crop
  ├─ [MediaPipe] (optional) Extract 468 landmarks
  ├─ [YOLOv8-cls OR TensorFlow] Classify drowsiness
  ├─ [Scikit-learn] Compute metrics (if ground truth available)
  ├─ [Matplotlib] (optional) Plot results
  └─ [OpenCV] Draw on frame + display
       ↓
[Scikit-learn] Evaluate on test set
  • confusion_matrix(), f1_score(), accuracy_score()
  • Generate classification report
       ↓
[Matplotlib/Seaborn] Visualize results
  • Confusion matrix heatmap
  • Per-fold accuracy bars
  • Training curves
  • Save PNG reports
```

---

---

## 💡 Framework Comparison: Why Each?

| Framework | M1-M4 | M5 | M6 | Reason |
|-----------|-------|----|----|--------|
| **TensorFlow/Keras** | ✅ Primary | ❌ No | ✅ Secondary | High-level, easy custom layers (attention for M3) |
| **PyTorch** | ❌ No | ❌ No | ✅ Primary | Dynamic graphs, cleaner API for temporal models |
| **YOLOv8** | ❌ No | ✅ Primary | ❌ No | Real-time, pretrained ImageNet, classification task |
| **MediaPipe** | ❌ No | ❌ No | ✅ For M4 | 468 landmarks, optimized for mobile, free |
| **OpenCV** | ✅ I/O | ✅ Demo | ✅ Demo | Universal video/image I/O, drawing, color conversion |
| **NumPy** | ✅ | ✅ | ✅ | All numeric ops (windowing, normalization, matrix) |
| **Scikit-learn** | ✅ | ✅ | ✅ | Metrics computation (F1, confusion matrix) |

---

## 📦 Installation Command

```bash
pip install -r requirements.txt
```

This installs:
- `tensorflow>=2.14.0` — M1-M4 training
- `torch` (via `pytorch` package, not in requirements but can be added)
- `ultralytics>=8.0.0` — M5 YOLOv8
- `mediapipe>=0.10.0` — Face landmarks
- `opencv-python>=4.8.0` — Video I/O
- `numpy`, `pandas`, `scikit-learn` — Data processing
- `matplotlib`, `seaborn` — Visualization
- `jupyter` — Notebooks

---

## 🚀 Recommended Usage Pattern

```python
# 1. Data Loading & Preprocessing (NumPy + Pandas)
from src.data.preprocess import ULDDProcessor
processor = ULDDProcessor(raw_dir='datasets/raw', processed_dir='datasets/processed')
processor.process_all_folds()

# 2. Training M1-M4 (TensorFlow/Keras)
from src.models.architecture import build_m1_facial_bilstm
from src.models.train import train_model
model = build_m1_facial_bilstm()
history = train_model(model, fold=0)

# 3. Training M5 (Ultralytics YOLOv8)
from ultralytics import YOLO
yolo_model = YOLO('yolov8n-cls.pt')
results = yolo_model.train(data='datasets/yolo_cls/fold_0', epochs=100)

# 4. Training M6 (PyTorch)
import torch
from src.models.m6_fusion import M6Lite
model = M6Lite()
optimizer = torch.optim.Adam(model.parameters())
# ... training loop

# 5. Real-Time Demo (OpenCV + Multiple Frameworks)
from src.realtime_demo import DrowsyDemoApp
demo = DrowsyDemoApp(model_type='m5')
demo.run()

# 6. Evaluation (Scikit-learn + Matplotlib)
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

---

**Summary**: This project uniquely combines **TensorFlow (M1-M4), PyTorch (M6), YOLOv8 (M5), and MediaPipe (M4)** to build a production-ready multimodal drowsiness detection system. Each framework excels at its specific task, integrated via NumPy arrays and standard Python interfaces.
