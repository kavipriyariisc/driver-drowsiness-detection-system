# M5 Real-Time Video Demo (Prerecorded)

## Overview

A real-time drowsiness detection demonstration using **M5 (YOLOv8)** on prerecorded video frames from the test set. Instead of live webcam input, this demo processes saved video or frame sequences with per-frame predictions and visual overlays.

## Features

✅ **Real-Time Inference** — Process frames at video framerate  
✅ **M5 Checkpoint** — Uses trained YOLOv8n model  
✅ **Visual Overlays** — Confidence scores, class labels, color-coded predictions  
✅ **Prerecorded Video** — No live webcam required  
✅ **Statistics** — Per-frame predictions, confidence distributions  
✅ **Output Video** — Saves annotated video with predictions  

## Quick Start

### Option 1: Run with Subject/Session/Class from Test Set

```bash
python src/video_demo.py \
  --subject A \
  --session A \
  --class-name Drowsy
```

This will:
1. Find frames in `datasets/yolo_frames/A_A/Drowsy/`
2. Create a temporary video from frames
3. Run M5 inference on every frame
4. Save annotated output video to `results/demo_videos/`

### Option 2: Run with Custom Video File

```bash
python src/video_demo.py \
  --video /path/to/video.mp4 \
  --output results/demo_videos/my_demo.mp4
```

### Option 3: Run with Frame Directory

```bash
python src/video_demo.py \
  --video-dir datasets/yolo_frames/A_A/Drowsy \
  --output results/demo_videos/demo_drowsy.mp4
```

## Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--video` | str | Path to video file (e.g., `video.mp4`) |
| `--video-dir` | str | Directory with frame images (will create temporary video) |
| `--subject` | str | Subject ID (A, B, C, etc.) — uses yolo_frames |
| `--session` | str | Session ID (A, B, etc.) — uses yolo_frames |
| `--class-name` | str | Class name (Drowsy, Low Vigilant, Alert) — uses yolo_frames |
| `--checkpoint` | str | Path to M5 checkpoint (default: `models/checkpoints/M5_fold0.pt`) |
| `--output` | str | Output video path (default: auto-generated in `results/demo_videos/`) |
| `--device` | str | Device to use: `cpu` or `cuda` (default: `cpu`) |
| `--sample-rate` | int | Process every Nth frame (default: 1 = all frames) |

## Example Commands

### Example 1: Demo with Drowsy Subject
```bash
python src/video_demo.py --subject A --session A --class-name Drowsy
```

### Example 2: Demo with Alert Subject
```bash
python src/video_demo.py --subject D --session A --class-name Alert --device cpu
```

### Example 3: Demo with Custom Output
```bash
python src/video_demo.py \
  --subject A \
  --session A \
  --class-name "Low Vigilant" \
  --output results/demo_videos/my_custom_demo.mp4
```

### Example 4: Process Every 5th Frame (for speed)
```bash
python src/video_demo.py \
  --subject A \
  --session A \
  --class-name Drowsy \
  --sample-rate 5
```

## Output Structure

```
results/demo_videos/
├── M5_demo_A_A_Drowsy.mp4              # Annotated video
├── M5_demo_A_A_Drowsy_stats.json       # Per-frame statistics
└── [other demo outputs]
```

## Statistics Output (JSON)

```json
{
  "video_path": "...",
  "total_frames_processed": 1000,
  "fps": 30.0,
  "resolution": {"width": 640, "height": 480},
  "predictions": [
    {"frame": 0, "class_name": "Drowsy", "confidence": 0.95, "smoothed_confidence": 0.93},
    {"frame": 1, "class_name": "Drowsy", "confidence": 0.92, "smoothed_confidence": 0.93},
    ...
  ],
  "class_distribution": {"Drowsy": 600, "Low Vigilant": 300, "Alert": 100},
  "class_percentages": {"Drowsy": 60.0, "Low Vigilant": 30.0, "Alert": 10.0},
  "confidence_stats": {
    "mean": 0.8542,
    "std": 0.1234,
    "min": 0.6001,
    "max": 0.9987
  }
}
```

## Notebook Demo

For interactive exploration, use the Jupyter notebook:

```bash
jupyter notebook notebooks/15-uldd-m5-video-demo.ipynb
```

This notebook:
1. Loads M5 checkpoint
2. Selects a sample from the test set
3. Creates video from frames
4. Runs inference with visualization
5. Generates statistics and plots

## Visual Output

The annotated video includes:

- **Main Prediction**: Large text with class label
  - 🚨 DROWSY (HIGH RISK) — Red border
  - ⚠️ LOW VIGILANT (MEDIUM RISK) — Orange border
  - ✓ ALERT (SAFE) — Green border

- **Confidence Score**: Smoothed confidence (5-frame window)
- **Frame Information**: Frame number and FPS
- **Color-Coded Border**: Indicates predicted class

## Performance Notes

- **Frame Processing**: Real-time on CPU (30+ FPS for 224×224 frames)
- **Memory**: Low memory usage (YOLOv8n is lightweight)
- **Output Video**: MP4 format, compatible with any player
- **Statistics**: JSON format for easy post-processing

## Troubleshooting

**Issue**: Video file not found
```bash
# Check path exists
ls datasets/yolo_frames/A_A/Drowsy/
```

**Issue**: Model checkpoint not found
```bash
# Verify checkpoint exists
ls models/checkpoints/M5_fold0.pt
```

**Issue**: YOLO not available
```bash
# Install ultralytics
pip install ultralytics
```

**Issue**: Slow processing
```bash
# Use sample-rate to skip frames
python src/video_demo.py --subject A --session A --class-name Drowsy --sample-rate 5
```

## For Thesis Presentation

This demo is ideal for:

✅ **Live Demonstration** — Prerecorded videos ensure consistency  
✅ **Visual Results** — See predictions on actual test data  
✅ **Quantitative Metrics** — Confidence scores and class distributions  
✅ **Reproducibility** — Same results every time  
✅ **Flexible Input** — Works with any prerecorded video

Simply run the script or notebook, and present the generated video and statistics!
