# M6 Extraction in Colab (No Google Drive Required)
# ================================================
# Run this notebook in VS Code's Colab integration
# or copy-paste cells into Colab directly

# ============================================================================
# CELL 1: Setup & Install Dependencies
# ============================================================================

import subprocess
subprocess.run(['pip', 'install', 'ultralytics', 'opencv-python', 'torch', 'numpy', 'pandas', '-q'])
print("✓ Dependencies installed")

# ============================================================================
# CELL 2: Clone or Download Project Code
# ============================================================================

# Option A: If code is in a Git repo
import subprocess
subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/driver-drowsiness-detection.git'])
import os
os.chdir('driver-drowsiness-detection-system')

# Option B: If uploading manually (skip this if using git)
# Upload src/models/m6_extractor_from_video.py manually first
# Then set path below

print("✓ Project code ready")

# ============================================================================
# CELL 3: Download/Mount Video Data
# ============================================================================

# ⚠️ Choose ONE option:

# --- OPTION 1: Mount Local Folder via Colab (If on same network)
# This works if your machine and Colab are on same network
# In Terminal: gcloud compute ssh instance-name -- -fNL 6006:localhost:6006
# Then: /content/mount_local (need Colab to support it)

# --- OPTION 2: Download from Cloud Storage (Fastest)
# If videos are on Google Drive (but NOT your main drive)
from google.colab import auth
auth.authenticate_user()

import gdown
# Download specific video zip from Google Drive
video_zip_id = "YOUR_FILE_ID"  # Get from Drive share link
gdown.download(f'https://drive.google.com/uc?id={video_zip_id}', 'videos.zip', quiet=False)

import zipfile
with zipfile.ZipFile('videos.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/ul_dd_videos')

print("✓ Videos downloaded")

# --- OPTION 3: Upload Small Subset (For Testing)
from google.colab import files
print("Upload a ZIP file with sample videos (A/, C/, E/ folders only)")
uploaded = files.upload()
for filename in uploaded.keys():
    import zipfile
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/videos')
print("✓ Videos uploaded")

# ============================================================================
# CELL 4: Set Paths (Configure Based on Your Setup)
# ============================================================================

import sys
from pathlib import Path

# If using git clone
PROJECT_ROOT = Path('/content/driver-drowsiness-detection-system')
sys.path.insert(0, str(PROJECT_ROOT))

# Video paths (choose based on upload method)
VIDEO_ROOT = Path('/content/ul_dd_videos')  # Option 2
# VIDEO_ROOT = Path('/content/videos')       # Option 3

# M5 checkpoint (must upload this separately - it's only ~50 MB)
CKPT_PATH = PROJECT_ROOT / 'models' / 'checkpoints' / 'M5_fold0.pt'

# Output path in Colab session (not persistent unless you download)
OUTPUT_DIR = Path('/content/embeddings_uldd')

print(f"Project root: {PROJECT_ROOT}")
print(f"Video root: {VIDEO_ROOT}")
print(f"Checkpoint: {CKPT_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# Verify paths exist
print(f"\n✓ Project exists: {PROJECT_ROOT.exists()}")
print(f"✓ Videos exist: {VIDEO_ROOT.exists()}")
print(f"✓ Checkpoint exists: {CKPT_PATH.exists()}")

# ============================================================================
# CELL 5: Import & Configure Extractor
# ============================================================================

from src.models.m6_extractor_from_video import extract_all_from_videos

# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 6: Run Extraction (MAIN CELL)
# ============================================================================

# Option A: Full extraction (all subjects)
print("Starting full extraction...")
extract_all_from_videos(
    ckpt_path=CKPT_PATH,
    video_root=VIDEO_ROOT,
    out_dir=OUTPUT_DIR,
    fps=60.0,
    batch_size=64,  # Increase if GPU has memory
    num_workers=2,
    overwrite=False,
)

# Option B: Test with subset (faster, for verification)
print("Starting subset extraction (test)...")
extract_all_from_videos(
    ckpt_path=CKPT_PATH,
    video_root=VIDEO_ROOT,
    out_dir=OUTPUT_DIR,
    fps=60.0,
    sessions=['A_A', 'C_A', 'E_A'],  # Only 3 sessions
    batch_size=64,
    num_workers=2,
    overwrite=False,
)

print("\n✓ Extraction complete!")

# ============================================================================
# CELL 7: Verify Results
# ============================================================================

import numpy as np
from pathlib import Path

output_files = list(OUTPUT_DIR.glob('*.npz'))
print(f"Created {len(output_files)} embedding files:")

total_frames = 0
total_size = 0
for f in sorted(output_files)[:5]:  # Show first 5
    size_mb = f.stat().st_size / 1e6
    data = np.load(f, allow_pickle=True)
    n_frames = data['embedding'].shape[0]
    print(f"  {f.name}: {n_frames:,} frames ({size_mb:.1f} MB)")
    total_frames += n_frames
    total_size += f.stat().st_size

if len(output_files) > 5:
    print(f"  ... and {len(output_files) - 5} more")

print(f"\nTotal: {total_frames:,} frames across {len(output_files)} sessions")
print(f"Total size: {total_size/1e9:.2f} GB")

# ============================================================================
# CELL 8: Download Results (Important!)
# ============================================================================

# Results are ONLY in Colab session, will be deleted when session ends
# You MUST download them to keep

import shutil

# Option A: Download as ZIP
print("Creating ZIP of results...")
shutil.make_archive('embeddings_uldd', 'zip', OUTPUT_DIR)
files.download('embeddings_uldd.zip')
print("✓ Downloaded: embeddings_uldd.zip")

# Option B: Copy to Google Drive (optional backup)
# Mount Drive first in CELL 2, then:
# shutil.copytree(OUTPUT_DIR, '/content/drive/MyDrive/embeddings_uldd')

print("\n✅ All done! Move downloaded files to: models/embeddings_uldd/")

# ============================================================================
# CELL 9: Comparison with Old Embeddings
# ============================================================================

# If you also have old 1 fps embeddings locally, compare:
import numpy as np

old_emb_path = PROJECT_ROOT / 'models' / 'embeddings' / 'A_A.npz'
new_emb_path = OUTPUT_DIR / 'A_A.npz'

if old_emb_path.exists() and new_emb_path.exists():
    old = np.load(old_emb_path, allow_pickle=True)
    new = np.load(new_emb_path, allow_pickle=True)
    
    print("COMPARISON: Old (1fps) vs New (60fps)")
    print(f"Old: {old['embedding'].shape[0]:,} frames")
    print(f"New: {new['embedding'].shape[0]:,} frames")
    print(f"Improvement: {new['embedding'].shape[0] / old['embedding'].shape[0]:.0f}x")
