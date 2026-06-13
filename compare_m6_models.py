"""
Compare M6_lite (vision-only) vs M6_full (multimodal fusion with CAN)
"""

from src.models.m6_train import train_one_fold
import json
from pathlib import Path

print("="*70)
print("M6 ARCHITECTURE COMPARISON")
print("="*70)
print()

print("M6_lite (Vision-only)")
print("-" * 70)
print("  Input modalities:")
print("    ✓ Video embeddings @ 60fps (256-dim from M5 backbone)")
print("    ✗ Facial Action Units (NOT used)")
print("    ✗ CAN/Telemetry signals (NOT used)")
print()
print("  Status: NO alignment with CAN signals")
print("  Previous result: 41% @ 1fps")
print()

print("M6_full (Multimodal Fusion)")
print("-" * 70)
print("  Input modalities:")
print("    ✓ Video embeddings @ 60fps (256-dim from M5 backbone) - NOW ALIGNED!")
print("    ✓ Facial Action Units (AU1-AU45)")
print("    ✓ CAN/Telemetry signals (speed, throttle, steering angle)")
print()
print("  Status: FULLY aligned with CAN signals (temporal fusion)")
print("  Previous result: 41% @ 1fps")
print("  Expected improvement: With 60fps embeddings, should be 50-65%")
print()

print("="*70)
print("TRAINING M6_full WITH 60fps EMBEDDINGS")
print("="*70)
print()

# Train all 5 folds and collect results
results = {}

for fold_idx in range(5):
    print(f"\n--- Fold {fold_idx} (M6_full) ---")
    result = train_one_fold(
        fold_idx=fold_idx,
        variant='full',  # Multimodal fusion
        epochs=10,       # Full training
        batch_size=32,
        verbose=True
    )
    results[fold_idx] = result
    print(f"  Accuracy: {result['accuracy']:.4f}")

# Calculate statistics
accs = [results[i]['accuracy'] for i in range(5)]
avg_acc = sum(accs) / len(accs)

print()
print("="*70)
print("FINAL RESULTS - M6_full (Multimodal + 60fps CAN-aligned)")
print("="*70)
print()
print("Per-fold accuracy:")
for fold_idx in range(5):
    print(f"  Fold {fold_idx}: {results[fold_idx]['accuracy']:.4f}")
print()
print(f"Average accuracy: {avg_acc:.4f}")
print()

print("COMPARISON WITH BASELINE")
print("-" * 70)
print(f"  M6_lite @ 1fps:       41.00% (vision-only)")
print(f"  M6_full @ 1fps:       41.00% (multimodal, NOT 60fps aligned)")
print(f"  M6_full @ 60fps:      {avg_acc:.2%}  (multimodal WITH 60fps alignment)")
print()

improvement = (avg_acc - 0.41) * 100
if improvement > 5:
    print(f"✓ IMPROVEMENT: +{improvement:.1f}% from temporal alignment!")
    print("  The 60fps extraction fixed the CAN signal synchronization issue!")
elif improvement > 0:
    print(f"→ SLIGHT IMPROVEMENT: +{improvement:.1f}%")
    print("  Marginal gain - model may need more data or tuning")
else:
    print(f"⚠ NO IMPROVEMENT: {improvement:.1f}%")
    print("  Issue may be elsewhere (data quality, model architecture, etc)")

print()
print("="*70)
