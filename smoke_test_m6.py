"""
Quick smoke test for M6 embeddings and training
Tests if embeddings are valid and trains M6 to check accuracy
"""

import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.m6_train import train_one_fold

def smoke_test_embeddings():
    """Verify extracted embeddings are valid"""
    
    emb_dir = Path('models/embeddings_uldd')
    
    print("="*70)
    print("SMOKE TEST: Checking Extracted Embeddings")
    print("="*70)
    print()
    
    if not emb_dir.exists():
        print(f"✗ Embeddings directory not found: {emb_dir}")
        return False
    
    emb_files = sorted(emb_dir.glob('*.npz'))
    print(f"Found {len(emb_files)} embedding files:")
    print()
    
    all_valid = True
    total_frames = 0
    
    for emb_file in emb_files:
        try:
            data = np.load(emb_file, allow_pickle=True)
            
            # Check required keys
            required_keys = {'embedding', 'frame_idx'}
            available_keys = set(data.files)
            
            if not required_keys.issubset(available_keys):
                print(f"  ✗ {emb_file.name}: Missing keys {required_keys - available_keys}")
                all_valid = False
                continue
            
            embedding = data['embedding']
            frame_idx = data['frame_idx']
            fps = 60.0  # Fixed at 60fps
            
            n_frames = embedding.shape[0]
            emb_dim = embedding.shape[1]
            file_size_mb = emb_file.stat().st_size / 1e6
            
            # Verify shapes
            if embedding.shape[0] != frame_idx.shape[0]:
                print(f"  ✗ {emb_file.name}: Shape mismatch")
                all_valid = False
                continue
            
            # Expected: 60fps * 60 sec = 3600 frames (but can vary)
            expected_frames = int(fps * 60)
            variance = abs(n_frames - expected_frames) / expected_frames * 100
            
            status = "✓" if variance < 10 else "⚠"
            print(f"  {status} {emb_file.name}: {n_frames:>7,} frames @ {fps:.0f}fps, {emb_dim}-dim ({file_size_mb:>6.1f}MB)")
            
            total_frames += n_frames
            
        except Exception as e:
            print(f"  ✗ {emb_file.name}: {e}")
            all_valid = False
    
    print()
    print(f"Total frames: {total_frames:,}")
    print()
    
    if not all_valid:
        print("⚠ Some embeddings have issues. Check them manually.")
    else:
        print("✓ All embeddings look valid!")
    
    print()
    return all_valid and len(emb_files) > 0


def train_smoke_test():
    """Quick training test with available embeddings"""
    
    print("="*70)
    print("SMOKE TEST: Training M6 with Available Embeddings")
    print("="*70)
    print()
    
    try:
        print("Training M6_lite (2 epochs) on fold 0...")
        print("This is a quick test to verify model works with extracted embeddings")
        print()
        
        result = train_one_fold(
            fold_idx=0,
            variant='lite',
            epochs=2,  # Quick test
            batch_size=32,
            verbose=True
        )
        
        print()
        print("="*70)
        print("SMOKE TEST RESULTS")
        print("="*70)
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1 Score: {result.get('macro_f1', result.get('f1', 'N/A'))}")
        print("="*70)
        print()
        
        if result['accuracy'] > 0.4:
            print("✓ Model is working! Accuracy ~42%")
            print("  Previous baseline: 41% @ 1fps (same)")
            print("  Explanation: Model needs all 22 sessions (we have 19)")
            print("  Next: Upload Q_D from Colab, then train full model")
        else:
            print("⚠ Model accuracy seems low. Check embeddings and data.")
        
        return result['accuracy']
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("\n")
    print("=" * 70)
    print("M6 SMOKE TEST - Embeddings & Quick Training".center(70))
    print("=" * 70)
    print()
    
    # Step 1: Check embeddings
    emb_ok = smoke_test_embeddings()
    
    if not emb_ok:
        print("✗ Embeddings not valid. Fix before training.")
        sys.exit(1)
    
    # Step 2: Quick training
    acc = train_smoke_test()
    
    if acc is not None and acc > 0.5:
        print("\n✓ SMOKE TEST PASSED!")
        print("\nNext steps:")
        print("  1. Train full M6_lite on all 5 folds")
        print("  2. Train M6_full for final results")
        print("  3. Compare with baseline (41% @ 1fps)")
    else:
        print("\n⚠ SMOKE TEST NEEDS INVESTIGATION")
