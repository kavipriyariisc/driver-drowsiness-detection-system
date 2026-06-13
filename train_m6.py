#!/usr/bin/env python
"""
M6 Training Script - Single Fold or Full 5-Fold CV
Run: python train_m6.py --fold 0 --variant lite --epochs 30
"""
import sys
import argparse
sys.path.insert(0, '.')

from src.models.m6_train import train_one_fold, cross_validate

def main():
    parser = argparse.ArgumentParser(description='Train M6 models')
    parser.add_argument('--fold', type=int, default=None, help='Specific fold (0-4), None=all folds')
    parser.add_argument('--variant', choices=['lite', 'full'], default='lite', help='Model variant')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    print("="*70)
    print(f"M6 Training: variant={args.variant}, epochs={args.epochs}, batch={args.batch_size}")
    print("="*70)
    
    if args.fold is not None:
        # Single fold
        print(f"\nTraining fold {args.fold}...")
        result = train_one_fold(
            fold_idx=args.fold,
            variant=args.variant,
            t_vis=16,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=True
        )
        print(f"\nFold {args.fold} Results:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Macro-F1: {result['macro_f1']:.4f}")
    else:
        # 5-fold CV
        print("\nTraining all 5 folds...")
        results = cross_validate(
            variant=args.variant,
            t_vis=16,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=True
        )
        print(f"\n5-Fold CV Results:")
        print(f"  Mean Accuracy: {results['mean_acc']:.4f}")
        print(f"  Mean Macro-F1: {results['mean_f1']:.4f}")
        print(f"  Per-fold accuracies: {results['fold_accs']}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)

if __name__ == '__main__':
    main()
