from src.models.m6_train import cross_validate
import sys

print("Testing M6_lite with the fixed dataset...")
results = cross_validate(variant='lite', epochs=10)
print('\n' + '='*60)
print('FINAL RESULTS (M6_lite, 10 epochs, 5-fold):')
print('='*60)
print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
print(f"Mean Macro-F1: {results['mean_macro_f1']:.4f}")
print(f"Std Accuracy:  {results['std_accuracy']:.4f}")
print(f"Std Macro-F1:  {results['std_macro_f1']:.4f}")
print('='*60)
