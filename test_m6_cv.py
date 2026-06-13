from src.models.m6_train import cross_validate
import json

print("Running M6_lite 5-fold cross-validation (10 epochs per fold)...")
print("="*70)
results = cross_validate(variant='lite', epochs=10)

print('\n' + '='*70)
print('FINAL RESULTS (M6_lite, 10 epochs, 5-fold CV):')
print('='*70)
for i, acc in enumerate(results['accuracies']):
    f1 = results['macro_f1s'][i]
    print(f"Fold {i}: Acc={acc:.4f}, F1={f1:.4f}")

print('-'*70)
print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
print(f"Mean Macro-F1: {results['mean_macro_f1']:.4f} ± {results['std_macro_f1']:.4f}")
print('='*70)

# Save results
out_path = 'results/reports/M6_lite_5fold.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
