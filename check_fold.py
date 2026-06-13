import numpy as np
fold = np.load('datasets/processed/ul_dd/fold_0.npz', allow_pickle=False)
print('Keys in fold_0.npz:')
for key in sorted(fold.files):
    val = fold[key]
    shape_str = str(val.shape) if hasattr(val, "shape") else str(len(val))
    print(f'  {key}: shape={shape_str}, dtype={val.dtype}')
    if len(val) > 0 and key not in ['X_mm', 'y_mm']:
        first_vals = val[:min(3, len(val))]
        print(f'    First few values: {first_vals}')
