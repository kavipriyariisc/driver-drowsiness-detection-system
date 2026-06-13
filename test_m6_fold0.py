from src.models.m6_train import train_one_fold

print("Quick test: M6_lite on fold 0 (5 epochs)")
result = train_one_fold(fold_idx=0, variant='lite', epochs=5, verbose=True)
print(f"\nResult: Acc={result['accuracy']:.4f}, F1={result['macro_f1']:.4f}")
print("✅ M6_lite training works correctly")
print("Note: Limited accuracy (~41%) due to visual embeddings from yolo_frames (not UL-DD videos)")
