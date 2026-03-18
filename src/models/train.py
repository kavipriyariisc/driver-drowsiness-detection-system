"""
Training Module for Drowsiness Detection Models
Handles data loading, model training, validation, and evaluation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from pathlib import Path
from datetime import datetime
import json
import cv2
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.architecture import DrowsinessDetectionModel


class DrowsinessModelTrainer:
    """Trainer class for drowsiness detection models"""
    
    def __init__(self, model, model_name='drowsiness_model'):
        """
        Initialize model trainer
        
        Args:
            model: Compiled TensorFlow model
            model_name: Name for saving checkpoints and logs
        """
        self.model = model
        self.model_name = model_name
        self.history = None
        
        # Create directories
        self.checkpoint_dir = 'models/checkpoints'
        self.logs_dir = 'results/logs'
        self.results_dir = 'results/experiments'
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def get_callbacks(self):
        """Get training callbacks"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                os.path.join(self.checkpoint_dir, f'{self.model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(self.logs_dir, f'{self.model_name}_{timestamp}'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def train(self, train_data, val_data, epochs=100, batch_size=32):
        """
        Train the model
        
        Args:
            train_data: Training data (X_train, y_train)
            val_data: Validation data (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}\n")
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data (X_test, y_test)
            
        Returns:
            Evaluation metrics dictionary
        """
        X_test, y_test = test_data
        
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*60}\n")
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Get predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        # Format results
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]) if len(results) > 2 else None,
            'recall': float(results[3]) if len(results) > 3 else None
        }
        
        print(f"\nTest Results:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath=None):
        """
        Save trained model
        
        Args:
            filepath: Path to save model (optional)
        """
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, f'{self.model_name}_final.h5')
        
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
        
        return filepath
    
    def save_training_summary(self):
        """Save training summary to JSON"""
        if self.history is None:
            print("No training history to save")
            return
        
        summary = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'final_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'best_val_accuracy': float(np.max(self.history.history['val_accuracy'])),
            'history': {key: [float(v) for v in val] for key, val in self.history.history.items()}
        }
        
        summary_path = os.path.join(self.results_dir, f'{self.model_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Training summary saved to {summary_path}")
        
        return summary_path


def load_and_preprocess_data(data_dir, validation_split=0.2, test_split=0.1):
    """
    Load and preprocess dataset
    
    Args:
        data_dir: Directory containing images
        validation_split: Fraction for validation set
        test_split: Fraction for test set
        
    Returns:
        Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    images = []
    labels = []
    
    print(f"Loading images from {data_dir}...")
    
    # Load images from subdirectories (e.g., awake/, drowsy/)
    class_names = os.listdir(data_dir)
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0  # Normalize
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"  ⚠ Error loading {img_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✓ Loaded {len(X)} images\n")
    
    # Split into train, val, test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )
    
    val_split_adjusted = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split_adjusted, random_state=42
    )
    
    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(epochs=50, batch_size=32, learning_rate=0.001):
    """Legacy function for backward compatibility"""
    print("Training drowsiness detection model...")
    
    try:
        train_data, val_data, test_data = load_and_preprocess_data(
            'datasets/processed',
            validation_split=0.15,
            test_split=0.15
        )
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Create model
    model_wrapper = DrowsinessDetectionModel(
        model_type='simple_cnn',
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Train
    trainer = DrowsinessModelTrainer(model_wrapper.model, 'drowsiness_detector')
    history = trainer.train(train_data, val_data, epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    trainer.evaluate(test_data)
    
    # Save
    trainer.save_model()
    trainer.save_training_summary()
    
    return history


if __name__ == "__main__":
    train_model()