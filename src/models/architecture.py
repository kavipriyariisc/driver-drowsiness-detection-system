"""
Model Architecture for Driver Drowsiness Detection
Includes:
- Simple CNN (baseline)
- Hybrid CNN-LSTM (temporal feature fusion)
- Vision Transformer (ViT) variant for better performance
- Multimodal fusion model (vision + CAN signals)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class SimpleCNNModel:
    """Baseline CNN model for drowsiness classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize simple CNN model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (2 for binary, 4 for multi-state)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build the CNN model"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        # Compile
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class CNNLSTMModel:
    """Hybrid CNN-LSTM model for temporal feature fusion"""
    
    def __init__(self, input_shape=(None, 224, 224, 3), num_classes=2):
        """
        Initialize CNN-LSTM model for video sequences
        
        Args:
            input_shape: Input video sequence shape (frames, height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build the CNN-LSTM model"""
        model = models.Sequential([
            # TimeDistributed CNN to process each frame
            layers.TimeDistributed(
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                input_shape=self.input_shape
            ),
            layers.TimeDistributed(layers.BatchNormalization()),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            
            layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
            layers.TimeDistributed(layers.BatchNormalization()),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            
            layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
            layers.TimeDistributed(layers.BatchNormalization()),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            
            layers.TimeDistributed(layers.Flatten()),
            
            # LSTM for temporal feature fusion
            layers.LSTM(256, activation='relu', return_sequences=True),
            layers.Dropout(0.5),
            
            layers.LSTM(128, activation='relu', return_sequences=False),
            layers.Dropout(0.5),
            
            # Fully connected layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        # Compile
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class MultimodalFusionModel:
    """Multimodal model fusing vision features (CNN-LSTM) with CAN signals"""
    
    def __init__(self, image_shape=(224, 224, 3), can_feature_size=6, num_classes=2):
        """
        Initialize multimodal fusion model
        
        Args:
            image_shape: Input image shape
            can_feature_size: Number of CAN signal features (speed, steer, throttle, brake, accel_x, accel_y)
            num_classes: Number of output classes
        """
        self.image_shape = image_shape
        self.can_feature_size = can_feature_size
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build multimodal fusion model with late fusion"""
        
        # Vision pathway (CNN)
        image_input = layers.Input(shape=self.image_shape, name='image_input')
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        vision_features = layers.Dense(256, activation='relu', name='vision_features')(x)
        vision_features = layers.Dropout(0.5)(vision_features)
        
        # CAN signal pathway (MLP)
        can_input = layers.Input(shape=(self.can_feature_size,), name='can_input')
        
        y = layers.Dense(128, activation='relu')(can_input)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.3)(y)
        
        y = layers.Dense(64, activation='relu')(y)
        y = layers.BatchNormalization()(y)
        can_features = layers.Dropout(0.3)(y)
        
        # Fusion (concatenate both pathways)
        fused = layers.Concatenate(name='fusion')([vision_features, can_features])
        
        # Fusion layers
        fused = layers.Dense(256, activation='relu')(fused)
        fused = layers.BatchNormalization()(fused)
        fused = layers.Dropout(0.5)(fused)
        
        fused = layers.Dense(128, activation='relu')(fused)
        fused = layers.BatchNormalization()(fused)
        fused = layers.Dropout(0.5)(fused)
        
        # Output layer
        output = layers.Dense(self.num_classes, 
                             activation='softmax' if self.num_classes > 2 else 'sigmoid',
                             name='output')(fused)
        
        # Create model
        model = models.Model(inputs=[image_input, can_input], outputs=output)
        
        # Compile
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class AttentionBasedFusionModel:
    """Attention-based model for multimodal fusion with interpretability"""
    
    def __init__(self, image_shape=(224, 224, 3), can_feature_size=6, num_classes=2):
        """
        Initialize attention-based fusion model
        
        Args:
            image_shape: Input image shape
            can_feature_size: Number of CAN signal features
            num_classes: Number of output classes
        """
        self.image_shape = image_shape
        self.can_feature_size = can_feature_size
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build attention-based fusion model"""
        
        # Vision pathway
        image_input = layers.Input(shape=self.image_shape, name='image_input')
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        
        vision_features = layers.Dense(256, activation='relu')(x)
        vision_features = layers.Dropout(0.5)(vision_features)
        
        # CAN signal pathway
        can_input = layers.Input(shape=(self.can_feature_size,), name='can_input')
        
        y = layers.Dense(128, activation='relu')(can_input)
        y = layers.BatchNormalization()(y)
        can_features = layers.Dropout(0.3)(y)
        
        # Attention mechanism (learns to weight importance of each modality)
        attention_weights = layers.Dense(2, activation='softmax', name='attention_weights')(
            layers.Concatenate()([vision_features, can_features])
        )
        
        # Apply attention to each modality
        vision_weighted = layers.Multiply(name='vision_weighted')(
            [vision_features, layers.Lambda(lambda x: x[:, 0:1])(attention_weights)]
        )
        can_weighted = layers.Multiply(name='can_weighted')(
            [can_features, layers.Lambda(lambda x: x[:, 1:2])(attention_weights)]
        )
        
        # Fusion
        fused = layers.Concatenate()([vision_weighted, can_weighted])
        
        # Classification layers
        fused = layers.Dense(128, activation='relu')(fused)
        fused = layers.BatchNormalization()(fused)
        fused = layers.Dropout(0.5)(fused)
        
        output = layers.Dense(self.num_classes,
                             activation='softmax' if self.num_classes > 2 else 'sigmoid',
                             name='output')(fused)
        
        # Create model
        model = models.Model(inputs=[image_input, can_input], outputs=output)
        
        # Compile
        loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class DrowsinessDetectionModel:
    """Main class for drowsiness detection model (for backward compatibility)"""
    
    def __init__(self, model_type='simple_cnn', input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize drowsiness detection model
        
        Args:
            model_type: 'simple_cnn', 'cnn_lstm', 'multimodal', or 'attention'
            input_shape: Input shape
            num_classes: Number of output classes
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        if model_type == 'simple_cnn':
            self.model_wrapper = SimpleCNNModel(input_shape, num_classes)
        elif model_type == 'cnn_lstm':
            self.model_wrapper = CNNLSTMModel((None, *input_shape), num_classes)
        elif model_type == 'multimodal':
            self.model_wrapper = MultimodalFusionModel(input_shape, can_feature_size=6, num_classes=num_classes)
        elif model_type == 'attention':
            self.model_wrapper = AttentionBasedFusionModel(input_shape, can_feature_size=6, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model_wrapper.model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def build_model(self):
        """Build the model"""
        return self.model