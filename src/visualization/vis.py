"""
Visualization and Interpretability Module
Includes training history visualization and Grad-CAM explanations
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple


def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()


class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    
    def __init__(self, model, layer_name):
        """
        Initialize Grad-CAM
        
        Args:
            model: TensorFlow model
            layer_name: Name of the convolutional layer to visualize
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = self._create_grad_model()
    
    def _create_grad_model(self):
        """Create a model that outputs both predictions and gradients"""
        layer = self.model.get_layer(self.layer_name)
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [layer.output, self.model.output]
        )
        return grad_model
    
    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Input image as numpy array (normalized to 0-1)
            pred_index: Index of the prediction class
            
        Returns:
            Heatmap (normalized to 0-1)
        """
        img_array = np.expand_dims(img_array, axis=0)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def visualize_on_image(self, img_array, heatmap, alpha=0.4):
        """
        Overlay Grad-CAM heatmap on the original image
        
        Args:
            img_array: Original input image (0-1 normalized)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay
            
        Returns:
            Image with overlaid heatmap (0-255 range)
        """
        h, w = img_array.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        if img_array.max() <= 1.0:
            img_display = (img_array * 255).astype(np.uint8)
        else:
            img_display = img_array.astype(np.uint8)
        
        if len(img_display.shape) == 2:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        
        overlaid = cv2.addWeighted(img_display, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlaid


class InterpretabilityModule:
    """Complete interpretability analysis for drowsiness model"""
    
    def __init__(self, model, layer_name='conv2d_3'):
        """
        Initialize interpretability module
        
        Args:
            model: Drowsiness detection model
            layer_name: Convolutional layer to visualize
        """
        self.model = model
        self.grad_cam = GradCAM(model, layer_name)
    
    def explain_prediction(self, image_path: str, show_grad_cam: bool = True) -> dict:
        """
        Explain model prediction for an image using Grad-CAM
        
        Args:
            image_path: Path to input image
            show_grad_cam: Whether to generate Grad-CAM visualization
            
        Returns:
            Dictionary with prediction and explanation
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        
        # Get prediction
        img_batch = np.expand_dims(img_normalized, axis=0)
        prediction = self.model.predict(img_batch, verbose=0)[0]
        
        result = {
            'image_path': image_path,
            'prediction': prediction.tolist(),
            'predicted_class': int(np.argmax(prediction)),
            'confidence': float(np.max(prediction))
        }
        
        # Generate Grad-CAM if requested
        if show_grad_cam:
            heatmap = self.grad_cam.generate_heatmap(img_normalized)
            result['heatmap'] = heatmap
            result['visualization'] = self.grad_cam.visualize_on_image(img_normalized, heatmap)
        
        return result
    
    def explain_batch(self, image_paths: list) -> list:
        """
        Explain predictions for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for img_path in image_paths:
            try:
                explanation = self.explain_prediction(img_path)
                explanations.append(explanation)
            except Exception as e:
                print(f"✗ Error explaining {img_path}: {e}")
        
        return explanations


def visualize_important_regions(image, heatmap, class_names=['Awake', 'Drowsy'], output_path=None):
    """
    Create comprehensive visualization showing important regions
    
    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        class_names: List of class names
        output_path: Path to save visualization
        
    Returns:
        Composite visualization image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlaid
    overlaid = cv2.addWeighted(
        (image * 255).astype(np.uint8),
        0.6,
        cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET),
        0.4,
        0
    )
    axes[2].imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Drowsiness-Important Regions')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_path}")
    
    return fig
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def visualize_predictions(images, predictions, true_labels):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f'Pred: {predictions[i]}, True: {true_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()