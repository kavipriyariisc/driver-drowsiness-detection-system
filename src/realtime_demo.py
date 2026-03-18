"""
Real-time Drowsiness Detection Demo
Integrates YOLO face detection, CNN classifier, and progressive scoring
Can run on video files or webcam input
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_detection import FacialAnalyzer
from scoring import DrowsinessScorer, AlertGenerator
from models.architecture import DrowsinessDetectionModel


class RealtimeDrowsinessDetector:
    """Real-time drowsiness detection pipeline"""
    
    def __init__(self, model_path=None, yolo_model='yolov8n.pt'):
        """
        Initialize drowsiness detector
        
        Args:
            model_path: Path to trained drowsiness model
            yolo_model: YOLOv8 model variant
        """
        print("Initializing Real-time Drowsiness Detection System...")
        
        # Load YOLO face analyzer
        self.analyzer = FacialAnalyzer(yolo_model)
        print("✓ YOLO face analyzer loaded")
        
        # Load drowsiness classifier
        if model_path and Path(model_path).exists():
            self.classifier = tf.keras.models.load_model(model_path)
            print(f"✓ Drowsiness classifier loaded from {model_path}")
        else:
            # Use simple CNN model
            self.classifier = DrowsinessDetectionModel('simple_cnn').model
            print("✓ Using default Simple CNN model")
        
        # Initialize scoring system
        self.scorer = DrowsinessScorer(num_levels=5)
        self.alert_gen = AlertGenerator(critical_threshold=80, warning_threshold=60)
        print("✓ Scoring and alert systems initialized\n")
        
        # Tracking variables
        self.frame_count = 0
        self.fps_counter = 0
        self.fps = 0
    
    def process_frame(self, frame):
        """
        Process a single frame for drowsiness detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detections and predictions
        """
        self.frame_count += 1
        
        # Face detection and feature extraction
        analysis = self.analyzer.analyze_frame(frame)
        
        # Prepare visualization
        annotated = frame.copy()
        
        if analysis['faces_detected'] == 0:
            cv2.putText(annotated, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        # Process each detected face
        faces = analysis['faces']
        features_list = analysis['facial_features']
        
        for i, ((x1, y1, x2, y2, conf), facial_features) in enumerate(zip(faces, features_list)):
            # Draw face bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Simulate CAN features (in real scenario, integrate with vehicle data)
            can_features = {
                'speed': np.random.uniform(30, 80),
                'steering': np.random.uniform(-0.2, 0.2),
                'acceleration_magnitude': np.random.uniform(0, 1)
            }
            
            # Calculate drowsiness score
            drowsiness_score = self.scorer.calculate_drowsiness_score(
                facial_features, can_features
            )
            self.scorer.update_history(drowsiness_score)
            
            # Get drowsiness level
            level_idx, level_name = self.scorer.get_drowsiness_level(drowsiness_score)
            
            # Generate alert
            alert = self.alert_gen.generate_alert(drowsiness_score, level_name)
            
            # Visualize on frame
            y_offset = 30 + i * 120
            
            # Face info
            cv2.putText(annotated, f"Face {i+1} - Conf: {conf:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Facial features
            cv2.putText(annotated, f"EAR: {facial_features['avg_ear']:.3f}", (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"MAR: {facial_features['mar']:.3f}", (10, y_offset + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Blink Rate: {facial_features['blink_rate']:.1f}", (10, y_offset + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Drowsiness score and level
            color = self._get_alert_color(alert['severity'])
            cv2.putText(annotated, f"Drowsiness Score: {drowsiness_score:.1f}/100", (10, y_offset + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, f"Level: {level_name}", (10, y_offset + 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Overall stats at top
        cv2.putText(annotated, f"Frame: {self.frame_count} | FPS: {self.fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated, f"Avg Score: {self.scorer.get_average_score():.1f} | Trend: {self.scorer.get_trend()}", 
                   (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return annotated
    
    @staticmethod
    def _get_alert_color(severity):
        """Get color based on alert severity"""
        colors = {
            'none': (0, 255, 0),      # Green
            'normal': (0, 255, 0),    # Green
            'warning': (0, 165, 255), # Orange
            'critical': (0, 0, 255)   # Red
        }
        return colors.get(severity, (200, 200, 200))
    
    def run_video_file(self, video_path, output_path=None):
        """
        Run detection on a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}, FPS: {fps:.1f}, Total frames: {total_frames}")
        
        # Setup output video writer if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated = self.process_frame(frame)
            
            # Write to output if requested
            if out:
                out.write(annotated)
            
            # Display
            cv2.imshow('Real-time Drowsiness Detection', annotated)
            
            # Update FPS
            frame_count += 1
            if frame_count % 30 == 0:
                self.fps = fps
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Video processing complete! Processed {frame_count} frames")
    
    def run_webcam(self):
        """Run detection on webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Cannot access webcam")
            return
        
        print("📷 Starting webcam detection (press 'q' to quit)...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated = self.process_frame(frame)
            
            # Display
            cv2.imshow('Real-time Drowsiness Detection', annotated)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    detector = RealtimeDrowsinessDetector(
        model_path=None,  # Will use default model
        yolo_model='yolov8n.pt'
    )
    
    # Example 1: Run on webcam
    print("Example 1: Running on webcam...")
    # detector.run_webcam()
    
    # Example 2: Run on video file
    print("Example 2: Running on video file...")
    video_path = 'path/to/video.mp4'
    if Path(video_path).exists():
        detector.run_video_file(video_path, output_path='drowsiness_detection_output.mp4')
    else:
        print(f"⚠ Video file not found: {video_path}")


if __name__ == "__main__":
    main()
