"""
Face Detection and ROI Extraction Module
Uses YOLOv8 for real-time face, eye, and mouth detection
Computes facial features like Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


class FaceDetector:
    """YOLO-based face detector"""
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize YOLO face detector
        
        Args:
            model_name: YOLOv8 model variant (nano, small, medium, large)
        """
        self.model = YOLO(model_name)
        print(f"✓ Loaded YOLOv8 model: {model_name}")
    
    def detect_faces(self, image, conf_threshold=0.45):
        """
        Detect faces in an image using YOLOv8
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detected face boxes [(x1, y1, x2, y2, conf), ...]
        """
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        faces = []
        if results and len(results) > 0:
            for detection in results[0].boxes:
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                conf = detection.conf[0].cpu().item()
                faces.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return faces
    
    def draw_detections(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            faces: List of face boxes
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn boxes
        """
        img_with_boxes = image.copy()
        
        for x1, y1, x2, y2, conf in faces:
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(img_with_boxes, f'Face {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img_with_boxes


class ROIExtractor:
    """Extract Regions of Interest (eyes, mouth) for drowsiness features"""
    
    @staticmethod
    def extract_eye_region(face_image, padding=0.2):
        """
        Extract eye region from face image
        Assumes standard face orientation
        
        Args:
            face_image: Cropped face image
            padding: Padding around detected region (as fraction)
            
        Returns:
            Left eye, Right eye, and their bounding boxes
        """
        h, w = face_image.shape[:2]
        
        # Eye region typically in upper half of face
        eye_y1 = int(h * 0.15)
        eye_y2 = int(h * 0.4)
        
        # Left eye (image's right side)
        left_eye_x1 = int(w * 0.05)
        left_eye_x2 = int(w * 0.45)
        left_eye = face_image[eye_y1:eye_y2, left_eye_x1:left_eye_x2]
        
        # Right eye (image's left side)
        right_eye_x1 = int(w * 0.55)
        right_eye_x2 = int(w * 0.95)
        right_eye = face_image[eye_y1:eye_y2, right_eye_x1:right_eye_x2]
        
        return left_eye, right_eye, {
            'left_eye': (eye_y1, eye_y2, left_eye_x1, left_eye_x2),
            'right_eye': (eye_y1, eye_y2, right_eye_x1, right_eye_x2)
        }
    
    @staticmethod
    def extract_mouth_region(face_image):
        """
        Extract mouth region from face image
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Mouth region image and bounding box
        """
        h, w = face_image.shape[:2]
        
        # Mouth region typically in lower half of face
        mouth_y1 = int(h * 0.55)
        mouth_y2 = int(h * 0.85)
        mouth_x1 = int(w * 0.15)
        mouth_x2 = int(w * 0.85)
        
        mouth = face_image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        
        return mouth, (mouth_y1, mouth_y2, mouth_x1, mouth_x2)


class FacialFeatureCalculator:
    """Calculate facial features for drowsiness detection"""
    
    @staticmethod
    def eye_aspect_ratio(eye_image):
        """
        Calculate Eye Aspect Ratio (EAR)
        Lower EAR indicates closed or drooping eyes (drowsiness indicator)
        
        Args:
            eye_image: Single eye region image
            
        Returns:
            EAR value (float)
        """
        try:
            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours (pupil/iris)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return 0.0
            
            # Get largest contour (pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate circularity: 4π * area / perimeter²
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0:
                return 0.0
            
            ear = 4 * np.pi * (area / (perimeter ** 2))
            return round(ear, 4)
        
        except Exception as e:
            print(f"⚠ Error calculating EAR: {e}")
            return 0.0
    
    @staticmethod
    def mouth_aspect_ratio(mouth_image):
        """
        Calculate Mouth Aspect Ratio (MAR)
        Higher MAR indicates open mouth (yawning indicator)
        
        Args:
            mouth_image: Mouth region image
            
        Returns:
            MAR value (float)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2GRAY)
            
            # Apply binary threshold
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return 0.0
            
            # Get largest contour (mouth opening)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            if w == 0:
                return 0.0
            
            mar = h / w
            return round(mar, 4)
        
        except Exception as e:
            print(f"⚠ Error calculating MAR: {e}")
            return 0.0
    
    @staticmethod
    def blink_rate(eye_history, threshold=0.3, window_size=5):
        """
        Calculate blink rate from EAR history
        Blink detected when EAR drops below threshold
        
        Args:
            eye_history: List of recent EAR values
            threshold: EAR threshold for closed eyes
            window_size: Number of frames to consider
            
        Returns:
            Blink rate (blinks per second, assuming 30 FPS)
        """
        if len(eye_history) < window_size:
            return 0.0
        
        blinks = 0
        for i in range(len(eye_history) - 1):
            # Detect transition from open to closed
            if eye_history[i] > threshold and eye_history[i + 1] <= threshold:
                blinks += 1
        
        # Approximate FPS = 30
        blink_rate = blinks / (window_size / 30.0)
        return round(blink_rate, 2)


class FacialAnalyzer:
    """Complete facial analysis pipeline for drowsiness detection"""
    
    def __init__(self, yolo_model='yolov8n.pt'):
        """
        Initialize facial analyzer
        
        Args:
            yolo_model: YOLOv8 model variant
        """
        self.face_detector = FaceDetector(yolo_model)
        self.roi_extractor = ROIExtractor()
        self.feature_calc = FacialFeatureCalculator()
        
        # Buffers for temporal features
        self.ear_history = []
        self.mar_history = []
        self.max_history_size = 30  # ~1 second at 30 FPS
    
    def analyze_frame(self, frame):
        """
        Complete analysis of a single frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'faces_detected': 0,
            'faces': [],
            'facial_features': []
        }
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        results['faces_detected'] = len(faces)
        results['faces'] = faces
        
        # Extract features for each face
        for x1, y1, x2, y2, conf in faces:
            face_image = frame[y1:y2, x1:x2]
            
            # Extract ROIs
            left_eye, right_eye, eye_boxes = self.roi_extractor.extract_eye_region(face_image)
            mouth, mouth_box = self.roi_extractor.extract_mouth_region(face_image)
            
            # Calculate features
            left_ear = self.feature_calc.eye_aspect_ratio(left_eye)
            right_ear = self.feature_calc.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = self.feature_calc.mouth_aspect_ratio(mouth)
            
            # Update history
            self.ear_history.append(avg_ear)
            self.mar_history.append(mar)
            
            if len(self.ear_history) > self.max_history_size:
                self.ear_history.pop(0)
            if len(self.mar_history) > self.max_history_size:
                self.mar_history.pop(0)
            
            # Calculate blink rate
            blink_rate = self.feature_calc.blink_rate(self.ear_history)
            
            features = {
                'left_ear': left_ear,
                'right_ear': right_ear,
                'avg_ear': avg_ear,
                'mar': mar,
                'blink_rate': blink_rate
            }
            
            results['facial_features'].append(features)
        
        return results
    
    def visualize_analysis(self, frame, analysis_results):
        """
        Create visualization with detected faces and features
        
        Args:
            frame: Input frame
            analysis_results: Results from analyze_frame()
            
        Returns:
            Annotated frame
        """
        annotated = self.face_detector.draw_detections(frame, analysis_results['faces'])
        
        # Add feature text
        for i, features in enumerate(analysis_results['facial_features']):
            y_offset = 30 + i * 60
            cv2.putText(annotated, f"Face {i+1}:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(annotated, f"EAR: {features['avg_ear']:.3f}", (10, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"MAR: {features['mar']:.3f}", (10, y_offset + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Blink Rate: {features['blink_rate']:.1f} blinks/s", (10, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated


def main():
    """Example usage of FacialAnalyzer"""
    analyzer = FacialAnalyzer(yolo_model='yolov8n.pt')
    
    # Example with a test image
    test_image_path = 'path/to/test/image.jpg'
    
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        results = analyzer.analyze_frame(frame)
        
        print(f"Detected {results['faces_detected']} face(s)")
        for i, features in enumerate(results['facial_features']):
            print(f"  Face {i+1}:")
            print(f"    EAR: {features['avg_ear']:.4f}")
            print(f"    MAR: {features['mar']:.4f}")
            print(f"    Blink Rate: {features['blink_rate']:.2f} blinks/s")
        
        # Visualize
        annotated = analyzer.visualize_analysis(frame, results)
        cv2.imwrite('test_output.jpg', annotated)
        print("✓ Test output saved to test_output.jpg")
    else:
        print(f"✗ Test image not found: {test_image_path}")


if __name__ == "__main__":
    main()
