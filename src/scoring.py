"""
Progressive Drowsiness Scoring System
Combines multimodal features (vision + CAN) into a comprehensive drowsiness score
Outputs multiple drowsiness levels for nuanced detection
"""

import numpy as np
from typing import Dict, List, Tuple


class DrowsinessScorer:
    """Calculate and progressively monitor drowsiness levels"""
    
    def __init__(self, num_levels=5):
        """
        Initialize drowsiness scorer
        
        Args:
            num_levels: Number of drowsiness levels (e.g., 5 = Alert, Mild, Moderate, Severe, Critical)
        """
        self.num_levels = num_levels
        self.score_history = []
        self.max_history = 30  # ~1 second buffer at 30 FPS
        
        # Define thresholds for each level
        self.level_thresholds = self._create_thresholds()
        
        # Feature weights
        self.feature_weights = {
            'ear': 0.3,      # Eye Aspect Ratio
            'mar': 0.1,      # Mouth Aspect Ratio
            'blink_rate': 0.15,  # Blink rate
            'steering': 0.15,    # Steering instability
            'speed': 0.1,        # Speed changes
            'acceleration': 0.1  # Acceleration changes
        }
    
    def _create_thresholds(self) -> List[float]:
        """
        Create drowsiness level thresholds
        
        Returns:
            List of thresholds for each level
        """
        return np.linspace(0, 100, self.num_levels + 1).tolist()
    
    def calculate_drowsiness_score(self, facial_features: Dict, can_features: Dict) -> float:
        """
        Calculate drowsiness score from facial and vehicle features
        Score ranges from 0-100 (0 = Alert, 100 = Critical Drowsiness)
        
        Args:
            facial_features: Dict with keys: 'ear', 'mar', 'blink_rate'
            can_features: Dict with keys: 'speed', 'steering', 'acceleration'
            
        Returns:
            Drowsiness score (0-100)
        """
        scores = {}
        
        # Vision-based scores
        # Low EAR indicates drooping/closed eyes (drowsiness)
        ear = facial_features.get('avg_ear', 0.4)
        scores['ear'] = self._ear_to_score(ear)
        
        # High MAR indicates yawning (drowsiness indicator)
        mar = facial_features.get('mar', 0.3)
        scores['mar'] = self._mar_to_score(mar)
        
        # Low blink rate indicates reduced alertness
        blink_rate = facial_features.get('blink_rate', 20)
        scores['blink_rate'] = self._blink_rate_to_score(blink_rate)
        
        # Vehicle behavior scores
        # Erratic steering indicates loss of control
        steering = can_features.get('steering', 0.0)
        scores['steering'] = self._steering_to_score(steering)
        
        # Speed fluctuations indicate inconsistency
        speed = can_features.get('speed', 0.0)
        scores['speed'] = self._speed_to_score(speed)
        
        # High acceleration changes indicate poor control
        acceleration = can_features.get('acceleration_magnitude', 0.0)
        scores['acceleration'] = self._acceleration_to_score(acceleration)
        
        # Weighted combination
        drowsiness_score = sum(
            scores[feature] * self.feature_weights[feature]
            for feature in self.feature_weights.keys()
        ) / sum(self.feature_weights.values())
        
        return round(drowsiness_score, 2)
    
    @staticmethod
    def _ear_to_score(ear: float) -> float:
        """
        Convert EAR (Eye Aspect Ratio) to drowsiness score
        
        Args:
            ear: Eye Aspect Ratio value
            
        Returns:
            Score (0-100)
        """
        # Normal EAR range: 0.4-0.5 (alert)
        # Low EAR < 0.2 indicates drooping/closed eyes
        if ear > 0.4:
            return 0.0  # Alert
        elif ear > 0.3:
            return 30.0  # Mild drowsiness
        elif ear > 0.2:
            return 60.0  # Moderate drowsiness
        else:
            return 100.0  # Severe drowsiness
    
    @staticmethod
    def _mar_to_score(mar: float) -> float:
        """
        Convert MAR (Mouth Aspect Ratio) to drowsiness score
        
        Args:
            mar: Mouth Aspect Ratio value
            
        Returns:
            Score (0-100)
        """
        # Normal MAR: ~0.3-0.4
        # High MAR (> 0.6) indicates yawning
        if mar < 0.4:
            return 0.0  # Not yawning
        elif mar < 0.6:
            return 40.0  # Slight yawn
        else:
            return 100.0  # Full yawn
    
    @staticmethod
    def _blink_rate_to_score(blink_rate: float) -> float:
        """
        Convert blink rate to drowsiness score
        
        Args:
            blink_rate: Blinks per second
            
        Returns:
            Score (0-100)
        """
        # Normal blink rate: 15-20 blinks/minute (~0.25-0.33 per second)
        # Reduced blink rate indicates drowsiness
        if blink_rate > 0.25:
            return 0.0  # Normal/Alert
        elif blink_rate > 0.15:
            return 30.0  # Mild reduction
        elif blink_rate > 0.05:
            return 60.0  # Significant reduction
        else:
            return 100.0  # Critical reduction
    
    @staticmethod
    def _steering_to_score(steering: float) -> float:
        """
        Convert steering variance to drowsiness score
        
        Args:
            steering: Steering angle (-1.0 to 1.0)
            
        Returns:
            Score (0-100)
        """
        # Erratic steering (sudden changes) indicates drowsiness
        # Steady steering < 0.1 is normal
        abs_steer = abs(steering)
        
        if abs_steer < 0.1:
            return 0.0
        elif abs_steer < 0.3:
            return 20.0
        elif abs_steer < 0.5:
            return 50.0
        else:
            return 100.0
    
    @staticmethod
    def _speed_to_score(speed: float) -> float:
        """
        Convert speed stability to drowsiness score
        
        Args:
            speed: Vehicle speed in km/h
            
        Returns:
            Score (0-100)
        """
        # Speed anomalies (too slow or too fast variation) indicate drowsiness
        # Assume normal city driving: 30-60 km/h
        # If speed is too low or unstable, increase score
        if 30 <= speed <= 60:
            return 0.0  # Normal speed
        elif speed < 20 or speed > 80:
            return 50.0  # Abnormal speed
        else:
            return 25.0  # Slightly off
    
    @staticmethod
    def _acceleration_to_score(acceleration: float) -> float:
        """
        Convert acceleration to drowsiness score
        
        Args:
            acceleration: Magnitude of acceleration
            
        Returns:
            Score (0-100)
        """
        # High acceleration indicates jerky or erratic control
        if acceleration < 0.5:
            return 0.0  # Smooth
        elif acceleration < 1.5:
            return 30.0  # Slightly jerky
        elif acceleration < 3.0:
            return 60.0  # Very jerky
        else:
            return 100.0  # Dangerous
    
    def get_drowsiness_level(self, score: float) -> Tuple[int, str]:
        """
        Convert score to discrete drowsiness level
        
        Args:
            score: Drowsiness score (0-100)
            
        Returns:
            Tuple of (level_index, level_name)
        """
        level_names = ['Alert', 'Mild Drowsiness', 'Moderate Drowsiness', 'Severe Drowsiness', 'Critical']
        
        for i, threshold in enumerate(self.level_thresholds[:-1]):
            if threshold <= score < self.level_thresholds[i + 1]:
                return i, level_names[i]
        
        return self.num_levels - 1, level_names[-1]
    
    def update_history(self, score: float):
        """
        Update score history for temporal analysis
        
        Args:
            score: Current drowsiness score
        """
        self.score_history.append(score)
        if len(self.score_history) > self.max_history:
            self.score_history.pop(0)
    
    def get_trend(self) -> str:
        """
        Determine if drowsiness is increasing, stable, or decreasing
        
        Returns:
            String: 'Increasing', 'Stable', or 'Decreasing'
        """
        if len(self.score_history) < 3:
            return 'Insufficient data'
        
        recent = np.array(self.score_history[-3:])
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 2:
            return 'Increasing'
        elif trend < -2:
            return 'Decreasing'
        else:
            return 'Stable'
    
    def get_average_score(self) -> float:
        """Get average drowsiness score from history"""
        if not self.score_history:
            return 0.0
        return round(np.mean(self.score_history), 2)


class AlertGenerator:
    """Generate alerts based on drowsiness scores"""
    
    def __init__(self, critical_threshold=80, warning_threshold=60):
        """
        Initialize alert generator
        
        Args:
            critical_threshold: Score above which critical alert is triggered
            warning_threshold: Score above which warning alert is triggered
        """
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.alert_history = []
    
    def generate_alert(self, score: float, drowsiness_level: str) -> Dict:
        """
        Generate alert based on drowsiness score
        
        Args:
            score: Current drowsiness score
            drowsiness_level: Drowsiness level string
            
        Returns:
            Alert dictionary with severity and message
        """
        alert = {
            'score': score,
            'level': drowsiness_level,
            'severity': 'none',
            'message': '',
            'recommended_action': ''
        }
        
        if score >= self.critical_threshold:
            alert['severity'] = 'critical'
            alert['message'] = f'⚠️ CRITICAL DROWSINESS DETECTED (Score: {score})'
            alert['recommended_action'] = 'PULL OVER IMMEDIATELY. Driver is critically drowsy.'
        elif score >= self.warning_threshold:
            alert['severity'] = 'warning'
            alert['message'] = f'⚠️ WARNING: DROWSINESS DETECTED (Score: {score})'
            alert['recommended_action'] = 'Driver should consider taking a break or rest.'
        else:
            alert['severity'] = 'normal'
            alert['message'] = f'✓ Normal alertness level (Score: {score})'
            alert['recommended_action'] = 'Continue driving normally.'
        
        self.alert_history.append(alert)
        return alert
    
    def get_last_alert(self) -> Dict:
        """Get the most recent alert"""
        if self.alert_history:
            return self.alert_history[-1]
        return None


def main():
    """Example usage of drowsiness scoring system"""
    
    scorer = DrowsinessScorer(num_levels=5)
    alert_gen = AlertGenerator(critical_threshold=80, warning_threshold=60)
    
    # Example 1: Alert driver
    facial_features_alert = {
        'avg_ear': 0.25,  # Low EAR (drowsy eyes)
        'mar': 0.5,       # Slightly open mouth
        'blink_rate': 0.1 # Low blink rate
    }
    can_features_normal = {
        'speed': 50.0,
        'steering': 0.05,
        'acceleration_magnitude': 0.2
    }
    
    score = scorer.calculate_drowsiness_score(facial_features_alert, can_features_normal)
    level_idx, level_name = scorer.get_drowsiness_level(score)
    alert = alert_gen.generate_alert(score, level_name)
    
    print(f"Example 1 - Drowsy Driver:")
    print(f"  Score: {score}")
    print(f"  Level: {level_name}")
    print(f"  Alert: {alert['message']}")
    print(f"  Action: {alert['recommended_action']}\n")
    
    # Example 2: Normal driver
    facial_features_normal = {
        'avg_ear': 0.45,  # Normal EAR
        'mar': 0.3,       # Closed mouth
        'blink_rate': 0.3 # Normal blink rate
    }
    
    score = scorer.calculate_drowsiness_score(facial_features_normal, can_features_normal)
    level_idx, level_name = scorer.get_drowsiness_level(score)
    alert = alert_gen.generate_alert(score, level_name)
    
    print(f"Example 2 - Alert Driver:")
    print(f"  Score: {score}")
    print(f"  Level: {level_name}")
    print(f"  Alert: {alert['message']}")
    print(f"  Action: {alert['recommended_action']}")


if __name__ == "__main__":
    main()
