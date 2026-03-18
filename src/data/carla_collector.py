"""
CARLA Data Collector for Driver Drowsiness Detection
Collects synchronized video frames and CAN signals (speed, steering, throttle, brake, acceleration)
from the CARLA simulator.
"""

import carla
import csv
import time
import math
import os
from datetime import datetime
import numpy as np
from pathlib import Path


class CarlaDataCollector:
    def __init__(self, host='localhost', port=2000, output_dir='carla_dataset'):
        """
        Initialize CARLA data collector
        
        Args:
            host: CARLA server host (default: localhost)
            port: CARLA server port (default: 2000)
            output_dir: Directory to save collected data
        """
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
        # Create output directory structure
        self._setup_output_dirs()
        
    def _setup_output_dirs(self):
        """Create output directory structure"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/frames").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/can_signals").mkdir(parents=True, exist_ok=True)
        
    def connect_to_carla(self):
        """Connect to CARLA server and initialize world"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"✓ Connected to CARLA server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            return False
    
    def get_vehicle(self):
        """Get the first available vehicle in the world"""
        try:
            vehicles = self.world.get_actors().filter('vehicle.*')
            if len(vehicles) > 0:
                self.vehicle = vehicles[0]
                print(f"✓ Vehicle selected: {self.vehicle.type_id}")
                return self.vehicle
            else:
                print("✗ No vehicles found in the world. Please spawn a vehicle first.")
                return None
        except Exception as e:
            print(f"✗ Error getting vehicle: {e}")
            return None
    
    def setup_camera(self):
        """Setup RGB camera sensor on the vehicle"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Set camera resolution
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            
            # Attach camera to vehicle (driver view)
            relative_transform = carla.Transform(
                carla.Location(x=0.5, z=1.3),  # Mounted on dashboard
                carla.Rotation(pitch=-15, yaw=0, roll=0)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp,
                relative_transform,
                attach_to=self.vehicle
            )
            print("✓ Camera sensor attached to vehicle")
            return True
        except Exception as e:
            print(f"✗ Error setting up camera: {e}")
            return False
    
    def get_can_telemetry(self, frame_id):
        """
        Extract telemetry data from vehicle (simulated CAN signals)
        
        Args:
            frame_id: Current frame number for synchronization
            
        Returns:
            Dictionary with CAN signal data
        """
        try:
            # Velocity → Speed (km/h)
            velocity = self.vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Control inputs
            control = self.vehicle.get_control()
            
            # Acceleration
            acceleration = self.vehicle.get_acceleration()
            accel_magnitude = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
            
            # Angular velocity (yaw rate) for steering behavior
            angular_vel = self.vehicle.get_angular_velocity()
            yaw_rate = angular_vel.z
            
            return {
                'frame_id': frame_id,
                'timestamp': time.time(),
                'speed_kmh': round(speed_kmh, 2),
                'steering': round(control.steer, 4),     # [-1.0, 1.0]
                'throttle': round(control.throttle, 4),  # [0.0, 1.0]
                'brake': round(control.brake, 4),        # [0.0, 1.0]
                'acceleration_x': round(acceleration.x, 4),
                'acceleration_y': round(acceleration.y, 4),
                'acceleration_z': round(acceleration.z, 4),
                'acceleration_magnitude': round(accel_magnitude, 4),
                'yaw_rate': round(yaw_rate, 4),
                'label': 0  # 0 = Alert, 1 = Drowsy (manual labeling required)
            }
        except Exception as e:
            print(f"✗ Error reading telemetry: {e}")
            return None
    
    def collect_data(self, duration_seconds=60, frame_rate=10):
        """
        Collect synchronized video frames and CAN signals
        
        Args:
            duration_seconds: How long to collect data
            frame_rate: Frames per second (determines sampling interval)
        """
        if not self.vehicle:
            print("✗ No vehicle selected. Call get_vehicle() first.")
            return False
        
        csv_filename = f"{self.output_dir}/can_signals/telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        frame_interval = 1.0 / frame_rate
        
        # CSV header
        csv_headers = [
            'frame_id', 'timestamp', 'speed_kmh', 'steering', 'throttle', 'brake',
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'acceleration_magnitude', 'yaw_rate', 'label'
        ]
        
        print(f"\n📊 Starting data collection for {duration_seconds} seconds at {frame_rate} FPS...")
        print(f"💾 CAN signals will be saved to: {csv_filename}")
        
        frame_id = 0
        start_time = time.time()
        last_frame_time = start_time
        
        try:
            with open(csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
                csv_writer.writeheader()
                
                while (time.time() - start_time) < duration_seconds:
                    current_time = time.time()
                    
                    # Collect frame at specified interval
                    if (current_time - last_frame_time) >= frame_interval:
                        telemetry = self.get_can_telemetry(frame_id)
                        if telemetry:
                            csv_writer.writerow(telemetry)
                            print(f"  Frame {frame_id}: Speed={telemetry['speed_kmh']} km/h, "
                                  f"Steer={telemetry['steering']:.3f}, "
                                  f"Throttle={telemetry['throttle']:.2f}")
                        
                        frame_id += 1
                        last_frame_time = current_time
                    
                    # Small sleep to prevent CPU spike
                    time.sleep(0.001)
            
            print(f"\n✓ Data collection complete! Collected {frame_id} frames")
            return True
            
        except Exception as e:
            print(f"✗ Error during data collection: {e}")
            return False
    
    def cleanup(self):
        """Cleanup: destroy camera and disconnect from CARLA"""
        try:
            if self.camera:
                self.camera.destroy()
                print("✓ Camera sensor destroyed")
            
            if self.client:
                print("✓ CARLA connection closed")
        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")


def main():
    """Example usage of CarlaDataCollector"""
    collector = CarlaDataCollector(
        host='localhost',
        port=2000,
        output_dir='carla_dataset'
    )
    
    # Connect and setup
    if not collector.connect_to_carla():
        return
    
    if not collector.get_vehicle():
        return
    
    if not collector.setup_camera():
        return
    
    # Collect data
    collector.collect_data(duration_seconds=30, frame_rate=10)
    
    # Cleanup
    collector.cleanup()


if __name__ == "__main__":
    main()
