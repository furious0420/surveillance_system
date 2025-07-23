#!/usr/bin/env python3
"""
HAL Surveillance System - Real-Time Webcam Detection
Uses your trained 92.7% accurate model for live threat detection
"""

import cv2
import numpy as np
import pickle
import time
import os
from datetime import datetime
from pathlib import Path
import threading
import winsound  # For Windows audio alerts

print("🎯 HAL Surveillance System - Real-Time Webcam Detection")
print("=" * 60)

class HALWebcamSurveillance:
    def __init__(self):
        # Load your trained model
        self.model_path = "HAL_Surveillance_Model_20250719_100007.pkl"
        self.weights_path = "HAL_Model_20250719_100007/HAL_Model_Weights_20250719_100007.pt"
        
        # HAL Classes and threat levels
        self.hal_classes = {
            0: 'person_civilian', 1: 'person_armed', 2: 'weapon_gun', 3: 'weapon_knife',
            4: 'weapon_rifle', 5: 'vehicle_car', 6: 'vehicle_bike', 7: 'vehicle_truck',
            8: 'vehicle_tank', 9: 'animal_dog', 10: 'animal_livestock', 11: 'animal_wild',
            12: 'uav', 13: 'unknown_entity'
        }
        
        # Threat level colors (BGR format for OpenCV)
        self.threat_colors = {
            'HIGH_THREAT': (0, 0, 255),      # Red
            'POTENTIAL_THREAT': (0, 255, 255), # Yellow
            'LOW_THREAT': (0, 255, 0),        # Green
            'SAFE': (0, 255, 0)               # Green
        }
        
        # Threat classifications
        self.threat_levels = {
            'SAFE': [0],                                    # person_civilian
            'HIGH_THREAT': [1, 2, 3, 4, 11, 12],          # person_armed, weapons, animal_wild, uav
            'POTENTIAL_THREAT': [5, 6, 7, 8, 13],         # vehicles, unknown_entity
            'LOW_THREAT': [9, 10]                          # animal_dog, animal_livestock
        }
        
        # Settings
        self.confidence_threshold = 0.5
        self.audio_alerts = True
        self.show_fps = True
        self.paused = False
        
        # Statistics
        self.total_detections = 0
        self.threat_count = {'HIGH_THREAT': 0, 'POTENTIAL_THREAT': 0, 'LOW_THREAT': 0, 'SAFE': 0}
        
        print(f"📁 Loading model: {self.model_path}")
        self.load_model()
        
    def load_model(self):
        """Load your trained HAL model"""
        try:
            # Try to load the YOLO model directly
            from ultralytics import YOLO
            
            if os.path.exists(self.weights_path):
                self.model = YOLO(self.weights_path)
                print(f"✅ Model loaded successfully!")
                print(f"📊 Model accuracy: 92.7% mAP@0.5")
                print(f"🎯 Classes: {len(self.hal_classes)}")
            else:
                print(f"❌ Model weights not found: {self.weights_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        return True
    
    def get_threat_level(self, class_id):
        """Get threat level for detected class"""
        for threat_level, class_ids in self.threat_levels.items():
            if class_id in class_ids:
                return threat_level
        return 'UNKNOWN'
    
    def play_alert_sound(self, threat_level):
        """Play audio alert for threats"""
        if not self.audio_alerts:
            return
            
        def play_sound():
            try:
                if threat_level == 'HIGH_THREAT':
                    # High pitch beep for high threats
                    for _ in range(3):
                        winsound.Beep(1000, 200)
                        time.sleep(0.1)
                elif threat_level == 'POTENTIAL_THREAT':
                    # Medium pitch beep for potential threats
                    winsound.Beep(800, 300)
            except:
                pass  # Ignore audio errors
        
        # Play sound in separate thread to avoid blocking
        threading.Thread(target=play_sound, daemon=True).start()
    
    def draw_detection(self, frame, box, class_id, confidence):
        """Draw detection box with threat level styling"""
        x1, y1, x2, y2 = map(int, box)
        class_name = self.hal_classes.get(class_id, 'unknown')
        threat_level = self.get_threat_level(class_id)
        color = self.threat_colors.get(threat_level, (255, 255, 255))
        
        # Draw bounding box
        thickness = 3 if threat_level == 'HIGH_THREAT' else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        threat_label = f"[{threat_level}]"
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        threat_size = cv2.getTextSize(threat_label, font, font_scale, 2)[0]
        
        # Draw label background
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 30
        cv2.rectangle(frame, (x1, label_y - 25), (x1 + max(label_size[0], threat_size[0]) + 10, label_y + 5), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1 + 5, label_y - 10), font, font_scale, (255, 255, 255), 2)
        cv2.putText(frame, threat_label, (x1 + 5, label_y - 2), font, font_scale, (0, 0, 0), 2)
        
        return threat_level
    
    def draw_statistics(self, frame):
        """Draw statistics overlay"""
        height, width = frame.shape[:2]
        
        # Background for statistics
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 150), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "HAL SURVEILLANCE SYSTEM", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Model Accuracy: 92.7%", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Statistics
        y_offset = 75
        cv2.putText(frame, f"Total Detections: {self.total_detections}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        for threat_level, count in self.threat_count.items():
            color = self.threat_colors.get(threat_level, (255, 255, 255))
            cv2.putText(frame, f"{threat_level}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 15
    
    def draw_controls(self, frame):
        """Draw control instructions"""
        height, width = frame.shape[:2]
        
        # Background for controls
        cv2.rectangle(frame, (width - 250, height - 120), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 250, height - 120), (width - 10, height - 10), (255, 255, 255), 1)
        
        controls = [
            "CONTROLS:",
            "Q - Quit",
            "S - Screenshot", 
            "A - Toggle Audio",
            "F - Toggle FPS",
            "SPACE - Pause"
        ]
        
        y_offset = height - 100
        for control in controls:
            cv2.putText(frame, control, (width - 240, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"HAL_Detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def run_surveillance(self):
        """Main surveillance loop"""
        print(f"\n🚀 Starting HAL Webcam Surveillance...")
        print(f"📹 Initializing camera...")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Camera initialized successfully")
        print(f"🎯 HAL Surveillance System ACTIVE!")
        print(f"📊 Monitoring for 14 threat classes...")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                if not self.paused:
                    # Run detection
                    results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                    
                    # Process detections
                    frame_threats = []
                    if results and len(results) > 0:
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    # Extract detection data
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    conf = float(box.conf[0].cpu().numpy())
                                    cls = int(box.cls[0].cpu().numpy())
                                    
                                    # Draw detection
                                    threat_level = self.draw_detection(frame, xyxy, cls, conf)
                                    frame_threats.append(threat_level)
                                    
                                    # Update statistics
                                    self.total_detections += 1
                                    self.threat_count[threat_level] += 1
                    
                    # Play alerts for high threats
                    if 'HIGH_THREAT' in frame_threats:
                        self.play_alert_sound('HIGH_THREAT')
                    elif 'POTENTIAL_THREAT' in frame_threats:
                        self.play_alert_sound('POTENTIAL_THREAT')
                
                # Draw overlays
                self.draw_statistics(frame)
                self.draw_controls(frame)
                
                # Calculate and display FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if self.show_fps:
                    cv2.putText(frame, f"FPS: {current_fps}", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show pause indicator
                if self.paused:
                    cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Display frame
                cv2.imshow('HAL Surveillance System - Live Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame)
                elif key == ord('a'):
                    self.audio_alerts = not self.audio_alerts
                    print(f"🔊 Audio alerts: {'ON' if self.audio_alerts else 'OFF'}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord(' '):
                    self.paused = not self.paused
                    print(f"⏸️  Detection: {'PAUSED' if self.paused else 'RESUMED'}")
        
        except KeyboardInterrupt:
            print("\n🛑 Surveillance stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            print(f"\n📊 SURVEILLANCE SESSION SUMMARY:")
            print(f"   Total Detections: {self.total_detections}")
            for threat_level, count in self.threat_count.items():
                print(f"   {threat_level}: {count}")
            print(f"🎯 HAL Surveillance System stopped.")

def main():
    """Main function"""
    try:
        # Initialize surveillance system
        surveillance = HALWebcamSurveillance()
        
        # Start surveillance
        surveillance.run_surveillance()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure your webcam is connected and model files exist")

if __name__ == "__main__":
    main()
