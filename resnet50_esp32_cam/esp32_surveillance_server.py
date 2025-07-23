#!/usr/bin/env python3
"""
ESP32 CAM + HAL Surveillance WebSocket Server
Real-time inference server for ESP32 camera integration
"""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime
import os
import threading
import glob
from hal_surveillance_detector import HALSurveillanceDetector, UltimateHALModel

class ESP32SurveillanceServer:
    def __init__(self, model_path="models/hal_surveillance_final.pkl", port=8765):
        """Initialize ESP32 Surveillance Server"""
        print("🛡️ ESP32 HAL SURVEILLANCE SERVER")
        print("=" * 50)
        
        self.port = port
        self.detector = None
        self.connected_clients = set()
        self.detection_history = []
        self.alert_threshold = 0.8
        self.save_detections = True
        
        # Create directories
        os.makedirs("esp32_captures", exist_ok=True)
        os.makedirs("esp32_alerts", exist_ok=True)

        # Auto-delete settings for space management
        self.auto_delete_enabled = True
        self.keep_images_seconds = 5  # Keep images for only 5 seconds

        # Initialize detector
        self.init_detector(model_path)

        # Start auto-delete thread
        if self.auto_delete_enabled:
            self.cleanup_thread = threading.Thread(target=self.auto_delete_old_images, daemon=True)
            self.cleanup_thread.start()
            print(f"🗑️ Auto-delete enabled: keeping images for {self.keep_images_seconds} seconds")
        
    def init_detector(self, model_path):
        """Initialize the HAL surveillance detector"""
        try:
            print(f"📦 Loading HAL surveillance model...")
            self.detector = HALSurveillanceDetector(model_path)
            print(f"✅ Model loaded successfully!")
            print(f"   Device: {self.detector.device}")
            print(f"   Classes: {self.detector.class_names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.detector = None

    def auto_delete_old_images(self):
        """Automatically delete old images to save disk space"""
        while True:
            try:
                current_time = time.time()

                # Delete old captures (keep for 5 seconds only)
                for image_path in glob.glob("esp32_captures/*.jpg"):
                    if os.path.exists(image_path):
                        file_age = current_time - os.path.getmtime(image_path)
                        if file_age > self.keep_images_seconds:
                            os.remove(image_path)

                # Delete old alerts (keep alerts longer - 30 seconds)
                for image_path in glob.glob("esp32_alerts/*.jpg"):
                    if os.path.exists(image_path):
                        file_age = current_time - os.path.getmtime(image_path)
                        if file_age > 30:  # Keep alerts for 30 seconds
                            os.remove(image_path)

                # Run cleanup every 2 seconds
                time.sleep(2)

            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
                time.sleep(5)

    def decode_image(self, base64_data):
        """Decode base64 image from ESP32"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"❌ Error decoding image: {e}")
            return None
    
    def encode_image(self, image):
        """Encode image to base64 for sending back"""
        try:
            # Convert PIL image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            return base64_data
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None
    
    def process_detection(self, image):
        """Process image and return detection results"""
        if not self.detector:
            return {
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Run inference with timing for performance monitoring
            start_time = time.time()
            result = self.detector.predict(image, multi_class_threshold=0.3)
            detection_time = time.time() - start_time
            
            # Create response with performance info
            detection_result = {
                'timestamp': datetime.now().isoformat(),
                'primary_class': result['primary_class'],
                'primary_confidence': float(result['primary_confidence']),
                'threat_level': result['highest_threat_level'],
                'threat_emoji': result['highest_threat_emoji'],
                'is_multi_class': result['is_multi_class'],
                'detected_classes': result['detected_classes'],
                'all_probabilities': {k: float(v) for k, v in result['all_probabilities'].items()},
                'alert': result['primary_confidence'] > self.alert_threshold and result['highest_threat_level'] in ['HIGH', 'CRITICAL'],
                'detection_time_ms': round(detection_time * 1000, 1)  # Add timing info
            }
            
            # Save detection history
            self.detection_history.append(detection_result)
            
            # Keep only last 100 detections
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-100:]
            
            # Save alert images
            if detection_result['alert'] and self.save_detections:
                self.save_alert_image(image, detection_result)
            
            return detection_result
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_alert_image(self, image, detection_result):
        """Save alert images to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            threat_level = detection_result['threat_level']
            primary_class = detection_result['primary_class']
            confidence = detection_result['primary_confidence']
            
            filename = f"esp32_alerts/ALERT_{timestamp}_{primary_class}_{confidence:.0%}_{threat_level}.jpg"
            image.save(filename, quality=95)
            print(f"🚨 Alert saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving alert: {e}")
    
    async def handle_client(self, websocket, path=None):
        """Handle WebSocket client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"📱 ESP32 connected: {client_id}")
        
        self.connected_clients.add(websocket)
        
        try:
            # Send welcome message
            welcome_msg = {
                'type': 'welcome',
                'message': 'Connected to HAL Surveillance Server',
                'server_time': datetime.now().isoformat(),
                'model_classes': self.detector.class_names if self.detector else [],
                'alert_threshold': self.alert_threshold
            }
            await websocket.send(json.dumps(welcome_msg))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'image':
                        # Process image detection
                        image_data = data.get('image')
                        if image_data:
                            # Decode image
                            image = self.decode_image(image_data)
                            if image:
                                # Run detection
                                detection_result = self.process_detection(image)
                                
                                # Send result back (convert numpy types to JSON-serializable types)
                                json_safe_result = {
                                    'primary_class': str(detection_result['primary_class']),
                                    'primary_confidence': float(detection_result['primary_confidence']),
                                    'threat_level': str(detection_result['threat_level']),
                                    'alert': bool(detection_result['alert']),
                                    'threat_emoji': str(detection_result['threat_emoji'])
                                }

                                response = {
                                    'type': 'detection_result',
                                    'result': json_safe_result
                                }
                                await websocket.send(json.dumps(response))
                                
                                # Print detection info
                                if 'error' not in detection_result:
                                    threat_emoji = detection_result['threat_emoji']
                                    primary_class = detection_result['primary_class']
                                    confidence = detection_result['primary_confidence']
                                    threat_level = detection_result['threat_level']
                                    
                                    print(f"🔍 {client_id}: {threat_emoji} {primary_class.upper()} ({confidence:.1%}) - {threat_level}")
                                    
                                    if detection_result['alert']:
                                        print(f"🚨 ALERT: {primary_class.upper()} detected with {confidence:.1%} confidence!")
                            else:
                                await websocket.send(json.dumps({
                                    'type': 'error',
                                    'message': 'Failed to decode image'
                                }))
                    
                    elif data.get('type') == 'ping':
                        # Respond to ping
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        }))
                    
                    elif data.get('type') == 'get_history':
                        # Send detection history
                        await websocket.send(json.dumps({
                            'type': 'history',
                            'detections': self.detection_history[-10:]  # Last 10 detections
                        }))
                
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 ESP32 disconnected: {client_id}")
        except Exception as e:
            print(f"❌ Connection error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        print(f"🚀 Starting WebSocket server on port {self.port}")
        print(f"📡 ESP32 can connect to: ws://YOUR_COMPUTER_IP:{self.port}")
        print(f"🛡️ HAL Surveillance ready for real-time detection!")
        print(f"📁 Alerts will be saved to: esp32_alerts/")
        print("-" * 50)
        
        server = await websockets.serve(
            self.handle_client,
            "0.0.0.0",  # Listen on all interfaces
            self.port
        )
        
        print(f"✅ Server running! Waiting for ESP32 connections...")
        await server.wait_closed()

def main():
    """Main function"""
    print("🛡️ ESP32 HAL SURVEILLANCE SERVER")
    print("=" * 50)
    print("📋 Setup Instructions:")
    print("1. Make sure your ESP32-CAM is configured with your WiFi")
    print("2. Update the websocket_server IP in ESP32 code to your computer's IP")
    print("3. Upload the Arduino code to ESP32-CAM")
    print("4. Power on ESP32-CAM and it will connect automatically")
    print("-" * 50)

    # Configuration
    MODEL_PATH = "models/hal_surveillance_final.pkl"
    PORT = 8765

    # Get computer's IP address
    import socket
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"💻 Your computer's IP: {local_ip}")
        print(f"📡 ESP32 should connect to: ws://{local_ip}:{PORT}")
    except:
        print("💻 Could not determine IP automatically. Use 'ipconfig' to find it.")

    print("-" * 50)

    # Create and start server
    server = ESP32SurveillanceServer(MODEL_PATH, PORT)

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("👋 Thank you for using HAL Surveillance!")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
