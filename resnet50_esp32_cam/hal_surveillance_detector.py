#!/usr/bin/env python3
"""
HAL Surveillance Detection System
Real-time detection using trained .pkl model with external input support
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

# Model class definition (needed for pickle loading)
class UltimateHALModel(nn.Module):
    def __init__(self, num_classes=6, backbone='resnet50'):
        super().__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.backbone.fc.in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)

class HALSurveillanceDetector:
    def __init__(self, model_path="models/hal_surveillance_final.pkl"):
        """Initialize the HAL Surveillance Detector"""
        print("🛡️ HAL SURVEILLANCE DETECTION SYSTEM")
        print("=" * 50)
        print(f"📦 Loading model from {model_path}...")

        # Try to load model package
        try:
            with open(model_path, 'rb') as f:
                self.model_package = pickle.load(f)

            # Extract model info
            self.class_names = self.model_package['class_names']
            self.class_to_idx = self.model_package['class_to_idx']
            self.idx_to_class = self.model_package['idx_to_class']

            # Create new model instance and load state dict
            self.model = UltimateHALModel(num_classes=6, backbone='resnet50')
            self.model.load_state_dict(self.model_package['model_state_dict'])

        except Exception as e:
            print(f"❌ Error loading pickle model: {e}")
            print("🔄 Trying to load from PyTorch checkpoint...")

            # Fallback: try loading from .pth file
            pth_path = "models/ultimate_hal_surveillance.pth"
            if os.path.exists(pth_path):
                checkpoint = torch.load(pth_path, map_location='cpu')

                self.model = UltimateHALModel(num_classes=6, backbone='resnet50')
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Default class mapping
                self.class_names = ['background', 'human', 'vehicle', 'weapon', 'uav', 'animal']
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
                self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

                # Create model package for compatibility
                self.model_package = {
                    'model_info': {
                        'final_accuracy': checkpoint.get('best_val_acc', 0.96),
                        'training_epoch': checkpoint.get('epoch', 0)
                    }
                }
            else:
                raise FileNotFoundError(f"Neither {model_path} nor {pth_path} found!")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Threat levels
        self.threat_levels = {
            'background': ('LOW', '🟢'),
            'human': ('MEDIUM', '🟡'), 
            'vehicle': ('HIGH', '🟠'),
            'weapon': ('CRITICAL', '🔴'),
            'uav': ('CRITICAL', '🔴'),
            'animal': ('LOW-MEDIUM', '🟡')
        }
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.class_names}")
        print(f"   Model accuracy: {self.model_package['model_info']['final_accuracy']:.2%}")
        
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        if isinstance(image, str):
            # Load from file path
            image = Image.open(image).convert('RGBA').convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert from OpenCV format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor
    
    def predict(self, image_input, confidence_threshold=0.7, multi_class_threshold=0.3):
        """Make prediction on image with multi-class detection support"""
        # Preprocess
        image_tensor = self.preprocess_image(image_input)
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            all_probs = probabilities[0].cpu().numpy()

        # Get all class probabilities
        all_probabilities = dict(zip(self.class_names, all_probs))

        # Find all classes above multi-class threshold
        detected_classes = []
        highest_threat_level = 'LOW'
        highest_threat_emoji = '🟢'

        for class_name, prob in zip(self.class_names, all_probs):
            if prob >= multi_class_threshold and class_name != 'background':
                threat_level, threat_emoji = self.threat_levels.get(class_name, ('UNKNOWN', '❓'))

                detected_classes.append({
                    'class': class_name,
                    'confidence': prob,
                    'threat_level': threat_level,
                    'threat_emoji': threat_emoji
                })

                # Update highest threat level
                threat_priority = {'LOW': 1, 'LOW-MEDIUM': 2, 'MEDIUM': 3, 'HIGH': 4, 'CRITICAL': 5}
                if threat_priority.get(threat_level, 0) > threat_priority.get(highest_threat_level, 0):
                    highest_threat_level = threat_level
                    highest_threat_emoji = threat_emoji

        # Sort detected classes by confidence
        detected_classes.sort(key=lambda x: x['confidence'], reverse=True)

        # Primary prediction (highest confidence)
        primary_class_idx = np.argmax(all_probs)
        primary_class = self.class_names[primary_class_idx]
        primary_confidence = all_probs[primary_class_idx]

        # Determine if detection is certain enough
        if primary_confidence < confidence_threshold:
            highest_threat_level = 'UNCERTAIN'
            highest_threat_emoji = '❓'

        return {
            'primary_class': primary_class,
            'primary_confidence': primary_confidence,
            'detected_classes': detected_classes,
            'multi_class_count': len(detected_classes),
            'highest_threat_level': highest_threat_level,
            'highest_threat_emoji': highest_threat_emoji,
            'all_probabilities': all_probabilities,
            'is_multi_class': len(detected_classes) > 1
        }
    
    def detect_image(self, image_path, save_result=True, output_dir="results", multi_class_threshold=0.3):
        """Detect objects in a single image with multi-class support"""
        print(f"\n🔍 Analyzing image: {image_path}")

        # Load original image
        original_image = Image.open(image_path).convert('RGB')

        # Make prediction with multi-class detection
        result = self.predict(image_path, multi_class_threshold=multi_class_threshold)

        # Print results
        print(f"📊 Detection Results:")
        print(f"   Primary Detection: {result['primary_class'].upper()} ({result['primary_confidence']:.2%})")
        print(f"   Overall Threat Level: {result['highest_threat_emoji']} {result['highest_threat_level']}")

        # Show multi-class detections
        if result['is_multi_class']:
            print(f"\n🎯 Multi-Class Detections (>{multi_class_threshold*100:.0f}% confidence):")
            for i, detection in enumerate(result['detected_classes'], 1):
                print(f"   {i}. {detection['class'].upper()}: {detection['confidence']:.2%} - {detection['threat_emoji']} {detection['threat_level']}")
        else:
            print(f"\n🎯 Single Class Detection")

        print(f"\n📈 All Class Probabilities:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            emoji = "🔥" if prob > 0.5 else "📊" if prob > 0.1 else "📉"
            print(f"   {emoji} {class_name.capitalize()}: {prob:.2%}")

        # Save annotated result
        if save_result:
            os.makedirs(output_dir, exist_ok=True)

            # Create annotated image
            draw = ImageDraw.Draw(original_image)

            # Create annotation text
            if result['is_multi_class']:
                main_text = f"MULTI-CLASS DETECTION"
                detail_text = f"Primary: {result['primary_class'].upper()} ({result['primary_confidence']:.1%})"
                threat_text = f"Threat: {result['highest_threat_emoji']} {result['highest_threat_level']}"

                # Add detected classes
                classes_text = "Detected: " + ", ".join([f"{d['class'].upper()}" for d in result['detected_classes'][:3]])

                full_text = f"{main_text}\n{detail_text}\n{threat_text}\n{classes_text}"
            else:
                full_text = f"{result['primary_class'].upper()} ({result['primary_confidence']:.1%})\nThreat: {result['highest_threat_level']}"

            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Add background rectangle for text
            bbox = draw.textbbox((10, 10), full_text, font=font)
            draw.rectangle((bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5), fill='black', outline='white', width=2)
            draw.text((10, 10), full_text, fill='white', font=font)

            # Save result
            output_path = os.path.join(output_dir, f"detected_{Path(image_path).name}")
            original_image.save(output_path)
            print(f"💾 Result saved to: {output_path}")

        return result
    
    def detect_video(self, video_path, output_path="results/detected_video.mp4", skip_frames=5):
        """Detect objects in video"""
        print(f"\n🎥 Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame for efficiency
            if frame_count % skip_frames == 0:
                # Make prediction with multi-class detection
                result = self.predict(frame, multi_class_threshold=0.4)
                detection_history.append(result)

                # Create annotation text
                if result['is_multi_class']:
                    main_text = f"MULTI: {result['primary_class'].upper()} ({result['primary_confidence']:.1%})"
                    threat_text = f"{result['highest_threat_emoji']} {result['highest_threat_level']}"
                    classes_text = f"+ {result['multi_class_count']-1} more"
                else:
                    main_text = f"{result['primary_class'].upper()} ({result['primary_confidence']:.1%})"
                    threat_text = f"{result['highest_threat_emoji']} {result['highest_threat_level']}"
                    classes_text = ""

                # Color based on threat level
                if result['highest_threat_level'] == 'CRITICAL':
                    color = (0, 0, 255)  # Red
                elif result['highest_threat_level'] == 'HIGH':
                    color = (0, 165, 255)  # Orange
                elif result['highest_threat_level'] == 'MEDIUM':
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green

                # Add background rectangles and text
                y_offset = 10
                for text_line in [main_text, threat_text, classes_text]:
                    if text_line:
                        (text_width, text_height), _ = cv2.getTextSize(text_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (10, y_offset), (10 + text_width + 10, y_offset + text_height + 10), (0, 0, 0), -1)
                        cv2.putText(frame, text_line, (15, y_offset + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += text_height + 15

                # Progress indicator
                if frame_count % (skip_frames * 30) == 0:  # Every 30 detections
                    progress = (frame_count / total_frames) * 100
                    if result['is_multi_class']:
                        print(f"   Progress: {progress:.1f}% - Multi-class: {result['primary_class']} + {result['multi_class_count']-1} more")
                    else:
                        print(f"   Progress: {progress:.1f}% - Single: {result['primary_class']} ({result['primary_confidence']:.1%})")
            
            # Write frame
            out.write(frame)
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Video analysis complete!")
        print(f"💾 Annotated video saved to: {output_path}")
        
        # Summary statistics
        if detection_history:
            most_common = max(set([d['predicted_class'] for d in detection_history]), 
                             key=[d['predicted_class'] for d in detection_history].count)
            avg_confidence = np.mean([d['confidence'] for d in detection_history])
            
            print(f"\n📊 Video Summary:")
            print(f"   Most detected class: {most_common}")
            print(f"   Average confidence: {avg_confidence:.2%}")
            print(f"   Total detections: {len(detection_history)}")
        
        return detection_history
    
    def detect_webcam(self, camera_index=0, confidence_threshold=0.7):
        """Real-time detection from webcam"""
        print(f"\n📷 Starting real-time detection from camera {camera_index}")
        print("Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make prediction with multi-class detection
            result = self.predict(frame, confidence_threshold, multi_class_threshold=0.4)

            # Create annotation text
            if result['is_multi_class']:
                main_text = f"MULTI-CLASS: {result['primary_class'].upper()} ({result['primary_confidence']:.1%})"
                threat_text = f"{result['highest_threat_emoji']} {result['highest_threat_level']}"
                detail_text = f"Detected {result['multi_class_count']} classes"
            else:
                main_text = f"{result['primary_class'].upper()} ({result['primary_confidence']:.1%})"
                threat_text = f"{result['highest_threat_emoji']} {result['highest_threat_level']}"
                detail_text = "Single class detection"

            # Color based on threat level
            if result['highest_threat_level'] == 'CRITICAL':
                color = (0, 0, 255)  # Red
            elif result['highest_threat_level'] == 'HIGH':
                color = (0, 165, 255)  # Orange
            elif result['highest_threat_level'] == 'MEDIUM':
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            # Add text with background
            cv2.rectangle(frame, (10, 10), (500, 100), (0, 0, 0), -1)
            cv2.putText(frame, main_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, threat_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, detail_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show frame
            cv2.imshow('HAL Surveillance - Real-time Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"results/screenshot_{screenshot_count:03d}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("📷 Real-time detection stopped")

def main():
    parser = argparse.ArgumentParser(description='HAL Surveillance Detection System')
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], required=True,
                       help='Detection mode')
    parser.add_argument('--input', type=str, help='Input file path (for image/video mode)')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory or file path')
    parser.add_argument('--model', type=str, default='models/hal_surveillance_final.pkl',
                       help='Path to trained model (.pkl file)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Primary confidence threshold (0.0-1.0)')
    parser.add_argument('--multi-threshold', type=float, default=0.3,
                       help='Multi-class detection threshold (0.0-1.0)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index for webcam mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HALSurveillanceDetector(args.model)
    
    # Run detection based on mode
    if args.mode == 'image':
        if not args.input:
            print("❌ Error: --input required for image mode")
            return
        detector.detect_image(args.input, output_dir=args.output, multi_class_threshold=getattr(args, 'multi_threshold', 0.3))

    elif args.mode == 'video':
        if not args.input:
            print("❌ Error: --input required for video mode")
            return
        output_path = args.output if args.output.endswith('.mp4') else os.path.join(args.output, 'detected_video.mp4')
        detector.detect_video(args.input, output_path)

    elif args.mode == 'webcam':
        detector.detect_webcam(args.camera, args.confidence)

if __name__ == "__main__":
    main()

# Example usage functions for direct import
def quick_detect_image(image_path, model_path="models/hal_surveillance_final.pkl", multi_class_threshold=0.3):
    """Quick function to detect objects in an image with multi-class support"""
    detector = HALSurveillanceDetector(model_path)
    return detector.detect_image(image_path, multi_class_threshold=multi_class_threshold)

def quick_detect_batch(image_folder, model_path="models/hal_surveillance_final.pkl", multi_class_threshold=0.3):
    """Quick function to detect objects in multiple images with multi-class support"""
    detector = HALSurveillanceDetector(model_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
        image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))

    results = []
    print(f"\n🔍 Processing {len(image_files)} images from {image_folder}")

    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        result = detector.detect_image(str(image_path), multi_class_threshold=multi_class_threshold)
        results.append({'file': image_path.name, 'result': result})

    return results
