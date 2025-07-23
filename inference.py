#!/usr/bin/env python3
"""
Inference script for YOLOv11s-obb aerial view detection
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import argparse

def run_inference(model_path, image_path, output_dir="inference_results", conf_threshold=0.25):
    """Run inference on aerial images"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    if os.path.isfile(image_path):
        # Single image
        images = [image_path]
    elif os.path.isdir(image_path):
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = [f for f in Path(image_path).rglob('*') 
                 if f.suffix.lower() in image_extensions]
    else:
        print(f"Invalid image path: {image_path}")
        return
    
    print(f"Processing {len(images)} images...")
    
    total_detections = 0
    
    for i, img_path in enumerate(images):
        print(f"Processing {i+1}/{len(images)}: {img_path}")
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold)
        
        # Save results
        for j, result in enumerate(results):
            # Save annotated image
            output_path = Path(output_dir) / f"{Path(img_path).stem}_result.jpg"
            result.save(str(output_path))
            
            # Print detection info
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb.cls) > 0:
                num_detections = len(result.obb.cls)
                total_detections += num_detections
                print(f"  Found {num_detections} objects")
                
                # Print class names and confidence scores
                for k in range(num_detections):
                    class_id = int(result.obb.cls[k])
                    confidence = float(result.obb.conf[k])
                    class_name = result.names[class_id]
                    print(f"    {class_name}: {confidence:.3f}")
                    
                # Save detection details to text file
                txt_path = Path(output_dir) / f"{Path(img_path).stem}_detections.txt"
                with open(txt_path, 'w') as f:
                    f.write(f"Image: {img_path}\n")
                    f.write(f"Total detections: {num_detections}\n\n")
                    for k in range(num_detections):
                        class_id = int(result.obb.cls[k])
                        confidence = float(result.obb.conf[k])
                        class_name = result.names[class_id]
                        
                        # Get oriented bounding box coordinates
                        if hasattr(result.obb, 'xyxyxyxy'):
                            coords = result.obb.xyxyxyxy[k].cpu().numpy()
                            f.write(f"Detection {k+1}:\n")
                            f.write(f"  Class: {class_name}\n")
                            f.write(f"  Confidence: {confidence:.3f}\n")
                            f.write(f"  Coordinates: {coords.tolist()}\n\n")
            else:
                print("  No objects detected")
    
    print(f"\nInference completed!")
    print(f"Total detections across all images: {total_detections}")
    print(f"Results saved to: {output_dir}")

def create_sample_aerial_image():
    """Create a sample aerial-like image for testing"""
    print("Creating sample aerial image for testing...")
    
    # Create a synthetic aerial image
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    # Add some background (grass/terrain)
    img[:, :] = [34, 139, 34]  # Forest green
    
    # Add some rectangular structures (buildings)
    cv2.rectangle(img, (200, 200), (300, 280), (128, 128, 128), -1)  # Building 1
    cv2.rectangle(img, (400, 150), (550, 250), (96, 96, 96), -1)     # Building 2
    cv2.rectangle(img, (600, 300), (750, 400), (160, 160, 160), -1)  # Building 3
    
    # Add some roads
    cv2.rectangle(img, (0, 350), (1024, 380), (64, 64, 64), -1)      # Horizontal road
    cv2.rectangle(img, (500, 0), (530, 1024), (64, 64, 64), -1)      # Vertical road
    
    # Add some circular structures
    cv2.circle(img, (800, 200), 50, (139, 69, 19), -1)               # Storage tank
    cv2.circle(img, (150, 600), 40, (139, 69, 19), -1)               # Storage tank
    
    # Add some vehicles (small rectangles)
    cv2.rectangle(img, (220, 360), (235, 375), (255, 255, 255), -1)  # Vehicle 1
    cv2.rectangle(img, (510, 360), (525, 375), (255, 0, 0), -1)      # Vehicle 2
    
    # Save the sample image
    sample_path = "sample_aerial_image.jpg"
    cv2.imwrite(sample_path, img)
    print(f"Sample aerial image created: {sample_path}")
    return sample_path

def main():
    parser = argparse.ArgumentParser(description='YOLOv11s-obb Aerial View Inference')
    parser.add_argument('model_path', help='Path to the YOLO model (.pt file)')
    parser.add_argument('image_path', help='Path to image file or directory')
    parser.add_argument('--output', '-o', default='inference_results', 
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample aerial image for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample_path = create_sample_aerial_image()
        print(f"Sample image created. You can now run:")
        print(f"python inference.py {args.model_path} {sample_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("\nAvailable models in current directory:")
        pt_files = list(Path('.').glob('*.pt'))
        if pt_files:
            for pt_file in pt_files:
                print(f"  {pt_file}")
        else:
            print("  No .pt files found")
        
        # Check training results
        training_dirs = list(Path('.').glob('training_results_*'))
        if training_dirs:
            print("\nAvailable trained models:")
            for training_dir in training_dirs:
                best_model = training_dir / "train" / "weights" / "best.pt"
                if best_model.exists():
                    print(f"  {best_model}")
        return
    
    run_inference(args.model_path, args.image_path, args.output, args.conf)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("YOLOv11s-obb Aerial View Inference")
        print("Usage examples:")
        print("  python inference.py yolo11s-obb.pt aerial_image.jpg")
        print("  python inference.py best.pt images_folder/ --output results/")
        print("  python inference.py yolo11s-obb.pt --create-sample")
        print("\nFor help: python inference.py --help")
    else:
        main()
