#!/usr/bin/env python3
"""
Script to download DOTA datasets for YOLOv11 OBB training
"""

import os
import subprocess
import sys
from ultralytics import YOLO

def download_dota8_dataset():
    """Download DOTA8 sample dataset"""
    print("Downloading DOTA8 sample dataset...")
    
    try:
        # Use YOLO to download the dataset
        # This will automatically download the DOTA8 dataset configuration
        model = YOLO("yolo11s-obb.pt")
        
        # Try to validate with DOTA8 dataset - this will trigger download
        print("Triggering DOTA8 dataset download...")
        try:
            # This command will download the dataset if it doesn't exist
            result = subprocess.run([
                sys.executable, "-c", 
                "from ultralytics import YOLO; model = YOLO('yolo11s-obb.pt'); model.val(data='dota8.yaml', verbose=False)"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("DOTA8 dataset download completed successfully!")
            else:
                print(f"DOTA8 download had some issues: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("DOTA8 download timed out, but may have partially completed")
        except Exception as e:
            print(f"Error downloading DOTA8: {e}")
            
    except Exception as e:
        print(f"Error setting up DOTA8 download: {e}")

def download_dota_full_dataset():
    """Download full DOTA v1.0 dataset"""
    print("\nDownloading full DOTA v1.0 dataset...")
    print("Note: This is a large dataset (several GB) and may take a while...")
    
    try:
        # Try to validate with DOTAv1 dataset - this will trigger download
        print("Triggering DOTAv1 dataset download...")
        try:
            result = subprocess.run([
                sys.executable, "-c", 
                "from ultralytics import YOLO; model = YOLO('yolo11s-obb.pt'); model.val(data='DOTAv1.yaml', verbose=False)"
            ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
            
            if result.returncode == 0:
                print("DOTAv1 dataset download completed successfully!")
            else:
                print(f"DOTAv1 download had some issues: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("DOTAv1 download timed out, but may have partially completed")
        except Exception as e:
            print(f"Error downloading DOTAv1: {e}")
            
    except Exception as e:
        print(f"Error setting up DOTAv1 download: {e}")

def check_dataset_locations():
    """Check where datasets are downloaded"""
    print("\nChecking dataset locations...")
    
    # Common locations where ultralytics downloads datasets
    possible_locations = [
        os.path.expanduser("~/datasets"),
        os.path.expanduser("~/.ultralytics/datasets"),
        "./datasets",
        os.path.join(os.getcwd(), "datasets")
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"Found datasets directory: {location}")
            try:
                contents = os.listdir(location)
                if contents:
                    print(f"  Contents: {contents}")
                else:
                    print("  (empty)")
            except PermissionError:
                print("  (permission denied)")
        else:
            print(f"Not found: {location}")

def create_dataset_info():
    """Create information about the datasets"""
    info_content = """
# DOTA Dataset Information

## DOTA8 (Sample Dataset)
- Small subset of DOTA dataset for testing
- Contains 8 sample images with oriented bounding box annotations
- Good for quick testing and validation
- Classes: plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool

## DOTAv1 (Full Dataset)
- Large-scale dataset for object detection in aerial images
- Contains 2,806 aerial images
- 188,282 instances across 15 categories
- Image sizes ranging from 800×800 to 4000×4000 pixels
- Oriented bounding box annotations
- Split into train/val/test sets

## Usage
```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11s-obb.pt")

# Train on DOTA8 (quick test)
model.train(data="dota8.yaml", epochs=10, imgsz=1024)

# Train on full DOTAv1 (production)
model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)

# Validate
model.val(data="dota8.yaml")  # or "DOTAv1.yaml"

# Predict
results = model("path/to/aerial/image.jpg")
```
"""
    
    with open("DATASET_INFO.md", "w") as f:
        f.write(info_content)
    
    print("Created DATASET_INFO.md with usage information")

if __name__ == "__main__":
    print("Starting dataset downloads...")
    
    # Download DOTA8 first (smaller, for testing)
    download_dota8_dataset()
    
    # Ask user if they want to download full DOTA dataset
    print("\nDOTA8 download initiated.")
    response = input("Do you want to download the full DOTAv1 dataset? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        download_dota_full_dataset()
    else:
        print("Skipping full DOTAv1 dataset download.")
    
    # Check where datasets are located
    check_dataset_locations()
    
    # Create info file
    create_dataset_info()
    
    print("\nDataset download process completed!")
