#!/usr/bin/env python3
"""
Script to find where Ultralytics datasets are stored
"""

import os
import sys
from pathlib import Path
import yaml

def find_ultralytics_datasets():
    """Find where Ultralytics datasets are stored"""
    print("Searching for Ultralytics datasets...")
    
    # Common locations
    home = Path.home()
    possible_locations = [
        home / "datasets",
        home / ".ultralytics" / "datasets",
        Path.cwd() / "datasets",
        Path("datasets"),
        home / "ultralytics" / "datasets",
        Path.cwd() / "ultralytics" / "datasets",
    ]
    
    # Check YOLO configs
    try:
        from ultralytics.cfg import get_cfg, DATASETS_DIR
        print(f"DATASETS_DIR from ultralytics config: {DATASETS_DIR}")
        possible_locations.append(Path(DATASETS_DIR))
    except:
        print("Could not import ultralytics config")
    
    # Check each location
    found_datasets = []
    for location in possible_locations:
        if location.exists():
            print(f"\nFound datasets directory: {location}")
            try:
                contents = list(location.iterdir())
                if contents:
                    print(f"  Contents: {[p.name for p in contents]}")
                    
                    # Check for DOTA datasets specifically
                    dota_dirs = [p for p in contents if p.name.lower().startswith('dota')]
                    for dota_dir in dota_dirs:
                        print(f"\n  DOTA dataset found: {dota_dir}")
                        try:
                            dota_contents = list(dota_dir.iterdir())
                            print(f"    Contents: {[p.name for p in dota_contents]}")
                            found_datasets.append(dota_dir)
                        except Exception as e:
                            print(f"    Error listing contents: {e}")
                else:
                    print("  (empty)")
            except Exception as e:
                print(f"  Error listing contents: {e}")
    
    # Try to find dataset YAML files
    print("\nSearching for dataset YAML files...")
    try:
        from ultralytics.utils.downloads import find_dir
        datasets_dir = find_dir()
        print(f"Ultralytics datasets directory: {datasets_dir}")
        
        # Look for YAML files
        yaml_files = list(Path(datasets_dir).glob("**/*.yaml"))
        print(f"Found {len(yaml_files)} YAML files:")
        for yaml_file in yaml_files:
            print(f"  {yaml_file}")
            
            # Try to read YAML content
            try:
                with open(yaml_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    if yaml_content and isinstance(yaml_content, dict):
                        if 'path' in yaml_content:
                            print(f"    Dataset path: {yaml_content['path']}")
                        if 'names' in yaml_content:
                            print(f"    Classes: {yaml_content['names']}")
            except Exception as e:
                print(f"    Error reading YAML: {e}")
    except Exception as e:
        print(f"Error finding dataset directory: {e}")
    
    return found_datasets

if __name__ == "__main__":
    datasets = find_ultralytics_datasets()
    
    if datasets:
        print(f"\nFound {len(datasets)} DOTA datasets")
    else:
        print("\nNo DOTA datasets found in common locations")
        print("The datasets may be in a different location or not yet downloaded")
        
    # Try to get dataset info from YOLO
    print("\nTrying to get dataset info from YOLO...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11s-obb.pt")
        
        # Try to get dataset info
        print("Dataset info for dota8.yaml:")
        try:
            result = model.val(data="dota8.yaml", verbose=False)
            print("  Validation successful")
        except Exception as e:
            print(f"  Error validating dota8: {e}")
            
        print("\nDataset info for DOTAv1.yaml:")
        try:
            result = model.val(data="DOTAv1.yaml", verbose=False)
            print("  Validation successful")
        except Exception as e:
            print(f"  Error validating DOTAv1: {e}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
