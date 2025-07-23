#!/usr/bin/env python3
"""
Comprehensive automated script for YOLOv11s-obb aerial view detection
Handles model download, dataset setup, training, and evaluation
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class AerialViewTrainer:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.model_path = self.base_dir / "yolo11s-obb.pt"
        self.results_dir = self.base_dir / "training_results"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

        # Create results directory
        self.results_dir.mkdir(exist_ok=True)

        # Training configurations
        self.training_configs = {
            'quick_test': {
                'epochs': 10,
                'imgsz': 640,
                'batch': 8,
                'dataset': 'dota8.yaml',
                'description': 'Quick test training on DOTA8'
            },
            'full_training': {
                'epochs': 100,
                'imgsz': 1024,
                'batch': 4,
                'dataset': 'DOTAv1.yaml',
                'description': 'Full training on DOTAv1'
            },
            'medium_training': {
                'epochs': 50,
                'imgsz': 1024,
                'batch': 6,
                'dataset': 'DOTAv1.yaml',
                'description': 'Medium training on DOTAv1'
            }
        }

    def log_message(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

        # Also save to log file
        log_file = self.results_dir / "training_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")

    def setup_model(self):
        """Download and setup YOLOv11s-obb model"""
        self.log_message("Setting up YOLOv11s-obb model...")

        try:
            # Load model (will download if not exists)
            self.model = YOLO("yolo11s-obb.pt")
            self.log_message(f"Model loaded successfully from: {self.model.ckpt_path}")
            self.log_message(f"Model type: {self.model.task}")
            self.log_message(f"Using device: {self.device}")

            return True
        except Exception as e:
            self.log_message(f"Error setting up model: {e}", "ERROR")
            return False

    def verify_datasets(self):
        """Verify that datasets are available"""
        self.log_message("Verifying datasets...")

        datasets_status = {}

        for config_name, config in self.training_configs.items():
            dataset_name = config['dataset']
            try:
                # Try to validate with the dataset
                result = self.model.val(data=dataset_name, verbose=False)
                datasets_status[dataset_name] = True
                self.log_message(f"Dataset {dataset_name} verified successfully")
            except Exception as e:
                datasets_status[dataset_name] = False
                self.log_message(f"Dataset {dataset_name} not available: {e}", "WARNING")

        return datasets_status

    def train_model(self, config_name='quick_test'):
        """Train the model with specified configuration"""
        if config_name not in self.training_configs:
            self.log_message(f"Unknown config: {config_name}", "ERROR")
            return False

        config = self.training_configs[config_name]
        self.log_message(f"Starting training: {config['description']}")
        self.log_message(f"Configuration: {config}")

        try:
            # Create training directory
            training_dir = self.results_dir / f"training_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            training_dir.mkdir(exist_ok=True)

            # Start training
            results = self.model.train(
                data=config['dataset'],
                epochs=config['epochs'],
                imgsz=config['imgsz'],
                batch=config['batch'],
                device=self.device,
                project=str(training_dir),
                name='train',
                save=True,
                save_period=10,  # Save every 10 epochs
                plots=True,
                verbose=True
            )

            self.log_message(f"Training completed successfully!")
            self.log_message(f"Results saved to: {training_dir}")

            # Save training summary
            summary = {
                'config_name': config_name,
                'config': config,
                'training_dir': str(training_dir),
                'device': self.device,
                'completion_time': datetime.now().isoformat(),
                'results': str(results) if results else None
            }

            summary_file = training_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            return training_dir

        except Exception as e:
            self.log_message(f"Training failed: {e}", "ERROR")
            return False

    def evaluate_model(self, model_path=None, dataset='dota8.yaml'):
        """Evaluate model performance"""
        self.log_message(f"Evaluating model on {dataset}...")

        try:
            # Use custom model if provided, otherwise use current model
            eval_model = YOLO(model_path) if model_path else self.model

            # Run validation
            results = eval_model.val(data=dataset, verbose=True)

            self.log_message("Evaluation completed successfully!")

            # Extract key metrics
            metrics = {
                'dataset': dataset,
                'model_path': str(model_path) if model_path else 'current_model',
                'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else None,
                'mAP50_95': float(results.box.map) if hasattr(results.box, 'map') else None,
                'evaluation_time': datetime.now().isoformat()
            }

            # Save evaluation results
            eval_file = self.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            self.log_message(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            self.log_message(f"Evaluation failed: {e}", "ERROR")
            return None

    def create_inference_script(self):
        """Create a script for running inference on new images"""
        inference_script = '''#!/usr/bin/env python3
"""
Inference script for YOLOv11s-obb aerial view detection
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def run_inference(model_path, image_path, output_dir="inference_results"):
    """Run inference on aerial images"""
    print(f"Loading model from: {model_path}")
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

    for i, img_path in enumerate(images):
        print(f"Processing {i+1}/{len(images)}: {img_path}")

        # Run inference
        results = model(str(img_path))

        # Save results
        for j, result in enumerate(results):
            # Save annotated image
            output_path = Path(output_dir) / f"{Path(img_path).stem}_result.jpg"
            result.save(str(output_path))

            # Print detection info
            if hasattr(result, 'obb') and result.obb is not None:
                num_detections = len(result.obb.cls)
                print(f"  Found {num_detections} objects")

                # Print class names and confidence scores
                for k in range(num_detections):
                    class_id = int(result.obb.cls[k])
                    confidence = float(result.obb.conf[k])
                    class_name = result.names[class_id]
                    print(f"    {class_name}: {confidence:.3f}")
            else:
                print("  No objects detected")

    print(f"Inference completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path> [output_dir]")
        print("Example: python inference.py yolo11s-obb.pt aerial_image.jpg")
        print("Example: python inference.py best.pt images_folder/ results/")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "inference_results"

    run_inference(model_path, image_path, output_dir)
'''

        inference_file = self.base_dir / "inference.py"
        with open(inference_file, 'w') as f:
            f.write(inference_script)

        self.log_message(f"Inference script created: {inference_file}")
        return inference_file

    def run_complete_workflow(self, training_mode='quick_test'):
        """Run the complete automated workflow"""
        self.log_message("=" * 60)
        self.log_message("STARTING AUTOMATED YOLO11S-OBB AERIAL VIEW WORKFLOW")
        self.log_message("=" * 60)

        # Step 1: Setup model
        if not self.setup_model():
            self.log_message("Failed to setup model. Exiting.", "ERROR")
            return False

        # Step 2: Verify datasets
        datasets_status = self.verify_datasets()
        available_datasets = [k for k, v in datasets_status.items() if v]

        if not available_datasets:
            self.log_message("No datasets available. Exiting.", "ERROR")
            return False

        self.log_message(f"Available datasets: {available_datasets}")

        # Step 3: Initial evaluation (baseline)
        self.log_message("Running baseline evaluation...")
        baseline_metrics = self.evaluate_model(dataset='dota8.yaml')

        # Step 4: Training
        if training_mode in self.training_configs:
            dataset_for_training = self.training_configs[training_mode]['dataset']
            if dataset_for_training in available_datasets:
                self.log_message(f"Starting training with mode: {training_mode}")
                training_dir = self.train_model(training_mode)

                if training_dir:
                    # Step 5: Post-training evaluation
                    best_model_path = training_dir / "train" / "weights" / "best.pt"
                    if best_model_path.exists():
                        self.log_message("Running post-training evaluation...")
                        final_metrics = self.evaluate_model(str(best_model_path), dataset='dota8.yaml')

                        # Compare metrics
                        if baseline_metrics and final_metrics:
                            improvement = {}
                            if baseline_metrics.get('mAP50') and final_metrics.get('mAP50'):
                                improvement['mAP50'] = final_metrics['mAP50'] - baseline_metrics['mAP50']
                            if baseline_metrics.get('mAP50_95') and final_metrics.get('mAP50_95'):
                                improvement['mAP50_95'] = final_metrics['mAP50_95'] - baseline_metrics['mAP50_95']

                            self.log_message(f"Performance improvement: {improvement}")
                    else:
                        self.log_message("Best model not found after training", "WARNING")
                else:
                    self.log_message("Training failed", "ERROR")
            else:
                self.log_message(f"Dataset {dataset_for_training} not available for training", "ERROR")
        else:
            self.log_message(f"Invalid training mode: {training_mode}", "ERROR")

        # Step 6: Create inference script
        self.create_inference_script()

        # Step 7: Generate final report
        self.generate_final_report()

        self.log_message("=" * 60)
        self.log_message("AUTOMATED WORKFLOW COMPLETED")
        self.log_message("=" * 60)

        return True

    def generate_final_report(self):
        """Generate a final report of the workflow"""
        report = f"""
# YOLOv11s-OBB Aerial View Detection - Training Report

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Setup Information
- Model: YOLOv11s-obb
- Device: {self.device}
- Base Directory: {self.base_dir}
- Results Directory: {self.results_dir}

## Available Training Configurations
"""

        for config_name, config in self.training_configs.items():
            report += f"### {config_name}\n"
            report += f"- Description: {config['description']}\n"
            report += f"- Epochs: {config['epochs']}\n"
            report += f"- Image Size: {config['imgsz']}\n"
            report += f"- Batch Size: {config['batch']}\n"
            report += f"- Dataset: {config['dataset']}\n\n"

        report += """
## Usage Instructions

### Training
```bash
python download_model.py --mode train --config quick_test
python download_model.py --mode train --config medium_training
python download_model.py --mode train --config full_training
```

### Inference
```bash
python inference.py yolo11s-obb.pt aerial_image.jpg
python inference.py best.pt images_folder/ results/
```

### Evaluation
```bash
python download_model.py --mode evaluate --model best.pt --dataset dota8.yaml
```

## Files Created
- `yolo11s-obb.pt` - Pre-trained model
- `inference.py` - Inference script
- `training_results/` - Training outputs and logs
- `DATASET_INFO.md` - Dataset information
- `training_log.txt` - Detailed logs

## Next Steps
1. Use the trained model for inference on your aerial images
2. Fine-tune hyperparameters if needed
3. Collect more domain-specific data for better performance
4. Export model to different formats (ONNX, TensorRT) for deployment
"""

        report_file = self.results_dir / "FINAL_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)

        self.log_message(f"Final report generated: {report_file}")

def main():
    """Main function to handle command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv11s-OBB Aerial View Detection Automation')
    parser.add_argument('--mode', choices=['full', 'train', 'evaluate', 'inference'],
                       default='full', help='Operation mode')
    parser.add_argument('--config', choices=['quick_test', 'medium_training', 'full_training'],
                       default='quick_test', help='Training configuration')
    parser.add_argument('--model', type=str, help='Model path for evaluation')
    parser.add_argument('--dataset', type=str, default='dota8.yaml', help='Dataset for evaluation')
    parser.add_argument('--image', type=str, help='Image path for inference')

    args = parser.parse_args()

    # Initialize trainer
    trainer = AerialViewTrainer()

    if args.mode == 'full':
        # Run complete workflow
        trainer.run_complete_workflow(args.config)
    elif args.mode == 'train':
        # Setup and train only
        if trainer.setup_model():
            trainer.train_model(args.config)
    elif args.mode == 'evaluate':
        # Evaluate only
        if trainer.setup_model():
            trainer.evaluate_model(args.model, args.dataset)
    elif args.mode == 'inference':
        # Create inference script only
        trainer.create_inference_script()
        if args.image:
            print(f"Run: python inference.py yolo11s-obb.pt {args.image}")

if __name__ == "__main__":
    main()
