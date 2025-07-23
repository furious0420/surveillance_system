#!/usr/bin/env python3
"""
Ultimate HAL Surveillance System - 5-Class Detection
Human, Vehicle, Weapon, UAV, and Animal Detection
Uses ALL datasets for maximum accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
import time
import json
import pickle
import numpy as np
from collections import defaultdict
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

print("🛡️  ULTIMATE HAL SURVEILLANCE SYSTEM")
print("5-Class Detection: Human, Vehicle, Weapon, UAV, Animal")
print("=" * 70)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")

class UltimateSurveillanceDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.data = []
        
        # 5-Class mapping
        self.class_to_idx = {
            'background': 0,
            'human': 1, 
            'vehicle': 2,
            'weapon': 3,
            'uav': 4,
            'animal': 5
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Advanced data augmentation for training
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.load_all_datasets()
        
    def load_all_datasets(self):
        print("📂 Loading ALL datasets for ultimate surveillance...")
        
        # 1. Load humans dataset
        self.load_humans_dataset()
        
        # 2. Load vehicles dataset  
        self.load_vehicles_dataset()
        
        # 3. Load weapons dataset
        self.load_weapons_dataset()
        
        # 4. Load UAV dataset
        self.load_uav_dataset()
        
        # 5. Load animals dataset
        self.load_animals_dataset()
        
        # Print final statistics
        self.print_dataset_stats()
        
    def load_humans_dataset(self):
        print("   Loading humans dataset...")
        humans_path = Path("humans dataset/human detection dataset")
        
        if humans_path.exists():
            # Background images (class 0)
            bg_dir = humans_path / "0"
            if bg_dir.exists():
                bg_images = list(bg_dir.glob("*.png")) + list(bg_dir.glob("*.jpg"))
                for img_file in bg_images:
                    self.data.append((str(img_file), 0))  # background
                print(f"     Background: {len(bg_images)} images")
            
            # Human images (class 1)
            human_dir = humans_path / "1"
            if human_dir.exists():
                human_images = list(human_dir.glob("*.png")) + list(human_dir.glob("*.jpg"))
                for img_file in human_images:
                    self.data.append((str(img_file), 1))  # human
                print(f"     Human: {len(human_images)} images")
    
    def load_vehicles_dataset(self):
        print("   Loading vehicles dataset...")
        vehicles_path = Path("vehicles")
        vehicle_count = 0
        
        if vehicles_path.exists():
            # Search through all subdirectories for vehicle images
            for img_file in vehicles_path.rglob("*.jpg"):
                if "labels" not in str(img_file) and "video" not in str(img_file).lower():
                    self.data.append((str(img_file), 2))  # vehicle
                    vehicle_count += 1
            
            for img_file in vehicles_path.rglob("*.png"):
                if "labels" not in str(img_file):
                    self.data.append((str(img_file), 2))  # vehicle
                    vehicle_count += 1
            
            print(f"     Vehicle: {vehicle_count} images")
    
    def load_weapons_dataset(self):
        print("   Loading weapons dataset...")
        weapons_path = Path("weapons/train-weapons_in_images/img")
        weapon_count = 0
        
        if weapons_path.exists():
            weapon_images = list(weapons_path.glob("*.jpg")) + list(weapons_path.glob("*.png"))
            for img_file in weapon_images:
                self.data.append((str(img_file), 3))  # weapon
                weapon_count += 1
            
            print(f"     Weapon: {weapon_count} images")
    
    def load_uav_dataset(self):
        print("   Loading UAV/Drone dataset...")
        uav_path = Path("uav/drone_dataset_yolo/dataset_txt")
        uav_count = 0
        
        if uav_path.exists():
            uav_images = list(uav_path.glob("*.jpg")) + list(uav_path.glob("*.png"))
            for img_file in uav_images:
                self.data.append((str(img_file), 4))  # uav
                uav_count += 1
            
            print(f"     UAV/Drone: {uav_count} images")
    
    def load_animals_dataset(self):
        print("   Loading animals dataset...")
        animals_path = Path("animals/animals")
        animal_count = 0
        
        if animals_path.exists():
            # Get all animal class directories
            class_dirs = [d for d in animals_path.iterdir() if d.is_dir()]
            
            for class_dir in class_dirs:
                # Get all images in this animal class
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                
                for img_file in image_files:
                    self.data.append((str(img_file), 5))  # animal
                    animal_count += 1
            
            print(f"     Animal: {animal_count} images")
    
    def print_dataset_stats(self):
        class_counts = defaultdict(int)
        for _, label in self.data:
            class_name = self.idx_to_class[label]
            class_counts[class_name] += 1
        
        print(f"\n✅ ULTIMATE DATASET LOADED: {len(self.data)} total images")
        print("📊 Class Distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name.capitalize()}: {count} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            # Return dummy data if image fails to load
            return torch.randn(3, 224, 224), 0

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
        elif backbone == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=True)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.backbone.classifier[1].in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:  # resnet18 for faster training
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.backbone.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
    def forward(self, x):
        return self.backbone(x)

def evaluate_model_comprehensive(model, data_loader, device, class_names):
    """
    Comprehensive model evaluation with detailed metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("🔍 Running comprehensive evaluation...")

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions, average=None)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Classification Report
    class_report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def print_evaluation_results(eval_results, class_names):
    """
    Print comprehensive evaluation results
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("="*80)

    # Overall Accuracy
    print(f"🎯 Overall Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")

    # Per-class metrics
    print(f"\n📈 Per-Class Performance:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)

    for i, class_name in enumerate(class_names):
        precision = eval_results['precision'][i]
        recall = eval_results['recall'][i]
        f1 = eval_results['f1_score'][i]
        support = eval_results['support'][i]

        print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")

    # Macro and weighted averages
    report = eval_results['classification_report']
    print(f"\n📊 Summary Metrics:")
    print(f"   Macro Avg    - Precision: {report['macro avg']['precision']:.4f}, Recall: {report['macro avg']['recall']:.4f}, F1: {report['macro avg']['f1-score']:.4f}")
    print(f"   Weighted Avg - Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1: {report['weighted avg']['f1-score']:.4f}")

def plot_confusion_matrix(cm, class_names, save_path="models/confusion_matrix.png"):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - HAL Surveillance Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved to: {save_path}")

def save_model_as_pkl(model, model_info, save_path="models/hal_surveillance_model.pkl"):
    """
    Save the complete model as a pickle file
    """
    # Prepare model package
    model_package = {
        'model': model,
        'model_state_dict': model.state_dict(),
        'model_info': model_info,
        'class_names': model_info.get('class_names', []),
        'class_to_idx': model_info.get('class_to_idx', {}),
        'idx_to_class': model_info.get('idx_to_class', {}),
        'backbone': model_info.get('backbone', 'resnet50'),
        'num_classes': model_info.get('num_classes', 6),
        'input_size': (224, 224),
        'normalization_params': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    # Save as pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"💾 Model saved as pickle file: {save_path}")
    print(f"   File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

    return save_path

def train_ultimate_model():
    print("\n🚀 Starting ULTIMATE HAL surveillance training...")

    # Create ultimate dataset
    ultimate_dataset = UltimateSurveillanceDataset(mode='train')
    if len(ultimate_dataset) == 0:
        print("❌ No data found")
        return False

    # Split data: 80% train, 20% validation (as requested)
    total_size = len(ultimate_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        ultimate_dataset, [train_size, val_size]
    )

    print(f"\n✅ Dataset split (80:20 Train:Validation):")
    print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"   Validation: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")

    # Create data loaders with optimal batch size
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Create ultimate model
    backbone = 'resnet50' if torch.cuda.is_available() else 'resnet18'
    model = UltimateHALModel(num_classes=6, backbone=backbone).to(device)

    print(f"✅ Ultimate model created: {backbone}")

    # Advanced training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    # Training parameters
    epochs = 100  # Ultimate training
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    start_epoch = 0

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # 🚀 RESUME FUNCTIONALITY - Check for existing checkpoint
    checkpoint_path = "models/ultimate_hal_surveillance.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found existing checkpoint - RESUMING training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            history = checkpoint.get('history', history)
            patience_counter = 0  # Reset patience counter

            print(f"✅ Resumed from Epoch {checkpoint['epoch']}")
            print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
            print(f"   Continuing from Epoch {start_epoch}")
            print(f"   Training History: {len(history['train_loss'])} epochs loaded")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Starting fresh training...")
            start_epoch = 0
    else:
        print("\n🆕 No checkpoint found - Starting fresh training...")

    print(f"🚀 Starting {epochs} epochs of ULTIMATE training...")
    if start_epoch > 0:
        print(f"   Resuming from Epoch {start_epoch} (skipping {start_epoch} completed epochs)")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'num_classes': 6,
                'class_to_idx': ultimate_dataset.class_to_idx,
                'backbone': backbone,
                'history': history
            }, "models/ultimate_hal_surveillance.pth")

            print(f"  ✅ NEW BEST MODEL! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping triggered after {patience} epochs without improvement")
            break

        print()

    total_time = time.time() - start_time

    # Load best model for final evaluation
    print("📂 Loading best model for comprehensive evaluation...")
    checkpoint = torch.load("models/ultimate_hal_surveillance.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get class names for evaluation
    class_names = list(ultimate_dataset.idx_to_class.values())

    # Comprehensive evaluation on validation set
    print("🧪 Running comprehensive evaluation on validation set...")
    val_eval_results = evaluate_model_comprehensive(model, val_loader, device, class_names)

    # Print detailed results
    print_evaluation_results(val_eval_results, class_names)

    # Plot and save confusion matrix
    plot_confusion_matrix(val_eval_results['confusion_matrix'], class_names)

    # Prepare model info for saving
    model_info = {
        'backbone': backbone,
        'num_classes': 6,
        'class_names': class_names,
        'class_to_idx': ultimate_dataset.class_to_idx,
        'idx_to_class': ultimate_dataset.idx_to_class,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_eval_results['accuracy'],
        'training_time_minutes': total_time/60,
        'total_samples': len(ultimate_dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'device_used': str(device),
        'evaluation_results': val_eval_results
    }

    # Save model as pickle file
    pkl_path = save_model_as_pkl(model, model_info)

    # Save detailed evaluation results
    evaluation_report = {
        'model_info': model_info,
        'training_history': history,
        'evaluation_metrics': {
            'accuracy': float(val_eval_results['accuracy']),
            'per_class_precision': val_eval_results['precision'].tolist(),
            'per_class_recall': val_eval_results['recall'].tolist(),
            'per_class_f1': val_eval_results['f1_score'].tolist(),
            'per_class_support': val_eval_results['support'].tolist(),
            'confusion_matrix': val_eval_results['confusion_matrix'].tolist(),
            'classification_report': val_eval_results['classification_report']
        },
        'class_mapping': {
            'class_names': class_names,
            'class_to_idx': ultimate_dataset.class_to_idx,
            'idx_to_class': ultimate_dataset.idx_to_class
        }
    }

    # Save training history and evaluation
    with open("models/ultimate_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open("models/evaluation_report.json", "w") as f:
        json.dump(evaluation_report, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 ULTIMATE HAL SURVEILLANCE TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Final Validation Accuracy: {val_eval_results['accuracy']:.4f} ({val_eval_results['accuracy']*100:.2f}%)")
    print(f"   Total Training Time: {total_time/60:.1f} minutes")
    print(f"   Device Used: {device}")
    print(f"   Model Architecture: {backbone}")
    print(f"   Total Samples: {len(ultimate_dataset)}")
    print(f"   Training Split: {len(train_dataset)} samples (80%)")
    print(f"   Validation Split: {len(val_dataset)} samples (20%)")

    print(f"\n📁 Generated Files:")
    print(f"   🔥 PyTorch Model: models/ultimate_hal_surveillance.pth")
    print(f"   📦 Pickle Model: {pkl_path}")
    print(f"   📊 Training History: models/ultimate_training_history.json")
    print(f"   📈 Evaluation Report: models/evaluation_report.json")
    print(f"   🎯 Confusion Matrix: models/confusion_matrix.png")

    print(f"\n📋 Classification Report Summary:")
    report = val_eval_results['classification_report']
    print(f"   Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"   Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}")
    print(f"   Total Samples Evaluated: {sum(val_eval_results['support'])}")

    # Show per-class performance summary
    print(f"\n🎯 Per-Class Performance Summary:")
    for i, class_name in enumerate(class_names):
        f1 = val_eval_results['f1_score'][i]
        support = val_eval_results['support'][i]
        print(f"   {class_name.capitalize():<12}: F1={f1:.3f} (n={support})")

    print(f"\n💾 Model Usage Instructions:")
    print(f"   To load the pickle model:")
    print(f"   ```python")
    print(f"   import pickle")
    print(f"   with open('{pkl_path}', 'rb') as f:")
    print(f"       model_package = pickle.load(f)")
    print(f"   model = model_package['model']")
    print(f"   class_names = model_package['class_names']")
    print(f"   ```")

    print(f"\n🎯 6-Class Mapping:")
    print(f"   0: Background (LOW threat)")
    print(f"   1: Human (MEDIUM threat)")
    print(f"   2: Vehicle (HIGH threat)")
    print(f"   3: Weapon (CRITICAL threat)")
    print(f"   4: UAV/Drone (CRITICAL threat)")
    print(f"   5: Animal (LOW-MEDIUM threat)")

    print(f"\n⚠️  Threat Assessment:")
    print(f"   🟢 LOW: Background, Animals")
    print(f"   🟡 MEDIUM: Humans")
    print(f"   🟠 HIGH: Vehicles")
    print(f"   🔴 CRITICAL: Weapons, UAVs")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Test model with ultimate_test.py")
    print(f"   2. Deploy to ESP32-CAM integration")
    print(f"   3. Real-time surveillance monitoring")
    print(f"   4. Military-grade threat detection")

    return True

if __name__ == "__main__":
    try:
        success = train_ultimate_model()
        if success:
            print("\n🛡️ ULTIMATE HAL SURVEILLANCE SYSTEM READY FOR DEPLOYMENT!")
        else:
            print("❌ Training failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
