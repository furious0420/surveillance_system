# HAL Surveillance System - Deep Learning Model

A focused deep learning system for EO (Electro-Optical) camera-based threat detection and surveillance, designed for HAL (Hindustan Aeronautics Limited) security applications.

## 🎯 Project Overview

This system performs real-time classification and identification of threats using EO camera video streams, with capabilities for:

- **3-Class Detection**: Human, Vehicle, and Weapon Detection
- **Threat Assessment**: Automated threat level grading (LOW, MEDIUM, HIGH, CRITICAL)
- **GPU Acceleration**: Optimized for NVIDIA RTX 3050 and CUDA training
- **ESP32-CAM Integration**: Real-time processing from ESP32 camera feeds

## 📁 Project Structure

```
EO_ML_MODEL/
├── src/
│   ├── data/                    # Data processing modules
│   │   ├── dataset_analyzer.py  # Dataset analysis utilities
│   │   ├── data_loader.py       # Custom data loaders
│   │   ├── preprocessing.py     # Image preprocessing
│   │   └── augmentation.py      # Data augmentation
│   ├── models/                  # Model architectures (to be created)
│   ├── training/                # Training scripts (to be created)
│   ├── inference/               # Inference pipeline (to be created)
│   └── utils/                   # Utility modules
│       ├── config_loader.py     # Configuration management
│       └── logger.py            # Logging utilities
├── animals/                     # Animals dataset (90 classes)
├── humans dataset/              # Human detection dataset
├── vehicles/                    # Vehicle detection dataset
├── config.yaml                  # Main configuration file
├── requirements.txt             # Python dependencies
├── test_setup.py               # Setup verification script
└── analyze_datasets.py         # Dataset analysis script
```

## 🚀 Quick Start

### 1. Verify Setup

First, test your environment setup:

```bash
python test_setup.py
```

This will check:
- Python version (3.8+ required)
- Project structure
- Dataset paths
- Required dependencies
- Configuration loading

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch & TorchVision
- OpenCV
- Pillow
- NumPy, Pandas
- Albumentations (for advanced augmentation)
- Ultralytics (for YOLO models)

### 3. Analyze Your Datasets

Run the dataset analysis to understand your data:

```bash
python analyze_datasets.py
```

This will:
- Analyze all three datasets (animals, humans, vehicles)
- Generate statistics and insights
- Create visualizations
- Provide recommendations
- Save results to `dataset_analysis_results.json`

## 📊 Dataset Information

### Animals Dataset
- **Format**: Image classification
- **Classes**: 90 different animal species
- **Structure**: `animals/animals/{class_name}/*.jpg`

### Humans Dataset
- **Format**: Binary classification
- **Classes**: Human (1) vs No Human (0)
- **Structure**: `humans dataset/human detection dataset/{0,1}/*.png`

### Vehicles Dataset
- **Format**: Object detection (YOLO format)
- **Variants**: Grayscale and Color versions
- **Formats**: YOLOv8, YOLOv9, COCO, TensorFlow
- **Structure**: `vehicles/{variant}/Vehicles_Detection.v9i.{format}/`

## ⚙️ Configuration

The system is configured through `config.yaml`:

```yaml
# Key configuration sections:
data:           # Dataset paths
model:          # Model architecture settings
training:       # Training parameters
inference:      # Inference settings
edge:          # Edge deployment options
```

## 🧠 Model Architecture

The system uses a unified approach combining:

1. **Classification Model**: EfficientNet-B3 backbone for multi-class classification
2. **Object Detection**: YOLOv8 for precise localization
3. **Threat Grading**: Custom threat assessment based on detected objects

### Threat Level Mapping

- **LOW**: Small animals (cats, dogs, birds)
- **MEDIUM**: Humans, medium animals
- **HIGH**: Large animals, vehicles
- **CRITICAL**: Dangerous animals (bears, lions, etc.)

## 🔄 Next Steps

After running the setup verification and dataset analysis:

1. **Model Development**: Create the classification and detection models
2. **Training Pipeline**: Implement training scripts with proper validation
3. **Inference System**: Build real-time inference pipeline
4. **Edge Optimization**: Convert models for edge deployment
5. **Integration**: Connect with ESP32-S3 camera system

## 📈 Features

### Data Processing
- ✅ Unified dataset loader for all three datasets
- ✅ Advanced image preprocessing and enhancement
- ✅ Comprehensive data augmentation pipeline
- ✅ Support for both classification and detection formats

### Model Capabilities
- 🔄 Multi-class classification (90+ classes)
- 🔄 Object detection with bounding boxes
- 🔄 Threat level assessment
- 🔄 Real-time inference pipeline

### Edge Deployment
- 🔄 TensorFlow Lite optimization
- 🔄 ONNX model conversion
- 🔄 Quantization for reduced model size
- 🔄 ESP32-S3 integration

## 🛠️ Development Status

- [x] Project setup and configuration
- [x] Data analysis and preprocessing pipeline
- [x] Multi-class classification model development
- [x] Object detection integration
- [x] Threat assessment system
- [ ] Model training and optimization
- [ ] Real-time inference pipeline
- [ ] Model evaluation and testing
- [ ] Edge deployment preparation

## 📝 Usage Examples

### Quick Start Commands

```bash
# 1. Test your setup
python test_setup.py

# 2. Analyze your datasets
python analyze_datasets.py

# 3. Train the model (dry run)
python train_model.py --dry-run --epochs 1

# 4. Run surveillance demo
python surveillance_demo.py --samples 3

# 5. Process single image
python demo_model.py --model models/best_model.pth --image path/to/image.jpg
```

### Python API Examples

```python
# Analyze Datasets
from src.data.dataset_analyzer import DatasetAnalyzer
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().config
analyzer = DatasetAnalyzer(config)
results = analyzer.analyze_all_datasets()

# Load Data
from src.data.data_loader import SurveillanceDataLoader
data_loader = SurveillanceDataLoader(config)
train_loader, val_loader, test_loader = data_loader.create_data_loaders()

# Object Detection
from src.models.detection_model import SurveillanceDetector
detector = SurveillanceDetector(model_size='m')
results = detector.detect(image)

# Threat Assessment
from src.models.threat_assessment import ThreatAssessmentModel
threat_assessor = ThreatAssessmentModel(config)
assessment = threat_assessor.assess_threat(detections)
```

## 🤝 Contributing

This is a specialized surveillance system for HAL. For questions or modifications, please refer to the project documentation or contact the development team.

## 📄 License

This project is developed for HAL (Hindustan Aeronautics Limited) surveillance applications.

---

**Ready to build your surveillance AI system!** 🚀

Start with `python test_setup.py` and then `python analyze_datasets.py` to begin your journey.
