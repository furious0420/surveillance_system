
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
