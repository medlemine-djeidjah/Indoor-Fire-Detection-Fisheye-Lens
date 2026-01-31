import os
from ultralytics import YOLO

# 1. Configuration
# Path to the data.yaml file provided in the dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets/Indoor-Fire-Smoke/data.yaml"))

# 2. Load Model
# We use the 'yolov8n.pt' which is the 'nano' version of YOLOv8.
# It is fast, lightweight, and perfect for a simple demonstration.
# The model will automatically download pre-trained weights from COCO.
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

# 3. Train the Model
# - data: path to the yaml file containing dataset info (train/val paths, classes)
# - epochs: We increase this to 50 for a real training run. 5 was too few for convergence.
# - close_mosaic: We can enable this to turn off mosaic augmentation in the last 10 epochs (YOLOv8 best practice).
print(f"Starting improved training for 50 epochs using dataset config at: {DATA_YAML_PATH}")
results = model.train(
    data=DATA_YAML_PATH,
    epochs=50,
    imgsz=640,
    plots=True,
    patience=10,  # Stop early if no improvement for 10 epochs
    name='fire_smoke_detection_v2'
)

print("\nTraining complete!")
print(f"Model weights saved to: {results.save_dir}")

# 4. Optional: Export the model
# You can export it to other formats like ONNX, OpenVINO, CoreML etc.
# path = model.export(format='onnx')
# print(f"Model exported to: {path}")
