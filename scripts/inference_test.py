import os
import glob
from ultralytics import YOLO
import cv2

# 1. Load the trained model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights are usually in 'runs/detect/fire_smoke_detection_v2/weights/best.pt'
# However, if run from where this script is, it's relative to CWD.
# We'll check both relative to script and relative to current CWD.
TRAINED_WEIGHTS = os.path.join(SCRIPT_DIR, 'runs/detect/fire_smoke_detection_v2/weights/best.pt')

if not os.path.exists(TRAINED_WEIGHTS):
    # Fallback to V1
    V1_WEIGHTS = os.path.join(SCRIPT_DIR, 'runs/detect/fire_smoke_detection/weights/best.pt')
    if os.path.exists(V1_WEIGHTS):
        print(f"V2 weights not found, using V1 weights from: {V1_WEIGHTS}")
        model = YOLO(V1_WEIGHTS)
    else:
        print(f"Warning: No trained weights found at {TRAINED_WEIGHTS}. Defaulting to pre-trained yolov8n.pt")
        model = YOLO('yolov8n.pt')
else:
    print(f"Loading improved trained weights from: {TRAINED_WEIGHTS}")
    model = YOLO(TRAINED_WEIGHTS)

# 2. Get some test images
TEST_IMAGES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets/Indoor-Fire-Smoke/test/images"))
image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))[:3] # Take 3 images for test

if not image_files:
    print(f"No test images found in {TEST_IMAGES_DIR}")
else:
    print(f"Running inference on {len(image_files)} test images...")
    
    # 3. Run Inference
    # - conf: lower this to 0.25 to catch more potential detections
    # - save: save the results to disk
    results = model.predict(image_files, conf=0.25, save=True, name='inference_results_v2')

    print("\nInference complete!")
    # results[0].save_dir is where the predicted images are saved
    print(f"Results saved to: {results[0].save_dir}")
