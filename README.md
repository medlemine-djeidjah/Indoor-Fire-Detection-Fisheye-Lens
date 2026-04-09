# Indoor Fire Detection - Fisheye Lens

Fire and smoke detection system optimized for fisheye lens cameras using YOLO models.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
  - [Available Datasets](#available-datasets)
  - [Download Links](#download-links)
  - [Dataset Placement](#dataset-placement)
  - [Expected Directory Structure](#expected-directory-structure)
- [Notebooks](#notebooks)
  - [YOLOv12 Training Notebook](#yolov12-training-notebook)
  - [Video Inference Notebook](#video-inference-notebook)
  - [Updating Paths in Notebooks](#updating-paths-in-notebooks)
- [Scripts](#scripts)
  - [train_model.py](#train_modelpy)
  - [finetune_yolo_augmented.py](#finetune_yolo_augmentedpy)
  - [augment_dataset.py](#augment_datasetpy)
  - [inference_test.py](#inference_testpy)
- [GUI Application](#gui-application)
- [Training Workflow](#training-workflow)
- [License](#license)

---

## Project Structure

```
Indoor-Fire-Detection-Fisheye-Lens/
├── datasets/                       # Place all datasets here (gitignored)
│   ├── Indoor-Fire-Smoke/          # Indoor fire/smoke dataset
│   ├── YOLOV12-DATASET/            # YOLOv12 training dataset
│   └── ...
├── notebooks/
│   ├── yolov8/
│   │   ├── smoke-fire-detection-yolo-v8.ipynb  # Training notebook
│   │   └── inference-yolo-v8.ipynb                             # Video inference notebook
│   └── yolov12/
│       ├── smoke-fire-detection-yolo-v12.ipynb  # Training notebook
│       └── inference-yolo-v12.ipynb                             # Video inference notebook
├── scripts/
│   ├── train_model.py              # Basic YOLOv8 training
│   ├── finetune_yolo_augmented.py  # Advanced training with augmentation
│   ├── augment_dataset.py          # Dataset augmentation utilities
│   └── inference_test.py           # Run inference on test images
├── gui/
│   └── app.py                      # Streamlit web application
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/medlemine-djeidjah/Indoor-Fire-Detection-Fisheye-Lens
   cd Indoor-Fire-Detection-Fisheye-Lens
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python torch torchvision streamlit pillow numpy
   ```

---

## Datasets

### Available Datasets

| Dataset | Description | Classes | Use Case |
|---------|-------------|---------|----------|
| **YOLOV12-DATASET** | Large fire/smoke dataset for YOLOv12 training | smoke, fire | Primary training |
| **Indoor-Fire-Smoke** | Indoor fire and smoke images | fire, smoke | Indoor-specific training |
| **Indoor-Outdoor-Dataset** | Mixed indoor/outdoor fire scenes | fire, smoke | Diverse training |
| **Fisheye-Lens-Images** | Fisheye distorted images | - | Fisheye augmentation |
| **Building-Out** | Building fire scenarios | fire, smoke | Building-specific training |

### Download Links

| Dataset | Link |
|---------|------|
| YOLOV12-DATASET | [Download](https://storage.googleapis.com/kaggle-data-sets/6556263/10592956/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260126%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260126T082828Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=56ba11358946f2ac46fdba58e9e9a5f05b59c50aaa29e2f839edd3b8a1376e4155cfcbcf1e91ebb4c5a39bf8b94cac8eef13e383a29c0a9ac91c72ecdd2bef641a8d522ae7293cdf742a947e6f3200fe7609930cedb277d8c2d6ae1c776d70f2649c088e3c66679ed2db67cfdd7f85d6580bb8f5382d6ed1a649b85092b8bb76ffb4de2fe1c88f27fd5b37ba677c1d6ba2789e7db4e34c7535a7dabcbee34ea4678ad8f43a2121b4e560c5eaecc7c63687f4bcfea44211a5cb049100d8156ec15e529e1c77c2967f52765f909813cb6146017e0c07fc7b0e8973ae3d2d8d1108e2fef069530f515c96f4cb46d635d752ccf999540cd17566fd2b480b9ec7f4a2) |
| Indoor-Fire-Smoke | [Download](https://zenodo.org/records/15826133/files/Indoor%20Fire%20Smoke.zip?download=1) |
| Indoor-Outdoor-Dataset | [Download](https://storage.googleapis.com/kaggle-data-sets/3652173/6343158/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260121%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260121T185642Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4492d0e7348969d532f495ee398adf07cd86f392d8b379e60577eb301ea763ce021345cf7dfb50e037ca4b74e01634b8fb5fb815ca7f1c54b5de405bd43e516841fbf0234c518df2f0fbd2b23482c463bd4f5bf1be0a867bd86fe73923790665e71351239b39ff952e550e8d073d1b5d40c3f23f5d98be0ed1872fd0d2eaa4eb42296366fff6c075fda71bd605ed5a78e07c2a13d57c04ee603bee52a1d73d7d8b6a9e3c95e6a181e5962e5b790ff463db16c0a1d3215c5b2625337436cfb001e126d0879c2511631b5ad578329af59c6d7953a33570127b1a90accddd12b33e466b237f1f4c2d81fa1d9c41a00f011140f7bee78306d27927000506d1be0f47) |
| Fisheye-Lens-Images | [Download](https://drive.google.com/file/d/1yq2YrJCD3dhzghrJEZbnMklXvqwiUwil/view?usp=sharing) |
| Building-Out | [Download](https://zenodo.org/records/15187630/files/Building_Out.zip?download=1) |

### Dataset Placement

All datasets must be placed in the `datasets/` directory. This directory is gitignored to prevent uploading large files to GitHub.

**Steps:**
1. Download the dataset ZIP file
2. Extract it to the `datasets/` folder
3. Rename the folder if necessary to match the expected name (no spaces)

**Example:**
```bash
# After downloading Indoor Fire Smoke.zip
unzip "Indoor Fire Smoke.zip" -d datasets/
mv "datasets/Indoor Fire Smoke" datasets/Indoor-Fire-Smoke
```

### Expected Directory Structure

After placing datasets, your `datasets/` folder should look like this:

```
datasets/
├── Indoor-Fire-Smoke/
│   ├── data.yaml              # Dataset configuration file
│   ├── train/
│   │   ├── images/            # Training images (.jpg, .png)
│   │   └── labels/            # YOLO format labels (.txt)
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── YOLOV12-DATASET/
│   ├── data.yaml
│   └── data/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
└── .gitkeep
```

**Important:** The `data.yaml` file in each dataset defines:
- Path to train/val/test splits
- Number of classes (`nc`)
- Class names (`names`)

Example `data.yaml`:
```yaml
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['fire', 'smoke']
```

---

## Notebooks

### YOLOv12 Training Notebook

**File:** `notebooks/yolov12/smoke-fire-detection-yolo-v12.ipynb`

This notebook handles:
- Data cleaning and validation
- Visualizing training samples
- Training YOLOv12 model
- Model validation and export

#### Updating Paths in This Notebook

The notebook uses relative paths. You need to either:

**Option A: Copy dataset to notebook directory**
```bash
cp -r datasets/YOLOV12-DATASET notebooks/yolov12/
```

**Option B: Update paths in the notebook cells**

Find and update these variables in the notebook:

| Cell | Original Path | Update To |
|------|---------------|-----------|
| Cell 2 | `ROOT = "./YOLOV12-DATASET/data"` | `ROOT = "../../datasets/YOLOV12-DATASET/data"` |
| Cell 2 | `BAD_IMAGES_DIR = "./working/bad_images"` | Keep as-is (creates local working dir) |
| Cell 6 | `train_img_path = "./YOLOV12-DATASET/data/train/images"` | `train_img_path = "../../datasets/YOLOV12-DATASET/data/train/images"` |
| Cell 6 | `train_label_path = "./YOLOV12-DATASET/data/train/labels"` | `train_label_path = "../../datasets/YOLOV12-DATASET/data/train/labels"` |
| Cell 10 | `data="./YOLOV12-DATASET/data.yaml"` | `data="../../datasets/YOLOV12-DATASET/data.yaml"` |

**Note:** After updating, also update the `data.yaml` file paths to be absolute or relative to where training runs.

---

### Video Inference Notebook

**File:** `notebooks/yolov12/app.ipynb`

This notebook performs video inference using a trained model.

#### Updating Paths in This Notebook

Update these variables in the first cell:

```python
# Model path - update to your trained model location
MODEL_PATH = "../../gui/models/your_trained_model.pt"
# Or use absolute path:
MODEL_PATH = "/path/to/your/trained/model.pt"

# Input video - update to your video file
INPUT_VIDEO = "/path/to/your/input/video.mp4"

# Output video - where to save results
OUTPUT_VIDEO = "/path/to/your/output/video.mp4"
```

---

## Scripts

All scripts are in the `scripts/` directory. Run them from the repository root.

### train_model.py

Basic YOLOv8 training script.

**Usage:**
```bash
python scripts/train_model.py
```

**Configuration:**
Edit the script to change settings:

```python
# Line 7 - Dataset path (update if using different dataset)
DATA_YAML_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets/Indoor-Fire-Smoke/data.yaml"))

# Line 14 - Base model (options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
model = YOLO('yolov8n.pt')

# Lines 21-27 - Training parameters
results = model.train(
    data=DATA_YAML_PATH,
    epochs=50,           # Number of training epochs
    imgsz=640,           # Image size
    patience=10,         # Early stopping patience
    name='fire_smoke_detection_v2'  # Run name
)
```

**Output:** Model weights saved to `runs/detect/fire_smoke_detection_v2/weights/best.pt`

---

### finetune_yolo_augmented.py

Advanced training with fisheye augmentation pipeline.

**Usage:**
```bash
# Basic usage with defaults
python scripts/finetune_yolo_augmented.py

# Full customization
python scripts/finetune_yolo_augmented.py \
    --input-dataset ../datasets/Indoor-Fire-Smoke \
    --output-dataset ../datasets/augmented_dataset \
    --model yolov8n.pt \
    --epochs 50 \
    --batch-size 16 \
    --imgsz 640 \
    --augmentations 3 \
    --workers 4
```

**All Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dataset` | `../datasets/Indoor-Fire-Smoke` | Path to original dataset |
| `--output-dataset` | `../datasets/augmented_dataset` | Output path for augmented dataset |
| `--model` | `yolov8n.pt` | Base YOLO model (`yolov8n.pt`, `yolov8s.pt`, `yolo12n.pt`, etc.) |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `16` | Training batch size |
| `--imgsz` | `640` | Training image size |
| `--patience` | `15` | Early stopping patience |
| `--device` | `auto` | Device (`cuda:0`, `cpu`, or auto-detect) |
| `--augmentations` | `3` | Augmented versions per image |
| `--workers` | `4` | Parallel workers for augmentation |
| `--skip-augmentation` | `false` | Skip augmentation, use existing data |
| `--validate-only` | `false` | Only run validation |
| `--export` | `[]` | Export formats (e.g., `--export onnx torchscript`) |
| `--project-name` | `fire_smoke_augmented` | Project name for saving results |

**Output:**
- Augmented dataset in `datasets/augmented_dataset/`
- Model weights in `scripts/runs/train/fire_smoke_augmented/weights/best.pt`
- Final model: `scripts/fire_smoke_augmented_final.pt`

---

### augment_dataset.py

Standalone data augmentation with fisheye distortion.

**Usage:**
```bash
python scripts/augment_dataset.py \
    --input ../datasets/Indoor-Fire-Smoke \
    --output ../datasets/augmented_output \
    --augmentations 3 \
    --splits train valid \
    --workers 4
```

**All Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *required* | Input dataset directory |
| `--output` | *required* | Output directory for augmented dataset |
| `--augmentations` | `3` | Number of augmentations per image |
| `--splits` | `['train']` | Splits to augment (e.g., `train valid test`) |
| `--workers` | `4` | Parallel processing workers |
| `--fisheye-prob` | `0.3` | Probability of fisheye distortion |
| `--color-prob` | `0.7` | Probability of color jitter |
| `--blur-prob` | `0.3` | Probability of blur |
| `--noise-prob` | `0.2` | Probability of noise |
| `--rotation-prob` | `0.5` | Probability of rotation |
| `--flip-prob` | `0.5` | Probability of flip |

**Augmentation Types Applied:**
- Fisheye/barrel distortion (simulates fisheye lens)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian and motion blur
- Gaussian and salt-pepper noise
- Rotation (-15 to +15 degrees)
- Horizontal/vertical flip

---

### inference_test.py

Run inference on test images.

**Usage:**
```bash
python scripts/inference_test.py
```

**Configuration:**
Edit the script to customize:

```python
# Line 11 - Trained weights path
TRAINED_WEIGHTS = os.path.join(SCRIPT_DIR, 'runs/detect/fire_smoke_detection_v2/weights/best.pt')

# Line 27 - Test images directory
TEST_IMAGES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../datasets/Indoor-Fire-Smoke/test/images"))

# Line 28 - Number of test images
image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))[:3]  # Change [:3] to test more

# Line 38 - Confidence threshold
results = model.predict(image_files, conf=0.25, save=True, name='inference_results_v2')
```

**Output:** Annotated images saved to `runs/detect/inference_results_v2/`

---

## GUI Application

A Streamlit-based web interface for real-time fire and smoke detection.

### Features
- Upload custom YOLO model weights (.pt files)
- Image analysis with detection visualization
- Live camera stream detection
- Adjustable confidence and IOU thresholds

### Running the GUI

```bash
cd gui
streamlit run app.py
```

Or from repository root:
```bash
streamlit run gui/app.py
```

### Using the GUI

1. **Open browser** at `http://localhost:8501`
2. **Upload model:** In the sidebar, upload your trained `.pt` file (e.g., `best.pt`)
3. **Adjust settings:** Set confidence threshold (default 0.45) and IOU threshold (default 0.70)
4. **Detect:**
   - **Image Analysis tab:** Upload an image and click "RUN ANALYTICS"
   - **Live Stream tab:** Enable camera and capture frames for analysis

### Where to Find Trained Models

After training, models are saved to:
- Basic training: `scripts/runs/detect/fire_smoke_detection_v2/weights/best.pt`
- Augmented training: `scripts/runs/train/fire_smoke_augmented/weights/best.pt`
- Notebook training: `notebooks/yolov12/runs/detect/train/weights/best.pt`

---

## Training Workflow

### Recommended Steps

1. **Prepare dataset:**
   ```bash
   # Download and extract dataset
   unzip "Indoor Fire Smoke.zip" -d datasets/
   mv "datasets/Indoor Fire Smoke" datasets/Indoor-Fire-Smoke
   ```

2. **Augment data (optional but recommended for fisheye):**
   ```bash
   python scripts/augment_dataset.py \
       --input datasets/Indoor-Fire-Smoke \
       --output datasets/augmented_dataset \
       --augmentations 3 \
       --splits train valid
   ```

3. **Train model:**
   ```bash
   python scripts/finetune_yolo_augmented.py \
       --input-dataset datasets/Indoor-Fire-Smoke \
       --output-dataset datasets/augmented_dataset \
       --epochs 50 \
       --model yolov8n.pt
   ```

4. **Test model:**
   ```bash
   python scripts/inference_test.py
   ```

5. **Deploy with GUI:**
   ```bash
   streamlit run gui/app.py
   # Upload the trained best.pt file
   ```

---

## Troubleshooting

### Common Issues

**"Dataset not found" error:**
- Ensure dataset is in `datasets/` directory
- Check folder name matches expected name (no spaces)
- Verify `data.yaml` exists and paths are correct

**"CUDA out of memory" error:**
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model yolov8n.pt`
- Reduce image size: `--imgsz 416`

**Poor detection results:**
- Train for more epochs: `--epochs 100`
- Use more augmentation: `--augmentations 5`
- Try larger model: `--model yolov8s.pt`

---

## License

[Add your license here]
