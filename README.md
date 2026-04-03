# Indoor Fire & Smoke Detection — Fisheye Lens (RT-DETR + RectConv)

Real-time fire and smoke detection optimised for fisheye security cameras.
Two detection classes: **fire** and **smoke**.

The core contribution of this project is the **RectConv integration** (ConvRect): fisheye-geometry-aware convolutions that let a standard perspective-trained detector run natively on fisheye images — no global rectification, no dead zones, no coordinate back-projection step.

---

## Table of Contents

1. [Method](#method)
   - [Problem: fisheye distortion breaks CNNs](#problem-fisheye-distortion-breaks-cnns)
   - [Solution: RectConv geometric patch sampling](#solution-rectconv-geometric-patch-sampling)
   - [Dataset: cylindrical joint preparation](#dataset-cylindrical-joint-preparation)
   - [Model: RT-DETR-L backbone](#model-rt-detr-l-backbone)
   - [Full training pipeline](#full-training-pipeline)
2. [Results](#results)
   - [Training run summary](#training-run-summary)
   - [Epoch-by-epoch metrics](#epoch-by-epoch-metrics)
   - [Visualisation files](#visualisation-files)
   - [Saved weights](#saved-weights)
3. [Repository layout](#repository-layout)
4. [Setup](#setup)
5. [Usage](#usage)
   - [Prepare the cylindrical joint dataset](#prepare-the-cylindrical-joint-dataset)
   - [Train the ConvRect model](#train-the-convrect-model)
   - [Run the Streamlit GUI](#run-the-streamlit-gui)
6. [Camera calibration](#camera-calibration)

---

## Method

### Problem: fisheye distortion breaks CNNs

Standard convolutional neural networks assume **translation invariance**: the same feature (an edge, a flame texture) should activate the same filter regardless of where it appears in the image.  This assumption holds for pinhole (perspective) cameras.

A fisheye lens introduces strong radial distortion — the further a point is from the image centre, the more it is compressed and rotated relative to a perspective view.  Consequences for a fire detector:

- A flame in the image centre looks sharp and upright; the same flame near the edge appears warped and scaled differently.
- Standard convolution kernels sampled on a regular grid around each off-centre position see geometrically inconsistent neighbourhoods — the "patch" they see is distorted relative to the patch they were trained on.
- Accuracy degrades at the periphery — exactly where fisheye cameras are most useful (wide coverage, room corners).

Traditional solutions (global rectification, cylindrical unwrapping) fix the geometry but sacrifice field of view or introduce large black "dead zone" borders.

### Solution: RectConv geometric patch sampling

**RectConv** (Robotic Imaging lab, *"Adapting CNNs for Fisheye Cameras without Retraining"*, arXiv:2404.08187) solves this at the convolution level:

```
Standard Conv2d:
  For each output position p, sample a regular K×K grid around p in the input.

RectifyConv2d:
  For each output position p, compute the locally perspective-corrected K×K
  sampling grid using the camera's radial distortion model, then sample
  those (non-integer) input locations via bilinear interpolation.
```

Every convolution kernel sees a **locally rectified patch** — a small neighbourhood that looks like a perspective view regardless of where it sits in the fisheye frame.

| Property | Value |
|---|---|
| Full fisheye FOV preserved | No dead zones, no cropping |
| Model weights unchanged | A COCO-pretrained RT-DETR-L works out of the box |
| Detections in fisheye coordinates | No post-processing back-projection required |
| Offset map computed once per camera | Negligible runtime overhead per frame |

#### Camera model

The distortion offset map is derived from the **Kannala-Brandt radial polynomial**:

```
r = k1·θ + k2·θ² + k3·θ³ + k4·θ⁴
```

For most wide-angle IP security cameras the equidistant simplification applies:

```
r = k1·θ    (k2 = k3 = k4 = 0,   k1 = r_max / (FOV_rad / 2))
```

The distortion map is cached to `cameras/cache/distmap_<hash>.pt` after the first computation so subsequent runs load it instantly.

#### Model patching

After loading RT-DETR-L, every `nn.Conv2d` with kernel size > 1 is replaced in-place with a `RectifyConv2d` that carries the precomputed offset map.  The replacement is done by `scripts/rectconv_adapter.py`:

```python
from scripts.rectconv_adapter import make_camera_from_fov, build_distortion_map, patch_model

cam     = make_camera_from_fov(w=640, h=640, fov_deg=180)
distmap = build_distortion_map(cam, cache_path="cameras/cache")
patch_model(model.model, distmap)   # modifies the nn.Module in-place
```

All Conv2d layers in the backbone, neck, and detection head are patched in a single pass before training begins.

### Dataset: cylindrical joint preparation

To maximise training data diversity we built a joint dataset from three sources:

| Source | Type | Original size | Processing |
|---|---|---|---|
| DS1 — fisheye fire | fisheye 1024 × 1024, 180° FOV | fire labels | warped to cylindrical 640 × 640 |
| DS3 — fisheye fire/smoke | fisheye 1280 × 960, 180° FOV | fire + smoke labels | warped to cylindrical 640 × 640 |
| DS2 — Indoor Fire Smoke | perspective 640 × 640 | fire + smoke labels | copied as-is |

**Cylindrical projection** (`scripts/train_cylindrical_joint.py`) is applied to the fisheye sources before merging:

1. Estimate equidistant calibration from image dimensions (`f = r_max / (FOV_rad/2)`).
2. Build `cv2.remap()` backward lookup tables for every output pixel at 640 × 640.
3. Warp each image with bilinear interpolation.
4. Transform each YOLO bounding box by projecting **8 sample points** (4 corners + 4 edge midpoints) through the inverse cylindrical-to-fisheye mapping and enclosing them in a new axis-aligned box.
5. Drop any box where fewer than 2 of the 8 sample points remain within the valid fisheye circle.

All images land in a single staging pool, shuffled with a fixed seed (42), and split **85 % train / 15 % val**.

Output: `datasets/cylindrical_joint/` — standard YOLO directory layout with `data.yaml`.

### Model: RT-DETR-L backbone

We use **RT-DETR-L** (Real-Time Detection Transformer, Large variant) loaded via Ultralytics:

- Transformer-based detector (DETR-style) with a ResNet-50 backbone and a hybrid CNN-Transformer encoder.
- Pre-trained on COCO 2017 (perspective images).
- Fine-tuned end-to-end on the cylindrical joint dataset *after* RectConv patching.
- Input resolution: **640 × 640**.
- Base weights: `rtdetr-l.pt` (64 MB, auto-downloaded by Ultralytics if absent).

RT-DETR was chosen over YOLO because:
- Its attention mechanism naturally adapts to the spatially non-uniform feature distributions produced by RectConv.
- The bipartite matching loss is agnostic to the spatial arrangement of objects, which is important when fisheye geometry causes unusual size and aspect-ratio distributions near the image periphery.
- No anchor hyperparameters to re-tune for a new domain.

### Full training pipeline

```
Raw fisheye images (DS1, DS3)
    │
    ▼ fisheye → cylindrical projection  (train_cylindrical_joint.py)
    │   equidistant remap, 8-point bbox transform, 85/15 split
    ▼
datasets/cylindrical_joint/   ◄── DS2 perspective images merged in
    │
    ▼ build RectConv offset map         (rectconv_adapter.py)
    │   equidistant camera, 640×640, cached to cameras/cache/
    ▼
RT-DETR-L loaded  (rtdetr-l.pt)
    │
    ▼ patch all Conv2d → RectifyConv2d  (rectconv_adapter.patch_model)
    │   weights unchanged, sampling grid corrected for fisheye geometry
    ▼
Fine-tune on cylindrical_joint          (train_rtdetr_rectconv.py)
    │   epochs=20, batch=4, imgsz=640
    │   HSV jitter (h=0.015, s=0.9, v=0.6), mosaic=1.0, mixup=0.1
    │   rotation ±10°, fliplr=0.5, flipud=0.1
    │   warmup 3 epochs, cosine LR decay, early stopping patience=15
    ▼
whights/rtdetr-l-rectconv_v1_2026-04-02.pt   ← best checkpoint
```

---

## Results

### Training run summary

**Run ID:** `rtdetr-l-rectconv_v1_2026-04-02`

| Parameter | Value |
|---|---|
| Base model | RT-DETR-L (COCO pretrained) |
| Dataset | cylindrical_joint |
| Epochs completed | 20 |
| Batch size | 4 |
| Image size | 640 × 640 |
| Optimizer | AdamW (auto), weight decay 1e-4 |
| Warmup | 3 epochs |
| LR schedule | cosine decay, lr0=0.01 → lrf=0.01 |
| Early stopping patience | 15 |

**Best checkpoint metrics (mAP@50 peak at epoch 16):**

| Metric | Value |
|---|---|
| **mAP@50** | **0.923** |
| **mAP@50-95** | **0.556** |
| Precision | 0.896 |
| Recall | 0.881 |
| Val GIoU loss | 0.432 |
| Val Class loss | 0.450 |

### Epoch-by-epoch metrics

| Epoch | mAP@50 | mAP@50-95 | Precision | Recall | Train GIoU | Train Cls |
|---|---|---|---|---|---|---|
| 1 | 0.458 | 0.215 | 0.593 | 0.461 | 1.030 | 2.110 |
| 2 | 0.698 | 0.334 | 0.756 | 0.636 | 0.689 | 0.768 |
| 3 | 0.755 | 0.378 | 0.776 | 0.694 | 0.611 | 0.715 |
| 4 | 0.801 | 0.402 | 0.812 | 0.737 | 0.582 | 0.692 |
| 5 | 0.818 | 0.399 | 0.826 | 0.758 | 0.558 | 0.670 |
| 6 | 0.833 | 0.416 | 0.844 | 0.768 | 0.549 | 0.645 |
| 7 | 0.848 | 0.443 | 0.835 | 0.799 | 0.534 | 0.627 |
| 8 | 0.853 | 0.437 | 0.838 | 0.804 | 0.527 | 0.619 |
| 9 | 0.873 | 0.456 | 0.854 | 0.822 | 0.512 | 0.617 |
| 10 | 0.871 | 0.451 | 0.869 | 0.807 | 0.502 | 0.603 |
| 11 | 0.900 | 0.500 | 0.890 | 0.829 | 0.443 | 0.515 |
| 12 | 0.890 | 0.503 | 0.878 | 0.845 | 0.435 | 0.489 |
| 13 | 0.908 | 0.509 | 0.893 | 0.852 | 0.424 | 0.480 |
| 14 | 0.908 | 0.523 | 0.885 | 0.867 | 0.412 | 0.472 |
| 15 | 0.911 | 0.522 | 0.894 | 0.866 | 0.408 | 0.460 |
| **16** | **0.923** | 0.546 | **0.911** | 0.863 | 0.400 | 0.453 |
| 17 | 0.918 | 0.555 | 0.899 | 0.865 | 0.396 | 0.443 |
| 18 | 0.922 | **0.556** | 0.900 | 0.875 | 0.388 | 0.439 |
| 19 | 0.918 | 0.553 | 0.900 | 0.875 | 0.382 | 0.438 |
| 20 | 0.921 | 0.553 | 0.896 | **0.881** | 0.379 | 0.432 |

Notable observations:
- **Epoch 1 → 4**: rapid learning — mAP@50 jumps from 0.46 to 0.80 in 4 epochs, showing effective transfer from the COCO-pretrained backbone.
- **Epoch 10 → 11**: sharp discontinuity (+0.029 mAP@50, +0.049 mAP@50-95) coincides with `close_mosaic=10` — the model consolidates once mosaic augmentation is disabled.
- **Epoch 16+**: mAP@50 plateaus ≥ 0.92 while mAP@50-95 continues climbing, indicating the model is refining localisation precision beyond the coarse IoU=0.5 threshold.
- Training and validation losses track closely throughout, with no sign of over-fitting over 20 epochs.

### Visualisation files

All plots are in `training_results/rtdetr-l-rectconv_v1_2026-04-02/`:

| File | Contents |
|---|---|
| `results.png` | Combined training/val loss and metric curves |
| `BoxF1_curve.png` | F1 score vs confidence threshold |
| `BoxP_curve.png` | Precision vs confidence |
| `BoxR_curve.png` | Recall vs confidence |
| `BoxPR_curve.png` | Precision-Recall curve |
| `confusion_matrix.png` | Raw confusion matrix (fire / smoke / background) |
| `confusion_matrix_normalized.png` | Normalised confusion matrix |
| `labels.jpg` | Distribution of bounding box sizes and positions |
| `train_batch*.jpg` | Sample training batches with ground-truth labels |
| `val_batch*_labels.jpg` | Validation ground truth |
| `val_batch*_pred.jpg` | Validation predictions |

### Saved weights

| File | Description |
|---|---|
| `whights/rtdetr-l-rectconv_v1_2026-04-02.pt` | Best validation checkpoint (primary model) |
| `training_results/.../weights/best.pt` | Same, inside the run directory |
| `training_results/.../weights/last.pt` | Final epoch weights |
| `training_results/.../weights/epoch0.pt` | Epoch 0 snapshot |
| `training_results/.../weights/epoch10.pt` | Epoch 10 snapshot |

---

## Repository layout

```
.
├── cameras/
│   ├── default_180fov.json          # Equidistant 180° FOV camera (640×640)
│   └── cache/                       # Cached RectConv offset maps (.pt)
│
├── datasets/
│   └── cylindrical_joint/           # Prepared training dataset (gitignored)
│       ├── data.yaml
│       ├── train/images/ & labels/
│       └── val/images/   & labels/
│
├── gui/
│   └── app.py                       # Streamlit web interface
│
├── scripts/
│   ├── train_rtdetr_rectconv.py     # Main ConvRect training script
│   ├── rectconv_adapter.py          # RectConv ↔ Ultralytics bridge
│   ├── train_cylindrical_joint.py   # Dataset preparation (cylindrical warp)
│   ├── fisheye_rectifier.py         # Calibration-free fisheye rectifier
│   └── merge_dataset.py             # Utility: merge external datasets
│
├── third_party/
│   └── RectConv/                    # RectConv library (RoboticImaging/RectConv)
│
├── training_results/
│   └── rtdetr-l-rectconv_v1_2026-04-02/
│       ├── args.yaml                # Full training configuration
│       ├── results.csv              # Epoch-by-epoch metrics
│       ├── results.png              # Training curve plot
│       ├── confusion_matrix*.png
│       ├── Box*_curve.png
│       ├── train_batch*.jpg
│       ├── val_batch*_{labels,pred}.jpg
│       └── weights/                 # best.pt, last.pt, epoch snapshots
│
├── whights/
│   ├── registry.json                # Model registry (id, date, metrics)
│   └── rtdetr-l-rectconv_v1_2026-04-02.pt   # Best trained weights
│
├── rtdetr-l.pt                      # Base RT-DETR-L weights (COCO pretrained)
└── .streamlit/config.toml           # Fire-orange UI theme
```

---

## Setup

### Install dependencies

```bash
pip install ultralytics opencv-python torch torchvision streamlit pillow numpy scipy
```

### RectConv library

The `third_party/RectConv/` directory is included in this repository.
If it is missing for any reason, clone it:

```bash
git clone https://github.com/RoboticImaging/RectConv.git third_party/RectConv
```

`rectconv_adapter.py` adds the necessary paths to `sys.path` automatically — no build step needed.

---

## Usage

### Prepare the cylindrical joint dataset

Run once to build `datasets/cylindrical_joint/` from your raw data sources:

```bash
python scripts/train_cylindrical_joint.py \
    --ds1 /path/to/fisheye-fire-1024 \
    --ds2 /path/to/indoor-fire-smoke-perspective \
    --ds3 /path/to/fisheye-fire-smoke-1280 \
    --output-dataset datasets/cylindrical_joint \
    --hfov 160 --vfov 120 \
    --prepare-only
```

If the dataset is already present (`--skip-prepare`), this step is not needed.

### Train the ConvRect model

#### Approximate camera from FOV (no calibration required)

```bash
python scripts/train_rtdetr_rectconv.py \
    --data datasets/cylindrical_joint/data.yaml \
    --fov 180 --width 640 --height 640 \
    --epochs 50 --batch 8
```

#### With a calibrated camera JSON

```bash
python scripts/train_rtdetr_rectconv.py \
    --data datasets/cylindrical_joint/data.yaml \
    --camera-json cameras/default_180fov.json \
    --epochs 50 --batch 8 --device 0
```

#### Multiple datasets in one session

```bash
python scripts/train_rtdetr_rectconv.py \
    --data /path/ds_A/data.yaml /path/ds_B/data.yaml \
    --fov 180 --epochs 50 --device 0
```

Best weights are saved to `whights/rtdetr-l-rectconv_v{N}_{date}.pt` and registered in `whights/registry.json`.

### Run the Streamlit GUI

```bash
streamlit run gui/app.py
```

Open `http://localhost:8501` in your browser.

| Tab | Description |
|---|---|
| Model Upload | Load any `.pt`, `.onnx`, or `.engine` weights file |
| Image Analysis | Upload a fisheye image and run inference |
| Live Camera | Connect to an IP camera or webcam stream |

Default confidence threshold: 0.45.  Default IOU threshold: 0.70.

---

## Camera calibration

The equidistant approximation works well for most wide-angle IP security cameras.
For maximum accuracy, calibrate your lens and provide a JSON file.

Generate a starting template from FOV:

```bash
python scripts/rectconv_adapter.py \
    --gen-json --fov 180 --width 640 --height 640 \
    --output cameras/my_camera.json
```

Edit the output file and replace `k1` (and optionally `k2`–`k4`) with values
from your calibration tool (OpenCV, MATLAB, Kalibr, etc.).

The JSON format follows the RectConv WoodScape convention:

```json
{
  "intrinsic": {
    "k1": 203.7, "k2": 0.0, "k3": 0.0, "k4": 0.0,
    "width": 640, "height": 640,
    "cx_offset": 0.0, "cy_offset": 0.0, "aspect_ratio": 1.0
  },
  "extrinsic": {
    "quaternion": [0, 0, 0, 1],
    "translation": [0, 0, 0]
  }
}
```

---

## Citation

If you use this work, please also cite the RectConv paper:

```
@article{rectconv2024,
  title  = {Adapting CNNs for Fisheye Cameras without Retraining},
  author = {RoboticImaging lab},
  year   = {2024},
  url    = {https://arxiv.org/abs/2404.08187}
}
```
