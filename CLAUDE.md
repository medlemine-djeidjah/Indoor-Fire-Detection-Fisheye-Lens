# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indoor fire and smoke detection system optimised for fisheye lens cameras.
Two detection classes: `fire` and `smoke`.
The approach is **RT-DETR-L + RectConv** (ConvRect): geometrically-aware convolutions that correct for fisheye radial distortion at the kernel level, applied to a COCO-pretrained RT-DETR-L detector.

## Key Commands

### Install dependencies
```bash
pip install ultralytics opencv-python torch torchvision streamlit pillow numpy scipy
```

### Prepare the cylindrical joint dataset
```bash
python scripts/train_cylindrical_joint.py \
    --ds1 /path/to/fisheye-fire-1024 \
    --ds2 /path/to/perspective-fire-smoke \
    --ds3 /path/to/fisheye-fire-smoke-1280 \
    --output-dataset datasets/cylindrical_joint \
    --prepare-only
```

### Train the ConvRect model
```bash
python scripts/train_rtdetr_rectconv.py \
    --data datasets/cylindrical_joint/data.yaml \
    --fov 180 --epochs 50 --batch 8
```

### Streamlit GUI
```bash
streamlit run gui/app.py
```

## Architecture

**1. Dataset preparation** (`scripts/train_cylindrical_joint.py`)
Warps fisheye images (equidistant model) to cylindrical 640×640, transforms YOLO bounding boxes via 8-point sampling, and merges with perspective images into a unified train/val split.

**2. RectConv integration** (`scripts/rectconv_adapter.py`)
Bridge between `third_party/RectConv` and Ultralytics RT-DETR.  Builds the distortion offset map from the camera model (or a calibration JSON), caches it to `cameras/cache/`, and patches all `nn.Conv2d` layers → `RectifyConv2d` in-place.

**3. Training** (`scripts/train_rtdetr_rectconv.py`)
Loads RT-DETR-L, patches with RectConv, fine-tunes on the cylindrical joint dataset.  Auto-versions runs in `training_results/` and copies best weights to `whights/`.

**4. Streamlit GUI** (`gui/app.py`)
Web interface with model upload, image analysis, and live camera stream tabs.  Uses `@st.cache_resource` for model caching.  Configurable confidence (default 0.45) and IOU (default 0.70) thresholds.

**5. Supporting scripts**
- `scripts/fisheye_rectifier.py` — calibration-free fisheye → perspective rectification, used for back-projecting boxes from rectified space to fisheye coordinates.
- `scripts/merge_dataset.py` — utility to merge external datasets (deduplication, class-count reporting).

## Data Layout

```
datasets/cylindrical_joint/
├── data.yaml          # nc: 2, names: ['fire', 'smoke']
├── train/
│   ├── images/
│   └── labels/        # YOLO: class_id x_center y_center width height (normalised)
└── val/
    ├── images/
    └── labels/
```

## Model Weights

- `whights/rtdetr-l-rectconv_v1_2026-04-02.pt` — best trained ConvRect model (mAP@50: 0.923)
- `whights/registry.json` — model registry (id, date, metrics, dataset)
- `rtdetr-l.pt` — base RT-DETR-L weights (COCO pretrained, 64 MB)
- Training run outputs: `training_results/rtdetr-l-rectconv_v1_2026-04-02/`

## Important Notes

- No `requirements.txt`; install dependencies manually via pip.
- All scripts use `SCRIPT_DIR`/`PROJECT_ROOT` relative paths and must be run from the repository root.
- `.pt` files are tracked in git (overridden in `.gitignore`).
- `third_party/RectConv/` must be present; `rectconv_adapter.py` adds it to `sys.path` at import time.
- If you change the fisheye FOV or image size, delete the cached distortion maps in `cameras/cache/` so they are recomputed.
- `train_cylindrical_joint.py` also has a full YOLO training sub-step (`train_joint_model()`), but the primary use is `--prepare-only` to build the dataset for `train_rtdetr_rectconv.py`.
