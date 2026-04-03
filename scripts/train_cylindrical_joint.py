"""
Joint training on perspective + cylindrical-warped fisheye images for indoor fire detection.

Following the guide: warp fisheye images to cylindrical projection (restores CNN
translation invariance), then jointly train on both perspective and cylindrical data
with a frozen backbone to prevent catastrophic forgetting.

Pipeline:
1. Estimate fisheye camera calibration from image properties (equidistant model)
2. Build cylindrical remap tables with cv2.remap()
3. Warp fisheye images to cylindrical, transform YOLO bounding boxes
4. Copy perspective images as-is
5. Merge into unified YOLO dataset with train/val split
6. Train with frozen backbone + safe augmentations only

Usage:
    python scripts/train_cylindrical_joint.py
    python scripts/train_cylindrical_joint.py --epochs 100 --batch-size 8
    python scripts/train_cylindrical_joint.py --prepare-only
"""

import os
import sys
import cv2
import numpy as np
import random
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# 1. Fisheye Calibration Estimation
# ---------------------------------------------------------------------------

def estimate_fisheye_calibration(img_h, img_w, fov_degrees=180.0):
    """
    Estimate fisheye camera calibration from image dimensions.
    Assumes equidistant fisheye model: r = f * theta.

    Args:
        img_h: Image height in pixels
        img_w: Image width in pixels
        fov_degrees: Full field of view in degrees (180 for typical fisheye)

    Returns:
        dict with cx, cy, radius, f (focal length in pixels), theta_max
    """
    cx = img_w / 2.0
    cy = img_h / 2.0
    radius = min(img_h, img_w) / 2.0
    theta_max = np.radians(fov_degrees) / 2.0  # half-angle
    f = radius / theta_max  # equidistant: r = f * theta => f = r / theta

    return {
        'cx': cx,
        'cy': cy,
        'radius': radius,
        'f': f,
        'theta_max': theta_max,
    }


# ---------------------------------------------------------------------------
# 2. Cylindrical Projection Remap Tables
# ---------------------------------------------------------------------------

def build_cylindrical_remap(calib, src_h, src_w, out_h=640, out_w=640,
                            hfov_degrees=160.0, vfov_degrees=120.0):
    """
    Build cv2.remap() lookup tables for fisheye-to-cylindrical projection.

    For each pixel (u, v) in the output cylindrical image, compute which
    pixel in the source fisheye image it maps to.

    Args:
        calib: Calibration dict from estimate_fisheye_calibration()
        src_h, src_w: Source fisheye image dimensions
        out_h, out_w: Output cylindrical image dimensions
        hfov_degrees: Horizontal FOV to unwrap (< full FOV to avoid extreme edges)
        vfov_degrees: Vertical FOV to unwrap

    Returns:
        (map_x, map_y): float32 arrays of shape (out_h, out_w) for cv2.remap()
    """
    cx, cy = calib['cx'], calib['cy']
    f_fish = calib['f']

    hfov_rad = np.radians(hfov_degrees)
    vfov_rad = np.radians(vfov_degrees)

    # Cylindrical focal length: maps horizontal FOV across output width
    f_cyl = out_w / hfov_rad

    # Output pixel grid
    u = np.arange(out_w, dtype=np.float64)
    v = np.arange(out_h, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(u, v)

    # Output pixel -> angular coordinates on cylinder
    phi = (u_grid - out_w / 2.0) / f_cyl       # horizontal angle
    y_angle = (v_grid - out_h / 2.0) / f_cyl   # vertical (linear mapping)

    # 3D ray direction on the cylinder
    X = np.sin(phi)
    Y = y_angle
    Z = np.cos(phi)

    # Project ray to fisheye (equidistant model)
    r_xyz = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r_xyz, Z)

    # Equidistant: r_fisheye = f * theta
    r_fisheye = f_fish * theta

    # Azimuthal angle in fisheye image plane
    phi_fisheye = np.arctan2(Y, X)

    # Fisheye pixel coordinates
    map_x = (cx + r_fisheye * np.cos(phi_fisheye)).astype(np.float32)
    map_y = (cy + r_fisheye * np.sin(phi_fisheye)).astype(np.float32)

    # Mask out-of-bounds pixels
    out_of_bounds = (
        (map_x < 0) | (map_x >= src_w) |
        (map_y < 0) | (map_y >= src_h) |
        (theta > calib['theta_max'])
    )
    map_x[out_of_bounds] = -1
    map_y[out_of_bounds] = -1

    return map_x, map_y


def warp_fisheye_to_cylindrical(image, map_x, map_y):
    """Apply pre-computed cylindrical remap to a fisheye image."""
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# ---------------------------------------------------------------------------
# 3. Bounding Box Transformation (Fisheye -> Cylindrical)
# ---------------------------------------------------------------------------

def fisheye_point_to_cylindrical(px, py, calib, out_h, out_w, f_cyl):
    """
    Map a single point from fisheye pixel coordinates to cylindrical pixel
    coordinates (inverse of the remap direction).

    Returns:
        (u_out, v_out) in cylindrical image, or None if outside valid region.
    """
    cx_fish, cy_fish = calib['cx'], calib['cy']
    f_fish = calib['f']

    # Fisheye pixel -> polar
    dx = px - cx_fish
    dy = py - cy_fish
    r_fisheye = np.sqrt(dx**2 + dy**2)
    phi_fisheye = np.arctan2(dy, dx)

    # Inverse equidistant: theta = r / f
    theta = r_fisheye / f_fish
    if theta > calib['theta_max']:
        return None

    # Angle -> 3D ray
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    X = sin_theta * np.cos(phi_fisheye)
    Y = sin_theta * np.sin(phi_fisheye)
    Z = cos_theta

    # 3D ray -> cylindrical coordinates
    phi_cyl = np.arctan2(X, Z)
    denom = np.sqrt(X**2 + Z**2)
    if denom < 1e-10:
        return None
    y_angle = Y / denom

    # Cylindrical coordinates -> output pixel
    u_out = phi_cyl * f_cyl + out_w / 2.0
    v_out = y_angle * f_cyl + out_h / 2.0

    return u_out, v_out


def transform_bbox_to_cylindrical(bbox, src_h, src_w, calib, out_h, out_w,
                                  hfov_degrees):
    """
    Transform a YOLO bounding box from fisheye to cylindrical coordinates.

    Maps 8 sample points (4 corners + 4 edge midpoints) through the inverse
    projection, then computes the enclosing axis-aligned bounding box.

    Args:
        bbox: [class_id, x_center, y_center, width, height] (normalized 0-1)
        src_h, src_w: Source fisheye image dimensions
        calib: Calibration dict
        out_h, out_w: Cylindrical output dimensions
        hfov_degrees: Horizontal FOV used for cylindrical projection

    Returns:
        Transformed [class_id, xc, yc, w, h] (normalized), or None if lost.
    """
    class_id, xc, yc, bw, bh = bbox
    f_cyl = out_w / np.radians(hfov_degrees)

    # YOLO normalized -> pixel corners
    x1 = (xc - bw / 2) * src_w
    y1 = (yc - bh / 2) * src_h
    x2 = (xc + bw / 2) * src_w
    y2 = (yc + bh / 2) * src_h

    # 4 corners + 4 edge midpoints for accuracy on non-linear warp
    sample_points = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        ((x1 + x2) / 2, y1),
        ((x1 + x2) / 2, y2),
        (x1, (y1 + y2) / 2),
        (x2, (y1 + y2) / 2),
    ]

    projected = []
    for px, py in sample_points:
        result = fisheye_point_to_cylindrical(px, py, calib, out_h, out_w, f_cyl)
        if result is not None:
            projected.append(result)

    if len(projected) < 2:
        return None

    # Enclosing axis-aligned bounding box in cylindrical image
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]

    u_min = max(0, min(us))
    u_max = min(out_w, max(us))
    v_min = max(0, min(vs))
    v_max = min(out_h, max(vs))

    if u_max <= u_min or v_max <= v_min:
        return None

    # Back to YOLO normalized
    new_xc = ((u_min + u_max) / 2) / out_w
    new_yc = ((v_min + v_max) / 2) / out_h
    new_bw = (u_max - u_min) / out_w
    new_bh = (v_max - v_min) / out_h

    # Minimum size filter
    if new_bw < 0.005 or new_bh < 0.005:
        return None

    # Clamp to valid range
    new_xc = np.clip(new_xc, 0.01, 0.99)
    new_yc = np.clip(new_yc, 0.01, 0.99)
    new_bw = min(new_bw, min(new_xc, 1 - new_xc) * 2)
    new_bh = min(new_bh, min(new_yc, 1 - new_yc) * 2)

    return [class_id, new_xc, new_yc, new_bw, new_bh]


# ---------------------------------------------------------------------------
# 4. Label I/O Helpers
# ---------------------------------------------------------------------------

def read_yolo_labels(label_path):
    """Read YOLO format labels from a text file."""
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:5]))
                    bboxes.append([class_id] + coords)
    return bboxes


def write_yolo_labels(label_path, bboxes):
    """Write YOLO format labels to a text file."""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id = int(bbox[0])
            xc, yc, w, h = bbox[1:5]
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def find_annotated_images(images_dir, labels_dir):
    """Return sorted list of image filenames that have a matching label file."""
    label_stems = {Path(f).stem for f in os.listdir(labels_dir) if f.endswith('.txt')}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    annotated = []
    for img_file in sorted(os.listdir(images_dir)):
        ext = Path(img_file).suffix.lower()
        if ext in image_extensions and Path(img_file).stem in label_stems:
            # Check label is non-empty
            lbl_path = os.path.join(labels_dir, Path(img_file).stem + '.txt')
            if os.path.getsize(lbl_path) > 0:
                annotated.append(img_file)

    return annotated


# ---------------------------------------------------------------------------
# 5. Process Fisheye Dataset (Warp to Cylindrical)
# ---------------------------------------------------------------------------

def process_fisheye_dataset(images_dir, labels_dir, map_x, map_y, calib,
                            src_h, src_w, out_h, out_w, hfov_degrees,
                            output_images_dir, output_labels_dir,
                            prefix, num_workers=4):
    """
    Warp all annotated fisheye images to cylindrical and save with
    transformed labels.

    Args:
        images_dir, labels_dir: Source dataset directories
        map_x, map_y: Pre-computed remap tables from build_cylindrical_remap()
        calib: Calibration dict
        src_h, src_w: Source image dimensions
        out_h, out_w: Output cylindrical dimensions
        hfov_degrees: HFOV used for cylindrical projection
        output_images_dir, output_labels_dir: Destination directories
        prefix: Filename prefix to avoid collisions (e.g., "ds1_")
        num_workers: Parallel workers for processing

    Returns:
        Number of images successfully processed
    """
    annotated = find_annotated_images(images_dir, labels_dir)
    print(f"  Found {len(annotated)} annotated images")

    processed = 0
    skipped = 0

    def process_one(img_file):
        stem = Path(img_file).stem
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, stem + '.txt')

        image = cv2.imread(img_path)
        if image is None:
            return False

        # Resize to expected dimensions if needed
        h, w = image.shape[:2]
        if h != src_h or w != src_w:
            image = cv2.resize(image, (src_w, src_h))

        # Warp to cylindrical
        cylindrical = warp_fisheye_to_cylindrical(image, map_x, map_y)

        # Transform bounding boxes
        bboxes = read_yolo_labels(lbl_path)
        new_bboxes = []
        for bbox in bboxes:
            result = transform_bbox_to_cylindrical(
                bbox, src_h, src_w, calib, out_h, out_w, hfov_degrees
            )
            if result is not None:
                new_bboxes.append(result)

        if len(new_bboxes) == 0:
            return False  # Skip images where all boxes are lost in projection

        # Save
        out_name = f"{prefix}{stem}"
        cv2.imwrite(
            os.path.join(output_images_dir, out_name + '.jpg'),
            cylindrical,
        )
        write_yolo_labels(
            os.path.join(output_labels_dir, out_name + '.txt'),
            new_bboxes,
        )
        return True

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, f): f for f in annotated}
        for future in as_completed(futures):
            try:
                if future.result():
                    processed += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"    Error processing {futures[future]}: {e}")
                skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} images (no valid boxes after projection)")

    return processed


# ---------------------------------------------------------------------------
# 6. Copy Perspective Dataset (As-Is)
# ---------------------------------------------------------------------------

def copy_perspective_dataset(source_dir, split, output_images_dir,
                             output_labels_dir, prefix="persp_"):
    """
    Copy annotated perspective images and labels as-is to the output dataset.

    Args:
        source_dir: Root of the perspective dataset (has train/, valid/, test/)
        split: "train", "valid", or "test"
        output_images_dir, output_labels_dir: Destination directories
        prefix: Filename prefix to avoid collisions

    Returns:
        Number of images copied
    """
    src_img_dir = os.path.join(source_dir, split, 'images')
    src_lbl_dir = os.path.join(source_dir, split, 'labels')

    if not os.path.isdir(src_img_dir):
        print(f"  Warning: {src_img_dir} does not exist, skipping")
        return 0

    count = 0
    for img_file in sorted(os.listdir(src_img_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        stem = Path(img_file).stem
        ext = Path(img_file).suffix
        lbl_file = stem + '.txt'
        src_lbl = os.path.join(src_lbl_dir, lbl_file)

        if not os.path.exists(src_lbl) or os.path.getsize(src_lbl) == 0:
            continue  # Only copy annotated images

        out_name = f"{prefix}{stem}"
        shutil.copy2(
            os.path.join(src_img_dir, img_file),
            os.path.join(output_images_dir, out_name + ext),
        )
        shutil.copy2(
            src_lbl,
            os.path.join(output_labels_dir, out_name + '.txt'),
        )
        count += 1

    return count


# ---------------------------------------------------------------------------
# 7. Unified Dataset Assembly
# ---------------------------------------------------------------------------

def prepare_joint_dataset(ds1_path, ds2_path, ds3_path, output_dir,
                          val_ratio=0.15, seed=42, hfov_degrees=160.0,
                          vfov_degrees=120.0, num_workers=4):
    """
    Prepare the unified joint training dataset.

    1. Warp DS1 + DS3 fisheye images to cylindrical 640x640
    2. Copy DS2 perspective images as-is
    3. Merge all into staging, random train/val split
    4. Create data.yaml

    Returns:
        Path to data.yaml
    """
    output_dir = Path(output_dir)

    # Staging directory (all images go here first, then split)
    staging_dir = output_dir / '_staging'
    staging_img = staging_dir / 'images'
    staging_lbl = staging_dir / 'labels'
    staging_img.mkdir(parents=True, exist_ok=True)
    staging_lbl.mkdir(parents=True, exist_ok=True)

    total = 0

    # ---- Dataset 1: fisheye 1024x1024 ----
    print("\n[1/3] Processing Dataset 1 (fisheye 1024x1024)...")
    calib1 = estimate_fisheye_calibration(1024, 1024, fov_degrees=180.0)
    map_x1, map_y1 = build_cylindrical_remap(
        calib1, 1024, 1024, out_h=640, out_w=640,
        hfov_degrees=hfov_degrees, vfov_degrees=vfov_degrees,
    )
    n1 = process_fisheye_dataset(
        os.path.join(ds1_path, 'images'),
        os.path.join(ds1_path, 'labels'),
        map_x1, map_y1, calib1,
        1024, 1024, 640, 640, hfov_degrees,
        str(staging_img), str(staging_lbl),
        prefix="ds1_", num_workers=num_workers,
    )
    print(f"  -> {n1} cylindrical images produced")
    total += n1

    # ---- Dataset 3: fisheye 1280x960 ----
    print("\n[2/3] Processing Dataset 3 (fisheye 1280x960)...")
    calib3 = estimate_fisheye_calibration(960, 1280, fov_degrees=180.0)
    map_x3, map_y3 = build_cylindrical_remap(
        calib3, 960, 1280, out_h=640, out_w=640,
        hfov_degrees=hfov_degrees, vfov_degrees=vfov_degrees,
    )
    n3 = process_fisheye_dataset(
        os.path.join(ds3_path, 'images'),
        os.path.join(ds3_path, 'labels'),
        map_x3, map_y3, calib3,
        960, 1280, 640, 640, hfov_degrees,
        str(staging_img), str(staging_lbl),
        prefix="ds3_", num_workers=num_workers,
    )
    print(f"  -> {n3} cylindrical images produced")
    total += n3

    # ---- Dataset 2: perspective (already split) ----
    print("\n[3/3] Processing Dataset 2 (perspective)...")
    n2_train = copy_perspective_dataset(
        ds2_path, 'train', str(staging_img), str(staging_lbl), prefix="persp_",
    )
    n2_valid = copy_perspective_dataset(
        ds2_path, 'valid', str(staging_img), str(staging_lbl), prefix="perspv_",
    )
    n2 = n2_train + n2_valid
    print(f"  -> {n2} perspective images copied ({n2_train} train + {n2_valid} valid)")
    total += n2

    # ---- Shuffle and split ----
    print(f"\nTotal unified images: {total}")

    all_images = sorted([
        f for f in os.listdir(str(staging_img))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    random.seed(seed)
    random.shuffle(all_images)

    val_count = max(1, int(len(all_images) * val_ratio))
    val_set = set(all_images[:val_count])

    print(f"Split: {len(all_images) - val_count} train / {val_count} val")

    # Create final directory structure
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Move from staging to final splits
    for img_file in all_images:
        stem = Path(img_file).stem
        ext = Path(img_file).suffix
        lbl_file = stem + '.txt'
        split = 'val' if img_file in val_set else 'train'

        shutil.move(
            str(staging_img / img_file),
            str(output_dir / split / 'images' / img_file),
        )
        lbl_src = staging_lbl / lbl_file
        if lbl_src.exists():
            shutil.move(
                str(lbl_src),
                str(output_dir / split / 'labels' / lbl_file),
            )

    # Clean up staging
    shutil.rmtree(str(staging_dir))

    # Create data.yaml
    data_yaml_path = output_dir / 'data.yaml'
    data_yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: 2\n"
        f"names: ['fire', 'smoke']\n"
    )
    print(f"Created {data_yaml_path}")

    return str(data_yaml_path)


# ---------------------------------------------------------------------------
# 8. Training with Frozen Backbone + Safe Augmentations
# ---------------------------------------------------------------------------

def train_joint_model(data_yaml, model_path, epochs=80, imgsz=640,
                      batch_size=16, patience=20, device=None, freeze=10,
                      project_name='cylindrical_joint'):
    """
    Train YOLO model on the joint cylindrical + perspective dataset.

    Uses frozen backbone and only augmentations that are safe for cylindrical
    images: horizontal flip, color jitter, brightness/contrast, noise.
    NO rotation, vertical flip, perspective transforms, or shear.
    """
    from ultralytics import YOLO
    import torch

    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available, using CPU (training will be slow)")

    print(f"\nLoading weights: {model_path}")
    model = YOLO(model_path)

    print(f"\nTraining configuration:")
    print(f"  Data:       {data_yaml}")
    print(f"  Epochs:     {epochs}")
    print(f"  Img size:   {imgsz}")
    print(f"  Batch:      {batch_size}")
    print(f"  Patience:   {patience}")
    print(f"  Freeze:     {freeze} layers (backbone)")
    print(f"  Device:     {device}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        device=device,
        freeze=freeze,
        project=str(SCRIPT_DIR / 'runs' / 'train'),
        name=project_name,
        verbose=True,
        plots=True,

        # SAFE augmentations only (per guide):
        # YES: color jitter (hsv), horizontal flip, translate, scale
        # NO: rotation, vertical flip, shear, perspective
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,       # NO rotation (breaks cylindrical vertical alignment)
        translate=0.1,
        scale=0.3,
        shear=0.0,          # NO shear
        perspective=0.0,     # NO perspective transform
        flipud=0.0,         # NO vertical flip (breaks up/down orientation)
        fliplr=0.5,         # Horizontal flip is safe
        mosaic=0.5,         # Reduced mosaic
        mixup=0.1,
        close_mosaic=10,
        erasing=0.0,
    )

    print(f"\nTraining complete. Weights at: {results.save_dir}")

    # Validation
    metrics = model.val(data=data_yaml)
    print(f"\nValidation results:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")

    return model, results


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Joint training on perspective + cylindrical-warped fisheye images'
    )

    # Dataset paths
    parser.add_argument('--ds1', type=str,
        default=str(Path.home() / 'Downloads' / 'Indoor Fire detection'
                    / 'camera' / 'dataset fire fisheye'),
        help='Dataset 1: fisheye fire (images/ + labels/)')
    parser.add_argument('--ds2', type=str,
        default=str(Path.home() / 'Downloads' / 'Indoor Fire detection'
                    / 'Datasets' / 'experimental' / 'Indoor Fire Smoke'
                    / 'Indoor Fire Smoke'),
        help='Dataset 2: perspective fire/smoke (train/valid/test)')
    parser.add_argument('--ds3', type=str,
        default=str(Path.home() / 'Downloads' / 'Indoor Fire detection'
                    / 'Datasets' / 'fisheye-lens-images'),
        help='Dataset 3: fisheye fire/smoke (images/ + labels/)')

    # Output
    parser.add_argument('--output-dataset', type=str,
        default=str(PROJECT_ROOT / 'datasets' / 'cylindrical_joint'),
        help='Output directory for the prepared joint dataset')

    # Cylindrical projection settings
    parser.add_argument('--hfov', type=float, default=160.0,
        help='Horizontal FOV for cylindrical projection (degrees)')
    parser.add_argument('--vfov', type=float, default=120.0,
        help='Vertical FOV for cylindrical projection (degrees)')

    # Training settings
    parser.add_argument('--weights', type=str,
        default=str(PROJECT_ROOT / 'whights' / 'YOLOv8-Fine-tuned.pt'),
        help='Base model weights to fine-tune')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--freeze', type=int, default=10,
        help='Number of backbone layers to freeze (10 = full backbone)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project-name', type=str, default='cylindrical_joint')

    # Control flow
    parser.add_argument('--skip-prepare', action='store_true',
        help='Skip dataset preparation, reuse existing prepared dataset')
    parser.add_argument('--prepare-only', action='store_true',
        help='Only prepare dataset, do not train')

    args = parser.parse_args()

    print("=" * 60)
    print("CYLINDRICAL JOINT TRAINING PIPELINE")
    print("=" * 60)
    print(f"  DS1 (fisheye):     {args.ds1}")
    print(f"  DS2 (perspective): {args.ds2}")
    print(f"  DS3 (fisheye):     {args.ds3}")
    print(f"  Output dataset:    {args.output_dataset}")
    print(f"  Base weights:      {args.weights}")
    print(f"  Freeze layers:     {args.freeze}")

    # Step 1: Prepare dataset
    if not args.skip_prepare:
        if os.path.exists(args.output_dataset):
            shutil.rmtree(args.output_dataset)

        data_yaml = prepare_joint_dataset(
            ds1_path=args.ds1,
            ds2_path=args.ds2,
            ds3_path=args.ds3,
            output_dir=args.output_dataset,
            val_ratio=args.val_ratio,
            seed=args.seed,
            hfov_degrees=args.hfov,
            vfov_degrees=args.vfov,
            num_workers=args.workers,
        )
    else:
        data_yaml = os.path.join(args.output_dataset, 'data.yaml')
        if not os.path.exists(data_yaml):
            print(f"Error: {data_yaml} not found. Run without --skip-prepare first.")
            sys.exit(1)
        print(f"\nReusing prepared dataset: {data_yaml}")

    if args.prepare_only:
        print("\nDataset prepared. Exiting (--prepare-only).")
        return

    # Step 2: Train
    if not os.path.exists(args.weights):
        print(f"Error: Weights not found at {args.weights}")
        sys.exit(1)

    model, results = train_joint_model(
        data_yaml=data_yaml,
        model_path=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        freeze=args.freeze,
        project_name=args.project_name,
    )

    # Save final weights
    final_path = str(PROJECT_ROOT / 'whights' / 'cylindrical_joint_final.pt')
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
