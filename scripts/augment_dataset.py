"""
Data Augmentation Script for Fire/Smoke Detection with YOLO

Based on insights from fisheye camera tutorials, this script implements:
1. Fisheye-style geometric distortions (barrel/pincushion distortion)
2. Standard augmentations optimized for fire/smoke detection

Augmentation Techniques:
- Fisheye distortion: Simulates wide-angle/fisheye lens effects
- Color jitter: Brightness, contrast, saturation variations
- Rotation & flipping: Geometric transformations
- Blur effects: Gaussian blur, motion blur (fire involves motion)
- Noise addition: Gaussian noise for robustness
- Hue shifting: Important for fire color variations
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def fisheye_distortion(image, strength=0.5):
    """
    Apply barrel/fisheye distortion to simulate fisheye lens effect.
    Based on fisheye tutorial geometric transformation principles.

    Args:
        image: Input image (BGR)
        strength: Distortion strength (0.0 to 1.0, negative for pincushion)
    Returns:
        Distorted image
    """
    h, w = image.shape[:2]

    # Create coordinate grids
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    # Calculate radial distance from center
    r = np.sqrt(X**2 + Y**2)

    # Apply barrel distortion formula
    # r' = r * (1 + k * r^2) where k controls distortion strength
    k = strength * 0.5
    r_distorted = r * (1 + k * r**2)

    # Avoid division by zero
    r[r == 0] = 1e-10

    # Calculate new coordinates
    scale = r_distorted / r
    X_new = X * scale
    Y_new = Y * scale

    # Convert back to pixel coordinates
    map_x = ((X_new + 1) * w / 2).astype(np.float32)
    map_y = ((Y_new + 1) * h / 2).astype(np.float32)

    # Apply remapping
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return distorted


def adjust_bbox_for_fisheye(bbox, img_shape, strength=0.5):
    """
    Adjust bounding box coordinates after fisheye distortion.
    Uses same transformation as the image distortion.

    Args:
        bbox: [class_id, x_center, y_center, width, height] (normalized)
        img_shape: (height, width)
        strength: Same strength used for image distortion
    Returns:
        Adjusted bbox in same format
    """
    h, w = img_shape[:2]
    class_id, x_c, y_c, bw, bh = bbox

    # Convert to image coordinates
    x_c_px = x_c * w
    y_c_px = y_c * h

    # Convert to normalized coordinates (-1 to 1)
    x_norm = (x_c_px / w) * 2 - 1
    y_norm = (y_c_px / h) * 2 - 1

    # Apply same distortion formula
    r = np.sqrt(x_norm**2 + y_norm**2)
    k = strength * 0.5
    r_distorted = r * (1 + k * r**2)

    if r > 1e-10:
        scale = r_distorted / r
        x_new = x_norm * scale
        y_new = y_norm * scale
    else:
        x_new, y_new = x_norm, y_norm

    # Convert back to normalized (0-1)
    x_c_new = (x_new + 1) / 2
    y_c_new = (y_new + 1) / 2

    # Scale width/height based on local distortion
    bw_new = bw * (1 + abs(k) * 0.5)
    bh_new = bh * (1 + abs(k) * 0.5)

    # Clamp values
    x_c_new = np.clip(x_c_new, 0.01, 0.99)
    y_c_new = np.clip(y_c_new, 0.01, 0.99)
    bw_new = min(bw_new, min(x_c_new, 1 - x_c_new) * 2)
    bh_new = min(bh_new, min(y_c_new, 1 - y_c_new) * 2)

    return [class_id, x_c_new, y_c_new, bw_new, bh_new]


def color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
    """
    Apply color jittering - important for fire/smoke color variations.
    """
    img = image.copy().astype(np.float32)

    # Brightness
    if random.random() < 0.5:
        factor = 1 + random.uniform(-brightness, brightness)
        img = img * factor

    # Contrast
    if random.random() < 0.5:
        factor = 1 + random.uniform(-contrast, contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean

    # Convert to HSV for saturation and hue
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation
    if random.random() < 0.5:
        factor = 1 + random.uniform(-saturation, saturation)
        hsv[:, :, 1] = hsv[:, :, 1] * factor

    # Hue (important for fire color variations: red, orange, yellow)
    if random.random() < 0.5:
        shift = random.uniform(-hue, hue) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

    hsv = np.clip(hsv, 0, [180, 255, 255])
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def apply_blur(image, blur_type='gaussian'):
    """
    Apply blur effects - motion blur is relevant for dynamic fire/smoke.
    """
    if blur_type == 'gaussian':
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    elif blur_type == 'motion':
        # Motion blur kernel
        size = random.randint(5, 15)
        kernel = np.zeros((size, size))
        angle = random.uniform(0, 180)

        # Create motion blur kernel
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))

        for i in range(size):
            x = int(size // 2 + (i - size // 2) * cos_a)
            y = int(size // 2 + (i - size // 2) * sin_a)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1

        kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
        return cv2.filter2D(image, -1, kernel)

    return image


def add_noise(image, noise_type='gaussian', intensity=0.05):
    """
    Add noise for robustness.
    """
    img = image.copy().astype(np.float32)

    if noise_type == 'gaussian':
        noise = np.random.randn(*img.shape) * (intensity * 255)
        img = img + noise

    elif noise_type == 'salt_pepper':
        prob = intensity
        salt = np.random.random(img.shape[:2]) < prob / 2
        pepper = np.random.random(img.shape[:2]) < prob / 2
        img[salt] = 255
        img[pepper] = 0

    return np.clip(img, 0, 255).astype(np.uint8)


def rotate_image_and_bbox(image, bboxes, angle):
    """
    Rotate image and adjust bounding boxes accordingly.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image size to contain rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust rotation matrix
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

    # Adjust bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        class_id, x_c, y_c, bw, bh = bbox

        # Convert to pixel coordinates
        x_c_px = x_c * w
        y_c_px = y_c * h

        # Rotate center point
        point = np.array([[x_c_px, y_c_px, 1]])
        new_point = M @ point.T

        # Convert back to normalized coordinates
        new_x_c = new_point[0, 0] / new_w
        new_y_c = new_point[1, 0] / new_h

        # Scale width/height
        new_bw = bw * (w / new_w)
        new_bh = bh * (h / new_h)

        # Validate and clip
        if 0.01 < new_x_c < 0.99 and 0.01 < new_y_c < 0.99:
            new_bw = min(new_bw, min(new_x_c, 1 - new_x_c) * 2)
            new_bh = min(new_bh, min(new_y_c, 1 - new_y_c) * 2)
            new_bboxes.append([class_id, new_x_c, new_y_c, new_bw, new_bh])

    return rotated, new_bboxes


def flip_image_and_bbox(image, bboxes, flip_code):
    """
    Flip image and adjust bounding boxes.
    flip_code: 0=vertical, 1=horizontal, -1=both
    """
    flipped = cv2.flip(image, flip_code)

    new_bboxes = []
    for bbox in bboxes:
        class_id, x_c, y_c, bw, bh = bbox

        if flip_code == 1:  # Horizontal
            x_c = 1 - x_c
        elif flip_code == 0:  # Vertical
            y_c = 1 - y_c
        elif flip_code == -1:  # Both
            x_c = 1 - x_c
            y_c = 1 - y_c

        new_bboxes.append([class_id, x_c, y_c, bw, bh])

    return flipped, new_bboxes


def read_yolo_labels(label_path):
    """Read YOLO format labels from file."""
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    bboxes.append([class_id, x_c, y_c, w, h])
    return bboxes


def write_yolo_labels(label_path, bboxes):
    """Write YOLO format labels to file."""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id = int(bbox[0])
            x_c, y_c, w, h = bbox[1:5]
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def augment_single_image(img_path, label_path, output_img_dir, output_label_dir,
                         augmentations_per_image=3, aug_config=None):
    """
    Apply augmentations to a single image and its labels.
    """
    if aug_config is None:
        aug_config = {
            'fisheye_prob': 0.3,
            'fisheye_strength_range': (0.2, 0.6),
            'color_jitter_prob': 0.7,
            'blur_prob': 0.3,
            'noise_prob': 0.2,
            'rotation_prob': 0.5,
            'rotation_range': (-15, 15),
            'flip_prob': 0.5,
        }

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read {img_path}")
        return []

    # Read labels
    bboxes = read_yolo_labels(label_path)

    base_name = Path(img_path).stem
    generated_files = []

    for aug_idx in range(augmentations_per_image):
        aug_image = image.copy()
        aug_bboxes = [b.copy() for b in bboxes]
        aug_suffix = f"_aug{aug_idx}"

        # Apply augmentations randomly

        # 1. Fisheye distortion
        if random.random() < aug_config['fisheye_prob']:
            strength = random.uniform(*aug_config['fisheye_strength_range'])
            if random.random() < 0.5:
                strength = -strength  # Pincushion
            aug_image = fisheye_distortion(aug_image, strength)
            aug_bboxes = [adjust_bbox_for_fisheye(b, aug_image.shape, strength) for b in aug_bboxes]
            aug_suffix += "_fish"

        # 2. Rotation
        if random.random() < aug_config['rotation_prob']:
            angle = random.uniform(*aug_config['rotation_range'])
            aug_image, aug_bboxes = rotate_image_and_bbox(aug_image, aug_bboxes, angle)
            aug_suffix += f"_rot{int(angle)}"

        # 3. Flip
        if random.random() < aug_config['flip_prob']:
            flip_code = random.choice([0, 1, -1])
            aug_image, aug_bboxes = flip_image_and_bbox(aug_image, aug_bboxes, flip_code)
            aug_suffix += f"_flip{flip_code}"

        # 4. Color jitter (doesn't affect bboxes)
        if random.random() < aug_config['color_jitter_prob']:
            aug_image = color_jitter(aug_image)
            aug_suffix += "_color"

        # 5. Blur
        if random.random() < aug_config['blur_prob']:
            blur_type = random.choice(['gaussian', 'motion'])
            aug_image = apply_blur(aug_image, blur_type)
            aug_suffix += f"_{blur_type[:3]}"

        # 6. Noise
        if random.random() < aug_config['noise_prob']:
            noise_type = random.choice(['gaussian', 'salt_pepper'])
            intensity = random.uniform(0.01, 0.05)
            aug_image = add_noise(aug_image, noise_type, intensity)
            aug_suffix += "_noise"

        # Filter out invalid bboxes
        valid_bboxes = []
        for bbox in aug_bboxes:
            class_id, x_c, y_c, w, h = bbox
            if 0.01 < x_c < 0.99 and 0.01 < y_c < 0.99 and w > 0.01 and h > 0.01:
                valid_bboxes.append(bbox)

        # Save augmented image and labels
        output_img_name = f"{base_name}{aug_suffix}.jpg"
        output_label_name = f"{base_name}{aug_suffix}.txt"

        output_img_path = os.path.join(output_img_dir, output_img_name)
        output_label_path = os.path.join(output_label_dir, output_label_name)

        cv2.imwrite(output_img_path, aug_image)
        write_yolo_labels(output_label_path, valid_bboxes)

        generated_files.append((output_img_path, output_label_path))

    return generated_files


def augment_dataset(input_dir, output_dir, augmentations_per_image=3,
                    splits=['train'], num_workers=4, aug_config=None):
    """
    Augment entire dataset with parallel processing.

    Args:
        input_dir: Root directory of YOLO dataset (contains train/val/test folders)
        output_dir: Output directory for augmented dataset
        augmentations_per_image: Number of augmented versions per original image
        splits: Which splits to augment (e.g., ['train', 'val'])
        num_workers: Number of parallel workers
        aug_config: Augmentation configuration dictionary
    """
    print(f"Starting dataset augmentation...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per image: {augmentations_per_image}")

    for split in splits:
        input_img_dir = os.path.join(input_dir, split, 'images')
        input_label_dir = os.path.join(input_dir, split, 'labels')
        output_img_dir = os.path.join(output_dir, split, 'images')
        output_label_dir = os.path.join(output_dir, split, 'labels')

        if not os.path.exists(input_img_dir):
            print(f"Warning: {input_img_dir} does not exist, skipping...")
            continue

        # Create output directories
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # Copy original images and labels first
        print(f"\nProcessing {split} split...")
        image_files = [f for f in os.listdir(input_img_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Found {len(image_files)} images in {split}")

        # Copy originals
        for img_file in image_files:
            src_img = os.path.join(input_img_dir, img_file)
            dst_img = os.path.join(output_img_dir, img_file)
            shutil.copy2(src_img, dst_img)

            label_file = Path(img_file).stem + '.txt'
            src_label = os.path.join(input_label_dir, label_file)
            dst_label = os.path.join(output_label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        print(f"Copied {len(image_files)} original images")

        # Generate augmented images in parallel
        total_augmented = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for img_file in image_files:
                img_path = os.path.join(input_img_dir, img_file)
                label_path = os.path.join(input_label_dir, Path(img_file).stem + '.txt')

                future = executor.submit(
                    augment_single_image,
                    img_path, label_path,
                    output_img_dir, output_label_dir,
                    augmentations_per_image, aug_config
                )
                futures.append(future)

            # Process results
            for future in as_completed(futures):
                try:
                    generated = future.result()
                    total_augmented += len(generated)
                except Exception as e:
                    print(f"Error during augmentation: {e}")

        print(f"Generated {total_augmented} augmented images for {split}")
        final_count = len(os.listdir(output_img_dir))
        print(f"Total images in {split}: {final_count}")

    print("\nAugmentation complete!")


def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset with fisheye and standard transforms')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for augmented dataset')
    parser.add_argument('--augmentations', type=int, default=3, help='Number of augmentations per image')
    parser.add_argument('--splits', nargs='+', default=['train'], help='Splits to augment')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')

    # Augmentation probabilities
    parser.add_argument('--fisheye-prob', type=float, default=0.3, help='Fisheye augmentation probability')
    parser.add_argument('--color-prob', type=float, default=0.7, help='Color jitter probability')
    parser.add_argument('--blur-prob', type=float, default=0.3, help='Blur probability')
    parser.add_argument('--noise-prob', type=float, default=0.2, help='Noise probability')
    parser.add_argument('--rotation-prob', type=float, default=0.5, help='Rotation probability')
    parser.add_argument('--flip-prob', type=float, default=0.5, help='Flip probability')

    args = parser.parse_args()

    aug_config = {
        'fisheye_prob': args.fisheye_prob,
        'fisheye_strength_range': (0.2, 0.6),
        'color_jitter_prob': args.color_prob,
        'blur_prob': args.blur_prob,
        'noise_prob': args.noise_prob,
        'rotation_prob': args.rotation_prob,
        'rotation_range': (-15, 15),
        'flip_prob': args.flip_prob,
    }

    augment_dataset(
        args.input, args.output,
        augmentations_per_image=args.augmentations,
        splits=args.splits,
        num_workers=args.workers,
        aug_config=aug_config
    )


if __name__ == '__main__':
    main()
