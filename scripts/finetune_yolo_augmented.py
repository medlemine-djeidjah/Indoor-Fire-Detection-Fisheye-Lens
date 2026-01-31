"""
Fine-tune YOLO model on augmented Indoor Fire/Smoke dataset.

This script:
1. Augments the Indoor Fire Smoke dataset using fisheye-style transforms
2. Creates the necessary data.yaml configuration
3. Fine-tunes a YOLO model on the augmented data

Usage:
    python finetune_yolo_augmented.py --epochs 50 --model yolov8n.pt
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Add current directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from augment_dataset import augment_dataset


def create_data_yaml(output_dir, dataset_path, classes=['fire', 'smoke']):
    """
    Create YOLO data.yaml configuration file.
    """
    yaml_content = f"""# Fire/Smoke Detection Dataset - Augmented
# Auto-generated configuration file

path: {dataset_path}

train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created data.yaml at {yaml_path}")
    return yaml_path


def run_augmentation(input_dataset, output_dataset, augmentations_per_image=3,
                     splits=['train', 'valid'], num_workers=4, aug_config=None):
    """
    Run the augmentation pipeline on the dataset.
    """
    print("=" * 60)
    print("STEP 1: DATA AUGMENTATION")
    print("=" * 60)

    # Check if augmented dataset already exists
    if os.path.exists(output_dataset):
        print(f"Warning: Output directory exists: {output_dataset}")
        response = input("Delete and regenerate? [y/N]: ").strip().lower()
        if response == 'y':
            shutil.rmtree(output_dataset)
        else:
            print("Using existing augmented dataset")
            return

    augment_dataset(
        input_dir=input_dataset,
        output_dir=output_dataset,
        augmentations_per_image=augmentations_per_image,
        splits=splits,
        num_workers=num_workers,
        aug_config=aug_config
    )


def finetune_yolo(data_yaml, model_name='yolov8n.pt', epochs=50, imgsz=640,
                  batch_size=16, patience=15, device=None, project_name='fire_smoke_augmented'):
    """
    Fine-tune YOLO model on the dataset.
    """
    print("\n" + "=" * 60)
    print("STEP 2: YOLO FINE-TUNING")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    # Determine device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = 0  # Use first GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("Warning: CUDA not available, using CPU (training will be slow)")

    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)

    print(f"Training configuration:")
    print(f"  - Data: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Device: {device}")

    # Training with augmentation settings optimized for fire/smoke
    # Note: YOLO's built-in augmentations complement our offline augmentations
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        device=device,
        project=os.path.join(SCRIPT_DIR, 'runs', 'train'),
        name=project_name,
        verbose=True,
        plots=True,

        # YOLO built-in augmentations (complement our offline augmentations)
        hsv_h=0.015,    # Hue variation (fire color)
        hsv_s=0.7,      # Saturation variation
        hsv_v=0.4,      # Value variation
        degrees=10.0,   # Rotation (less than our offline to avoid doubling)
        translate=0.1,  # Translation
        scale=0.5,      # Scale variation
        shear=0.0,      # No shear
        flipud=0.0,     # No vertical flip (fire goes up)
        fliplr=0.5,     # Horizontal flip
        mosaic=1.0,     # Mosaic augmentation
        mixup=0.1,      # Mixup augmentation

        # Close mosaic in final epochs for better convergence
        close_mosaic=10,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model weights saved to: {results.save_dir}")

    return model, results


def validate_model(model, data_yaml):
    """
    Run validation on the trained model.
    """
    print("\n" + "=" * 60)
    print("STEP 3: MODEL VALIDATION")
    print("=" * 60)

    metrics = model.val(data=data_yaml)

    print("\nValidation Results:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")

    return metrics


def export_model(model, formats=['onnx']):
    """
    Export model to different formats.
    """
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EXPORT")
    print("=" * 60)

    for fmt in formats:
        print(f"Exporting to {fmt.upper()}...")
        try:
            path = model.export(format=fmt)
            print(f"  Exported to: {path}")
        except Exception as e:
            print(f"  Error exporting to {fmt}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLO on augmented fire/smoke dataset')

    # Dataset paths
    parser.add_argument('--input-dataset', type=str,
                        default=os.path.join(SCRIPT_DIR, '../datasets/Indoor-Fire-Smoke'),
                        help='Path to original dataset')
    parser.add_argument('--output-dataset', type=str,
                        default=os.path.join(SCRIPT_DIR, '../datasets/augmented_dataset'),
                        help='Path for augmented dataset output')

    # Augmentation settings
    parser.add_argument('--augmentations', type=int, default=3,
                        help='Number of augmented versions per image')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for augmentation')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Skip augmentation step (use existing augmented data)')

    # Training settings
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Base YOLO model to fine-tune (e.g., yolov8n.pt, yolo12n.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--project-name', type=str, default='fire_smoke_augmented',
                        help='Project name for saving results')

    # Additional options
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation on existing model')
    parser.add_argument('--export', nargs='*', default=[],
                        help='Export formats (e.g., onnx, torchscript)')

    args = parser.parse_args()

    # Resolve paths
    args.input_dataset = os.path.abspath(args.input_dataset)
    args.output_dataset = os.path.abspath(args.output_dataset)

    print("=" * 60)
    print("YOLO FIRE/SMOKE DETECTION - AUGMENTED TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input dataset: {args.input_dataset}")
    print(f"  Output dataset: {args.output_dataset}")
    print(f"  Base model: {args.model}")
    print(f"  Epochs: {args.epochs}")

    # Augmentation configuration based on fisheye tutorial insights
    aug_config = {
        'fisheye_prob': 0.3,              # Apply fisheye distortion 30% of the time
        'fisheye_strength_range': (0.2, 0.6),  # Moderate distortion
        'color_jitter_prob': 0.7,         # Important for fire color variations
        'blur_prob': 0.3,                 # Motion blur for dynamic fire
        'noise_prob': 0.2,                # Robustness to noise
        'rotation_prob': 0.5,             # Rotation
        'rotation_range': (-15, 15),      # Degrees
        'flip_prob': 0.5,                 # Horizontal flip
    }

    # Step 1: Augmentation
    if not args.skip_augmentation and not args.validate_only:
        run_augmentation(
            input_dataset=args.input_dataset,
            output_dataset=args.output_dataset,
            augmentations_per_image=args.augmentations,
            splits=['train', 'valid'],  # Only augment train and val, keep test clean
            num_workers=args.workers,
            aug_config=aug_config
        )

        # Copy test set without augmentation
        test_src = os.path.join(args.input_dataset, 'test')
        test_dst = os.path.join(args.output_dataset, 'test')
        if os.path.exists(test_src) and not os.path.exists(test_dst):
            print("\nCopying test set (no augmentation)...")
            shutil.copytree(test_src, test_dst)

    # Create data.yaml
    data_yaml = create_data_yaml(
        output_dir=args.output_dataset,
        dataset_path=args.output_dataset,
        classes=['fire', 'smoke']
    )

    # Step 2: Fine-tune YOLO
    if not args.validate_only:
        model, results = finetune_yolo(
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
            project_name=args.project_name
        )

        # Step 3: Validate
        validate_model(model, data_yaml)

        # Step 4: Export (optional)
        if args.export:
            export_model(model, formats=args.export)

        # Save final model
        final_model_path = os.path.join(SCRIPT_DIR, f'{args.project_name}_final.pt')
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
