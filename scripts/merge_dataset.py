"""
Merge annotated images from an external fisheye dataset into the project dataset.

Only images that have a matching label file are copied — unannotated images are
skipped as they provide no supervised signal.

Files are renamed with a source prefix (default: "fisheye_v1_") to:
  - Avoid any filename collision with existing data
  - Make the data origin traceable

Usage:
    python scripts/merge_dataset.py [--dry-run] [--split train]

    # Preview what would be copied without touching any files:
    python scripts/merge_dataset.py --dry-run

    # Execute the merge into the train split:
    python scripts/merge_dataset.py

    # Merge into the val split instead:
    python scripts/merge_dataset.py --split val
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASET_DST = REPO_ROOT / "datasets" / "cylindrical_joint"

SOURCE_ROOT = Path.home() / "Downloads" / "Indoor Fire detection" \
              / "camera" / "dataset fire fisheye v1" / "dataset fire fisheye"

SOURCE_IMAGES = SOURCE_ROOT / "images"
SOURCE_LABELS = SOURCE_ROOT / "labels"


def parse_args():
    p = argparse.ArgumentParser(description="Merge external fisheye dataset into project dataset")
    p.add_argument(
        "--split", choices=["train", "val"], default="train",
        help="Destination split (default: train)"
    )
    p.add_argument(
        "--prefix", default="fisheye_v1_",
        help="Prefix added to every copied filename to identify its source "
             "(default: 'fisheye_v1_')"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be copied without writing any files"
    )
    return p.parse_args()


def main():
    args = parse_args()

    dst_images = DATASET_DST / args.split / "images"
    dst_labels = DATASET_DST / args.split / "labels"

    if not args.dry_run:
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

    # --- Discover annotated pairs ---
    label_files = sorted(SOURCE_LABELS.glob("*.txt"))
    pairs = []
    missing_images = []

    for lbl_path in label_files:
        stem = lbl_path.stem
        img_path = SOURCE_IMAGES / f"{stem}.jpg"
        if not img_path.exists():
            # Try other extensions just in case
            for ext in (".png", ".jpeg", ".JPG", ".PNG"):
                candidate = SOURCE_IMAGES / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                missing_images.append(stem)
                continue
        pairs.append((img_path, lbl_path))

    print(f"Source         : {SOURCE_ROOT}")
    print(f"Destination    : {DATASET_DST / args.split}")
    print(f"Annotated pairs: {len(pairs)}")
    print(f"Missing images : {len(missing_images)}")
    if missing_images:
        print(f"  (label files with no matching image — skipped: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''})")
    print(f"Prefix         : '{args.prefix}'")
    print(f"Dry run        : {args.dry_run}")
    print()

    # --- Check for collisions before writing anything ---
    collisions = []
    for img_src, lbl_src in pairs:
        img_dst = dst_images / f"{args.prefix}{img_src.name}"
        lbl_dst = dst_labels / f"{args.prefix}{lbl_src.name}"
        if img_dst.exists() or lbl_dst.exists():
            collisions.append(img_src.stem)

    if collisions:
        print(f"[WARN] {len(collisions)} file(s) already exist in destination with this prefix.")
        print("       They will be overwritten. Use a different --prefix to avoid this.")
        print(f"       First few: {collisions[:5]}")
        print()

    # --- Copy ---
    copied = 0
    skipped_empty = 0

    for img_src, lbl_src in pairs:
        # Skip empty label files (no annotations = background-only, not useful here)
        if lbl_src.stat().st_size == 0:
            skipped_empty += 1
            continue

        img_dst = dst_images / f"{args.prefix}{img_src.name}"
        lbl_dst = dst_labels / f"{args.prefix}{lbl_src.name}"

        if args.dry_run:
            print(f"  [DRY] {img_src.name} → {img_dst.name}")
        else:
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
            copied += 1

        if copied and copied % 100 == 0:
            print(f"  Copied {copied} / {len(pairs)} ...")

    # --- Summary ---
    print()
    if args.dry_run:
        print(f"[DRY RUN] Would copy {len(pairs) - skipped_empty} annotated pairs "
              f"({skipped_empty} empty label files skipped).")
    else:
        print(f"[DONE] Copied {copied} image+label pairs into {DATASET_DST / args.split}")
        print(f"       Empty label files skipped: {skipped_empty}")

        # Recount totals
        n_train_img = len(list((DATASET_DST / "train" / "images").glob("*")))
        n_val_img   = len(list((DATASET_DST / "val"   / "images").glob("*")))
        n_train_lbl = len(list((DATASET_DST / "train" / "labels").glob("*.txt")))
        n_val_lbl   = len(list((DATASET_DST / "val"   / "labels").glob("*.txt")))
        print()
        print("Updated dataset totals:")
        print(f"  train: {n_train_img} images, {n_train_lbl} labels")
        print(f"  val  : {n_val_img} images, {n_val_lbl} labels")

        # Class distribution of newly added labels
        fire_count = smoke_count = 0
        for img_src, lbl_src in pairs:
            if lbl_src.stat().st_size == 0:
                continue
            for line in lbl_src.read_text().strip().splitlines():
                cls = line.split()[0]
                if cls == "0":
                    fire_count += 1
                elif cls == "1":
                    smoke_count += 1
        print()
        print("Added annotations (new data only):")
        print(f"  fire  (class 0): {fire_count} boxes")
        print(f"  smoke (class 1): {smoke_count} boxes")


if __name__ == "__main__":
    main()
