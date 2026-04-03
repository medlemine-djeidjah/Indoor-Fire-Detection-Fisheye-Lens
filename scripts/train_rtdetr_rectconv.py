"""
train_rtdetr_rectconv.py
------------------------
Fine-tune RT-DETR on fisheye fire/smoke data with RectConv layers active.

Pipeline
--------
1.  Build a Camera from your fisheye lens parameters (FOV or calibration JSON).
2.  Compute the RectConv distortion map (once, then cached).
3.  Load the base RT-DETR model.
4.  Patch ALL Conv2d layers → RectifyConv2d (weights unchanged, sampling grid
    now geometrically correct for your fisheye lens).
5.  Fine-tune the patched model on your fire/smoke dataset.
6.  Save the fine-tuned weights + update the model registry.

Why fine-tune after patching?
------------------------------
The base model was trained on perspective images (pinhole).  After patching,
the sampling grid is corrected for fisheye geometry, but the model has never
seen fire/smoke.  Fine-tuning on your fisheye-aware augmented dataset gives
the best of both worlds:
  • RectConv-correct geometry (no distortion artifacts)
  • Fire/smoke domain expertise from your training data

This builds on train_rtdetr.py — all the same versioning and registry logic
applies.  Weights are saved as:
    whights/rtdetr-l-rectconv_v{N}_{date}.pt

Usage
------
    # Approximate camera from FOV (no calibration required)
    python scripts/train_rtdetr_rectconv.py \\
        --data /abs/path/dataset/data.yaml \\
        --fov 180 --width 640 --height 640

    # Multiple datasets
    python scripts/train_rtdetr_rectconv.py \\
        --data /path/ds_A/data.yaml /path/ds_B/data.yaml \\
        --fov 180 --epochs 50 --batch 8

    # Precise calibrated camera
    python scripts/train_rtdetr_rectconv.py \\
        --data /path/dataset/data.yaml \\
        --camera-json cameras/my_camera.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# ── Shared paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent.resolve()
PROJECT_ROOT  = SCRIPT_DIR.parent
WEIGHTS_DIR   = PROJECT_ROOT / "whights"
RESULTS_ROOT  = PROJECT_ROOT / "training_results"
REGISTRY_PATH = WEIGHTS_DIR / "registry.json"
SUMMARY_PATH  = RESULTS_ROOT / "summary.json"
CACHE_DIR     = PROJECT_ROOT / "cameras" / "cache"

DEFAULT_MODEL  = "rtdetr-l.pt"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH  = 8
DEFAULT_IMGSZ  = 640
DEFAULT_FOV    = 180.0


# ──────────────────────────────────────────────────────────────────────────────
# Registry / summary helpers (same as train_rtdetr.py)
# ──────────────────────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": []}


def _save_registry(reg: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def _load_summary() -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {"runs": []}


def _save_summary(s: dict) -> None:
    with open(SUMMARY_PATH, "w") as f:
        json.dump(s, f, indent=2)


def _next_version(registry: dict, prefix: str) -> int:
    versions = [m["version"] for m in registry["models"]
                if m.get("id", "").startswith(prefix)]
    return max(versions, default=0) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────────

def _train_single(
    data_yaml: str,
    distmap,                   # torch.Tensor
    camera_desc: str,
    args: argparse.Namespace,
    registry: dict,
) -> dict[str, Any]:
    """Train one RT-DETR+RectConv run on a single dataset."""

    from ultralytics import RTDETR
    import torch

    # ── Versioning ─────────────────────────────────────────────────────────
    arch_key    = Path(args.model).stem.lower() + "-rectconv"   # e.g. rtdetr-l-rectconv
    arch_label  = arch_key.upper()
    version     = _next_version(registry, arch_key)
    today       = date.today().isoformat()
    run_name    = f"{arch_key}_v{version}_{today}"
    weights_name = f"{run_name}.pt"
    run_dir     = RESULTS_ROOT / run_name

    data_path    = Path(data_yaml).resolve()
    dataset_name = data_path.parent.name

    print(f"\n{'='*66}")
    print(f"  FireGuard AI — {arch_label}  ·  v{version}")
    print(f"  Dataset    : {data_path}")
    print(f"  Camera     : {camera_desc}")
    print(f"  Date       : {today}")
    print(f"  Epochs     : {args.epochs}  Batch: {args.batch}  ImgSz: {args.imgsz}")
    print(f"  Output     : {run_dir}")
    print(f"{'='*66}\n")

    # ── Load base model ───────────────────────────────────────────────────
    print("[1/3] Loading base RT-DETR model…")
    model = RTDETR(args.model)

    # ── Patch with RectConv ───────────────────────────────────────────────
    print("[2/3] Patching Conv2d layers with RectConv…")
    # Access the underlying nn.Module inside the Ultralytics wrapper
    nn_module = model.model
    from rectconv_adapter import patch_model
    patch_model(nn_module, distmap)

    # Count patched layers
    from conv.RectifyConv2d import RectifyConv2d
    n_patched = sum(1 for m in nn_module.modules()
                    if isinstance(m, RectifyConv2d))
    print(f"     Patched {n_patched} Conv2d layers → RectifyConv2d")

    # ── Train ─────────────────────────────────────────────────────────────
    print("[3/3] Fine-tuning on fisheye dataset…\n")
    results = model.train(
        data    = str(data_path),
        epochs  = args.epochs,
        batch   = args.batch,
        imgsz   = args.imgsz,
        device  = args.device if args.device else None,
        project = str(RESULTS_ROOT),
        name    = run_name,
        exist_ok = False,
        # Fire-specific augmentations
        hsv_h   = 0.015,
        hsv_s   = 0.9,
        hsv_v   = 0.6,
        degrees = 10.0,
        flipud  = 0.1,
        fliplr  = 0.5,
        mosaic  = 1.0,
        mixup   = 0.1,
        # Regularisation
        weight_decay  = 0.0001,
        warmup_epochs = 3,
        patience      = 15,
        save          = True,
        save_period   = 10,
        plots         = True,
    )

    # ── Copy best weights → whights/ ──────────────────────────────────────
    best_src = run_dir / "weights" / "best.pt"
    best_dst = WEIGHTS_DIR / weights_name
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n[✔] Best weights → {best_dst}")
    else:
        print(f"\n[!] WARNING: best.pt not found at {best_src}")
        best_dst = Path("(not found)")

    # ── Extract metrics ───────────────────────────────────────────────────
    mAP50    = 0.0
    mAP50_95 = 0.0
    try:
        mAP50    = float(results.results_dict.get("metrics/mAP50(B)",    0.0))
        mAP50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
    except Exception:
        pass

    entry: dict[str, Any] = {
        "id":           f"{arch_key}_v{version}",
        "architecture": arch_label,
        "version":      version,
        "trained_date": today,
        "fisheye":      True,
        "rectconv":     True,
        "camera":       camera_desc,
        "dataset":      str(data_path),
        "dataset_name": dataset_name,
        "epochs":       args.epochs,
        "imgsz":        args.imgsz,
        "mAP50":        round(mAP50,    4),
        "mAP50_95":     round(mAP50_95, 4),
        "weights_path": f"whights/{weights_name}",
        "run_dir":      f"training_results/{run_name}",
    }

    print(f"\n{'='*66}")
    print(f"  Run complete  ·  {arch_label} v{version}")
    print(f"  mAP@50       : {mAP50:.4f}")
    print(f"  mAP@50-95    : {mAP50_95:.4f}")
    print(f"  Weights      : {best_dst}")
    print(f"  Results      : {run_dir}")
    print(f"{'='*66}\n")

    return entry


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Imports ──────────────────────────────────────────────────────────
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics") from e

    try:
        from rectconv_adapter import (
            make_camera_from_fov,
            make_camera_from_json,
            build_distortion_map,
        )
    except ImportError as e:
        raise SystemExit(f"rectconv_adapter import failed: {e}") from e

    # ── Validate data.yaml paths ─────────────────────────────────────────
    import yaml as _yaml
    for data_yaml in args.data:
        p = Path(data_yaml).resolve()
        if not p.exists():
            raise SystemExit(f"[ERROR] data.yaml not found: {p}")
        with open(p) as f:
            cfg = _yaml.safe_load(f)
        ds_path = cfg.get("path", "")
        if str(ds_path).strip() in ("your_path", "", ".", "None", "null"):
            raise SystemExit(
                f"[ERROR] data.yaml has a placeholder path: '{ds_path}'\n"
                f"  Fix: set 'path' to the absolute dataset directory in {p}"
            )

    # ── Build camera and distortion map ──────────────────────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.camera_json:
        cam = make_camera_from_json(args.camera_json)
        camera_desc = f"JSON:{Path(args.camera_json).name}"
    else:
        w = args.width  if args.width  else args.imgsz
        h = args.height if args.height else args.imgsz
        cam = make_camera_from_fov(w=w, h=h, fov_deg=args.fov)
        camera_desc = f"approx-equidistant FOV={args.fov}° {w}×{h}px"

    print(f"\n[camera]  {camera_desc}")
    print(f"[camera]  k1={cam.lens.coefficients[0]:.3f}  "
          f"k2={cam.lens.coefficients[1]:.3f}  "
          f"size={cam.width}×{cam.height}")

    distmap = build_distortion_map(cam, cache_path=str(CACHE_DIR))

    # ── Run one training job per dataset ─────────────────────────────────
    registry       = _load_registry()
    summary        = _load_summary()
    session_entries: list[dict] = []

    for i, data_yaml in enumerate(args.data, 1):
        if len(args.data) > 1:
            print(f"\n[{i}/{len(args.data)}] Dataset: {data_yaml}")
        entry = _train_single(data_yaml, distmap, camera_desc, args, registry)
        registry["models"].append(entry)
        _save_registry(registry)
        summary["runs"].append(entry)
        _save_summary(summary)
        session_entries.append(entry)

    # ── Comparison summary ────────────────────────────────────────────────
    if len(session_entries) > 1:
        print(f"\n{'='*80}")
        print("  MULTI-DATASET COMPARISON  (RectConv)")
        print(f"  {'ID':<30} {'Dataset':<22} {'mAP@50':>8} {'mAP@50-95':>10}")
        print(f"  {'-'*30} {'-'*22} {'-'*8} {'-'*10}")
        best = max(range(len(session_entries)), key=lambda i: session_entries[i]["mAP50"])
        for i, e in enumerate(session_entries):
            m = "  ◄ best" if i == best else ""
            print(f"  {e['id']:<30} {e['dataset_name']:<22} "
                  f"{e['mAP50']:>8.4f} {e['mAP50_95']:>10.4f}{m}")
        print(f"{'='*80}\n")

    print(f"[✔] Registry  : {REGISTRY_PATH}")
    print(f"[✔] Summary   : {SUMMARY_PATH}")
    print(f"[✔] Results   : {RESULTS_ROOT}/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune RT-DETR + RectConv for fisheye fire/smoke detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Approximate camera from FOV
  python scripts/train_rtdetr_rectconv.py \\
      --data /abs/path/dataset/data.yaml --fov 180

  # Multiple datasets, GPU 0
  python scripts/train_rtdetr_rectconv.py \\
      --data /path/ds_A/data.yaml /path/ds_B/data.yaml \\
      --fov 180 --epochs 50 --device 0

  # Precise calibrated camera
  python scripts/train_rtdetr_rectconv.py \\
      --data /path/dataset/data.yaml \\
      --camera-json cameras/my_camera.json
        """,
    )
    p.add_argument("--data", nargs="+", required=True, metavar="DATA_YAML",
                   help="Absolute path(s) to data.yaml file(s).")
    p.add_argument("--model",  default=DEFAULT_MODEL,
                   help=f"Base RT-DETR weights  (default: {DEFAULT_MODEL})")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"Training epochs per run  (default: {DEFAULT_EPOCHS})")
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                   help=f"Batch size  (default: {DEFAULT_BATCH})")
    p.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ,
                   help=f"Input image size  (default: {DEFAULT_IMGSZ})")
    p.add_argument("--device", default="",
                   help="Device: '0', 'cpu', etc.  (default: auto)")

    cam_group = p.add_argument_group("Camera (choose one)")
    cam_exclusive = cam_group.add_mutually_exclusive_group()
    cam_exclusive.add_argument("--camera-json", metavar="PATH",
                   help="Path to a calibration JSON file (most accurate).")
    cam_exclusive.add_argument("--fov", type=float, default=DEFAULT_FOV,
                   help=f"Fisheye full FOV in degrees — uses equidistant approximation  "
                        f"(default: {DEFAULT_FOV})")
    p.add_argument("--width",  type=int, default=None,
                   help="Image width for FOV approximation  (default: --imgsz value)")
    p.add_argument("--height", type=int, default=None,
                   help="Image height for FOV approximation  (default: --imgsz value)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args)
