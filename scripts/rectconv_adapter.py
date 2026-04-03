"""
rectconv_adapter.py
-------------------
Integration bridge between the RectConv library (third_party/RectConv) and
our RT-DETR / YOLO fire-detection pipeline.

What RectConv does
------------------
Standard convolution samples a regular pixel grid around each position.
RectConv replaces this with a geometrically-warped grid computed from the
fisheye camera model: each kernel "sees" a locally perspective-corrected
patch, exactly as if the camera were a pinhole.  The key properties:

  • No global rectification → full fisheye FOV is preserved (no dead zones)
  • Model weights are UNCHANGED — a perspective-trained model works as-is
  • Detections land directly in fisheye coordinates — no back-projection step
  • The offset tensor is computed ONCE per camera, then reused for every frame

Camera model used
-----------------
RadialPolyCamProjection (Kannala-Brandt polynomial):
    r = k1·θ + k2·θ² + k3·θ³ + k4·θ⁴

For an equidistant fisheye (most wide-angle security cameras):
    r = k1·θ   (k2=k3=k4=0, k1 = r_max / (fov_rad/2))

Public API
----------
    from scripts.rectconv_adapter import (
        make_camera_from_fov,
        make_camera_from_json,
        build_distortion_map,
        patch_model,
    )

    # Build an approximate camera from your lens FOV
    cam      = make_camera_from_fov(w=640, h=640, fov_deg=180)

    # (or load a precise calibration JSON)
    cam      = make_camera_from_json("cameras/my_camera.json")

    # Compute offset map — slow first time, cached to disk afterwards
    distmap  = build_distortion_map(cam, cache_path="cameras/cache/distmap.pt")

    # Patch any PyTorch model in-place (weights unchanged)
    patch_model(my_yolo_nn_module, distmap)

    # The patched model now takes a fisheye image and returns detections
    # in fisheye pixel coordinates — no extra post-processing needed.

CLI (generate a calibration JSON from FOV)
------------------------------------------
    python scripts/rectconv_adapter.py \\
        --gen-json --fov 180 --width 640 --height 640 \\
        --output cameras/my_cam.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Import RectConv from third_party ─────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RECTCONV_DIR = PROJECT_ROOT / "third_party" / "RectConv"

for _p in [str(RECTCONV_DIR), str(RECTCONV_DIR / "scripts"), str(RECTCONV_DIR / "conv")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scripts.projection import (       # noqa: E402
        Camera,
        RadialPolyCamProjection,
        read_cam_from_json,
    )
    from scripts.util import generate_offset, convert_to_rectconv  # noqa: E402
    _RECTCONV_OK = True
except ImportError as _e:
    _RECTCONV_OK = False
    _RECTCONV_ERR = str(_e)


def _require_rectconv() -> None:
    if not _RECTCONV_OK:
        raise RuntimeError(
            f"RectConv library not available: {_RECTCONV_ERR}\n"
            f"Expected at: {RECTCONV_DIR}\n"
            "Run: git clone https://github.com/RoboticImaging/RectConv.git "
            "third_party/RectConv"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Camera construction
# ──────────────────────────────────────────────────────────────────────────────

def make_camera_from_fov(
    w: int,
    h: int,
    fov_deg: float = 180.0,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
) -> "Camera":
    """
    Build an approximate fisheye Camera object from image dimensions and FOV.

    Uses the equidistant model (r = k1·θ), which is accurate for most
    wide-angle IP-security fisheye cameras.  For a more accurate model,
    calibrate your lens and use make_camera_from_json() instead.

    Parameters
    ----------
    w, h        : image width and height in pixels.
    fov_deg     : full field-of-view of the fisheye lens in degrees.
    cx_offset   : horizontal principal-point offset from image centre (pixels).
    cy_offset   : vertical principal-point offset from image centre (pixels).
    """
    _require_rectconv()
    from scipy.spatial.transform import Rotation as SciRot

    fov_rad = np.deg2rad(fov_deg)
    r_max   = min(w, h) / 2.0          # inscribed-circle radius
    k1      = r_max / (fov_rad / 2.0)  # equidistant: r = k1·θ

    cam = Camera(
        lens            = RadialPolyCamProjection([k1, 0.0, 0.0, 0.0]),
        translation     = np.array([0.0, 0.0, 0.0]),
        rotation        = SciRot.from_euler("xyz", [0, 0, 0]).as_matrix(),
        size            = (w, h),
        principle_point = (cx_offset, cy_offset),
        aspect_ratio    = 1.0,
    )
    return cam


def make_camera_from_json(path: str | Path) -> "Camera":
    """
    Load a Camera from a calibration JSON file.

    JSON format (same as RectConv WoodScape format):
    {
      "intrinsic": {
        "k1": 203.7, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "width": 640, "height": 640,
        "cx_offset": 0.0, "cy_offset": 0.0, "aspect_ratio": 1.0
      },
      "extrinsic": {
        "quaternion": [0, 0, 0, 1],    // (x, y, z, w)
        "translation": [0, 0, 0]
      }
    }
    """
    _require_rectconv()
    return read_cam_from_json(str(path))


def camera_to_json(
    cam: "Camera",
    output_path: str | Path,
    note: str = "",
) -> None:
    """Serialise a Camera object to the RectConv JSON format."""
    _require_rectconv()
    from scipy.spatial.transform import Rotation as SciRot

    k = list(cam.lens.coefficients)
    while len(k) < 4:
        k.append(0.0)

    data = {
        "_note": note,
        "intrinsic": {
            "k1": float(k[0]), "k2": float(k[1]),
            "k3": float(k[2]), "k4": float(k[3]),
            "width":       int(cam.width),
            "height":      int(cam.height),
            "cx_offset":   float(cam.cx_offset),
            "cy_offset":   float(cam.cy_offset),
            "aspect_ratio": float(cam.aspect_ratio),
        },
        "extrinsic": {
            "quaternion":  SciRot.from_matrix(cam.rotation).as_quat().tolist(),
            "translation": cam.translation.tolist(),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[rectconv] Camera JSON written → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Distortion map
# ──────────────────────────────────────────────────────────────────────────────

def _camera_hash(cam: "Camera", max_kernel_size: int, interp_step: int, scaling: float) -> str:
    """Stable hash of camera parameters for cache key."""
    _require_rectconv()
    data = json.dumps({
        "k":        [float(v) for v in cam.lens.coefficients],
        "size":     [int(cam.width), int(cam.height)],
        "cx":       float(cam.cx_offset),
        "cy":       float(cam.cy_offset),
        "ar":       float(cam.aspect_ratio),
        "rot":      [[float(v) for v in row] for row in cam.rotation],
        "t":        [float(v) for v in cam.translation],
        "ks":       int(max_kernel_size),
        "step":     int(interp_step),
        "scaling":  float(scaling),
    }, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()[:12]


def build_distortion_map(
    cam: "Camera",
    cache_path: Optional[str | Path] = None,
    max_kernel_size: int = 7,
    interp_step: int = 32,
    scaling: float = 0.9,
) -> torch.Tensor:
    """
    Compute (or load from cache) the RectConv distortion offset map.

    The map has shape  (max_kernel_size, max_kernel_size, 2, H//interp_step, W//interp_step)
    and is computed once per camera configuration.

    Parameters
    ----------
    cam             : Camera object (from make_camera_from_fov or make_camera_from_json).
    cache_path      : If given, the map is saved here on first compute and loaded
                      on subsequent calls.  Pass None to disable caching.
    max_kernel_size : Maximum conv kernel size in the target model (default 7).
    interp_step     : Spatial interpolation stride — lower = more precise but slower
                      and more memory (default 32).
    scaling         : Controls the size of each locally-rectified patch (default 0.9).

    Returns
    -------
    torch.Tensor  distortion offset map.
    """
    _require_rectconv()

    h_key = _camera_hash(cam, max_kernel_size, interp_step, scaling)

    # ── Try loading from cache ─────────────────────────────────────────────
    if cache_path is not None:
        cp = Path(cache_path)
        if not cp.suffix:                          # auto-name if dir given
            cp = cp / f"distmap_{h_key}.pt"
        if cp.exists():
            distmap = torch.load(cp, map_location="cpu", weights_only=False)
            print(f"[rectconv] Loaded distortion map from cache: {cp}")
            return distmap

    # ── Compute ───────────────────────────────────────────────────────────
    print(f"[rectconv] Computing distortion map  "
          f"(img {cam.width}×{cam.height}, step={interp_step}, ks={max_kernel_size})…")
    t0 = time.time()
    distmap = generate_offset(cam, max_kernel_size=max_kernel_size,
                              interp_step=interp_step, scaling=scaling)
    elapsed = time.time() - t0
    print(f"[rectconv] Done in {elapsed:.1f}s  shape={tuple(distmap.shape)}")

    # ── Save cache ────────────────────────────────────────────────────────
    if cache_path is not None:
        cp = Path(cache_path)
        if not cp.suffix:
            cp = cp / f"distmap_{h_key}.pt"
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(distmap, cp)
        print(f"[rectconv] Saved distortion map → {cp}")

    return distmap


# ──────────────────────────────────────────────────────────────────────────────
# Model patching
# ──────────────────────────────────────────────────────────────────────────────

def patch_model(nn_module: "torch.nn.Module", distmap: torch.Tensor) -> None:
    """
    Replace every nn.Conv2d (kernel_size > 1) in nn_module with a
    RectifyConv2d that samples locally-rectified patches.

    This modifies nn_module IN-PLACE.  Model weights are copied verbatim —
    no retraining is required.  After patching the model processes fisheye
    images natively and its outputs are in fisheye pixel coordinates.

    Parameters
    ----------
    nn_module : Any torch.nn.Module (YOLO backbone, RT-DETR, etc.).
    distmap   : Distortion map tensor from build_distortion_map().
    """
    _require_rectconv()
    convert_to_rectconv(nn_module, distmap)
    print("[rectconv] Model patched with RectConv layers.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI — generate camera JSON from FOV
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Generate a RectConv camera calibration JSON from image FOV."
    )
    p.add_argument("--gen-json",  action="store_true",
                   help="Generate and save a camera JSON (required flag).")
    p.add_argument("--fov",       type=float, default=180.0,
                   help="Full fisheye FOV in degrees  (default: 180)")
    p.add_argument("--width",     type=int, default=640,
                   help="Image width in pixels  (default: 640)")
    p.add_argument("--height",    type=int, default=640,
                   help="Image height in pixels  (default: 640)")
    p.add_argument("--cx-offset", type=float, default=0.0,
                   help="Horizontal principal-point offset  (default: 0)")
    p.add_argument("--cy-offset", type=float, default=0.0,
                   help="Vertical principal-point offset  (default: 0)")
    p.add_argument("--output",    default="cameras/generated_cam.json",
                   help="Output JSON path  (default: cameras/generated_cam.json)")
    args = p.parse_args()

    if not args.gen_json:
        p.print_help()
        return

    cam = make_camera_from_fov(
        w=args.width, h=args.height,
        fov_deg=args.fov,
        cx_offset=args.cx_offset,
        cy_offset=args.cy_offset,
    )
    note = (f"Auto-generated equidistant camera: FOV={args.fov}°, "
            f"{args.width}×{args.height}px.  "
            "Replace k1-k4 with calibrated values for better accuracy.")
    camera_to_json(cam, args.output, note=note)
    print(f"  k1 = {cam.lens.coefficients[0]:.4f}  (equidistant, k2=k3=k4=0)")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    _cli()
