"""
fisheye_rectifier.py
--------------------
Calibration-free fisheye-to-perspective rectification using the
Otto & Weinhaus geometric model (same math as the `defisheye` library,
but implemented directly so we retain the remap arrays for bounding-box
back-projection without any file I/O).

Supported lens projection models
---------------------------------
equidistant   r = f·θ               (most IP-security fisheye cameras)
equisolid     r = 2f·sin(θ/2)       (photographic fisheye lenses)
orthographic  r = f·sin(θ)          (ortho projection, narrow FOV)
stereographic r = 2f·tan(θ/2)       (conformal / Ricoh-style lenses)

Usage
------
    from scripts.fisheye_rectifier import FisheyeRectifier

    rect = FisheyeRectifier(fov=180, pfov=120, dtype="equidistant")
    rectified = rect.rectify(fisheye_bgr)          # perspective image

    # After running YOLO on rectified:
    polygons = rect.project_boxes_to_fisheye(boxes_xyxy)   # list of 4-pt polys

    annotated = rect.draw_fisheye_boxes(
        fisheye_bgr, boxes_xyxy, labels, confidences
    )
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette: fire=red, smoke=orange, other=blue
# ──────────────────────────────────────────────────────────────────────────────
_LABEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "fire":  (0,  30, 220),    # BGR – vivid red
    "smoke": (0, 140, 255),    # BGR – orange
}
_DEFAULT_COLOUR = (220, 60, 0)


class FisheyeRectifier:
    """
    Geometric fisheye rectifier.

    Parameters
    ----------
    fov   : int   Full fisheye field-of-view in degrees (default 180).
    pfov  : int   Desired perspective output field-of-view in degrees (default 120).
    dtype : str   Lens projection model: equidistant | equisolid |
                  orthographic | stereographic  (default equidistant).
    """

    def __init__(
        self,
        fov: int = 180,
        pfov: int = 120,
        dtype: str = "equidistant",
    ) -> None:
        self.fov   = fov
        self.pfov  = pfov
        self.dtype = dtype.lower()
        if self.dtype not in {"equidistant", "equisolid", "orthographic", "stereographic"}:
            raise ValueError(f"Unknown lens type '{dtype}'. "
                             "Choose equidistant / equisolid / orthographic / stereographic.")

        # Remap arrays – built lazily per image size
        self._xmap: np.ndarray | None = None
        self._ymap: np.ndarray | None = None
        self._last_shape: tuple[int, int] | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Map construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_maps(self, h: int, w: int) -> None:
        """
        Pre-compute backward remap arrays.

        xmap[y_rect, x_rect] → x_fisheye
        ymap[y_rect, x_rect] → y_fisheye

        Passing these to cv2.remap produces the rectified image,
        and indexing them at bounding-box corners back-projects into
        the original fisheye frame.
        """
        if self._last_shape == (h, w):
            return

        cx, cy   = w / 2.0, h / 2.0
        fov_rad  = np.deg2rad(self.fov)
        pfov_rad = np.deg2rad(self.pfov)
        max_r    = min(cx, cy)            # inscribed-circle radius

        # ── Perspective output focal length ──────────────────────────────
        f_out = cx / np.tan(pfov_rad / 2.0)

        # ── Grid of output (perspective) pixels ──────────────────────────
        xv, yv = np.meshgrid(
            np.arange(w, dtype=np.float64),
            np.arange(h, dtype=np.float64),
        )
        xn = (xv - cx) / f_out          # normalised perspective x
        yn = (yv - cy) / f_out          # normalised perspective y
        r_proj = np.sqrt(xn ** 2 + yn ** 2)   # perspective radius

        # ── Angle from optical axis ───────────────────────────────────────
        theta = np.arctan(r_proj)       # ∈ [0, π/2]

        # ── Fisheye projection: 2-D radius in input image ─────────────────
        dt = self.dtype
        if dt == "equidistant":
            f_in   = max_r / (fov_rad / 2.0)
            r_fish = f_in * theta
        elif dt == "equisolid":
            f_in   = max_r / (2.0 * np.sin(fov_rad / 4.0))
            r_fish = 2.0 * f_in * np.sin(theta / 2.0)
        elif dt == "orthographic":
            f_in   = max_r / np.sin(fov_rad / 2.0)
            r_fish = f_in * np.sin(theta)
        else:  # stereographic
            f_in   = max_r / (2.0 * np.tan(fov_rad / 4.0))
            r_fish = 2.0 * f_in * np.tan(theta / 2.0)

        # ── Back-project to fisheye pixel coords ─────────────────────────
        # Use safe_proj to avoid division-by-zero warnings; np.where evaluates
        # both branches before selecting, so we pre-clamp the denominator.
        safe_proj = np.where(r_proj > 1e-10, r_proj, 1.0)
        scale     = np.where(r_proj > 1e-10, r_fish / safe_proj, 0.0)
        self._xmap = (xn * scale + cx).astype(np.float32)
        self._ymap = (yn * scale + cy).astype(np.float32)
        self._last_shape = (h, w)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def rectify(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Warp a fisheye image into a perspective-corrected image of the
        same spatial resolution.

        Parameters
        ----------
        image_bgr : np.ndarray  BGR image (H × W × 3 uint8).

        Returns
        -------
        np.ndarray  Rectified BGR image.
        """
        h, w = image_bgr.shape[:2]
        self._build_maps(h, w)
        return cv2.remap(
            image_bgr,
            self._xmap,
            self._ymap,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def project_boxes_to_fisheye(
        self,
        boxes_xyxy: Sequence[Sequence[float]],
    ) -> list[list[tuple[int, int]]]:
        """
        Back-project rectangular bounding boxes from rectified space to
        fisheye space.

        Parameters
        ----------
        boxes_xyxy : list of [x1, y1, x2, y2] in rectified-image pixels.

        Returns
        -------
        List of 4-point polygon corners [ (x,y), ... ] in fisheye space,
        one polygon per input box, ordered TL → TR → BR → BL.
        """
        if self._xmap is None:
            raise RuntimeError("Call rectify() before project_boxes_to_fisheye().")

        h, w   = self._xmap.shape
        polys  = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = [float(v) for v in box]
            corners_rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            corners_fish = []
            for xr, yr in corners_rect:
                xi = int(np.clip(round(xr), 0, w - 1))
                yi = int(np.clip(round(yr), 0, h - 1))
                corners_fish.append(
                    (int(round(float(self._xmap[yi, xi]))),
                     int(round(float(self._ymap[yi, xi]))))
                )
            polys.append(corners_fish)
        return polys

    def draw_fisheye_boxes(
        self,
        fisheye_bgr: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        labels: Sequence[str],
        confidences: Sequence[float],
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw back-projected bounding polygons and labels on the original
        fisheye image.

        Parameters
        ----------
        fisheye_bgr   : original fisheye BGR image.
        boxes_xyxy    : list of [x1, y1, x2, y2] in rectified coordinates.
        labels        : class names (one per box).
        confidences   : confidence scores (one per box).
        thickness     : line thickness in pixels.

        Returns
        -------
        np.ndarray  Annotated fisheye image (copy of input + drawn polygons).
        """
        if self._xmap is None:
            raise RuntimeError("Call rectify() before draw_fisheye_boxes().")

        canvas = fisheye_bgr.copy()
        polys  = self.project_boxes_to_fisheye(boxes_xyxy)

        for polygon, label, conf in zip(polys, labels, confidences):
            colour = _LABEL_COLOURS.get(label, _DEFAULT_COLOUR)
            pts    = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=colour, thickness=thickness)

            # Label at the top-left corner of the polygon
            lx, ly = polygon[0]
            text   = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, thickness
            )
            ty = max(ly - 6, th + baseline)
            # Background rect for readability
            cv2.rectangle(
                canvas,
                (lx, ty - th - baseline),
                (lx + tw, ty + baseline),
                colour, -1,
            )
            cv2.putText(
                canvas, text, (lx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        return canvas
