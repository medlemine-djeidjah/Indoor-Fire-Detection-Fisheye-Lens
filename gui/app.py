import sys
import os
# Allow importing from scripts/ when running from the project root or gui/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile

try:
    from fisheye_rectifier import FisheyeRectifier
    _FISHEYE_AVAILABLE = True
except ImportError:
    _FISHEYE_AVAILABLE = False

try:
    from rectconv_adapter import (
        make_camera_from_fov,
        make_camera_from_json,
        build_distortion_map,
        patch_model,
    )
    _RECTCONV_AVAILABLE = True
except Exception:
    _RECTCONV_AVAILABLE = False

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FireGuard AI — Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS — Dark theme with fire-orange accents
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  [data-testid="stAppViewContainer"] { background: #ffffff; }
  [data-testid="stSidebar"]          { background: #fafafa; border-right: 1px solid #e5e7eb; }
  .block-container                   { padding-top: 1.5rem; }

  /* ── Header banner ── */
  .app-header {
    background: linear-gradient(135deg, #fff4ee 0%, #ffe8d6 50%, #fff4ee 100%);
    border: 1px solid #f97316aa;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .app-header h1 { margin: 0; font-size: 1.9rem; color: #c74b00; }
  .app-header p  { margin: 4px 0 0; color: #6b7280; font-size: 0.9rem; }

  /* ── Status badge ── */
  .badge-ready    { display:inline-block; padding:4px 12px; border-radius:999px;
                    background:#f0fdf4; color:#16a34a; font-size:.8rem; font-weight:700;
                    border:1px solid #86efac; }
  .badge-notready { display:inline-block; padding:4px 12px; border-radius:999px;
                    background:#fef2f2; color:#dc2626; font-size:.8rem; font-weight:700;
                    border:1px solid #fca5a5; }

  /* ── Metric cards ── */
  .metric-row  { display:flex; gap:12px; margin-top:.8rem; }
  .metric-card {
    flex:1; background:#fff7ed; border:1px solid #fed7aa;
    border-radius:10px; padding:14px 18px;
  }
  .metric-card .label { font-size:.75rem; color:#9ca3af; text-transform:uppercase; letter-spacing:.05em; }
  .metric-card .value { font-size:1.6rem; font-weight:700; color:#c74b00; margin-top:2px; }
  .metric-card .unit  { font-size:.75rem; color:#9ca3af; }

  /* ── Detection alert ── */
  .alert-danger {
    background:#fef2f2; border:1px solid #fca5a5; border-left:4px solid #dc2626;
    border-radius:8px; padding:12px 16px; color:#b91c1c;
  }
  .alert-safe {
    background:#f0fdf4; border:1px solid #86efac; border-left:4px solid #16a34a;
    border-radius:8px; padding:12px 16px; color:#15803d;
  }
  .alert-warning {
    background:#fffbeb; border:1px solid #fcd34d; border-left:4px solid #d97706;
    border-radius:8px; padding:12px 16px; color:#b45309;
  }

  /* ── Buttons ── */
  .stButton > button {
    width:100%; background:linear-gradient(135deg,#c74b00,#f97316);
    color:white; font-weight:700; border:none; border-radius:8px;
    padding:.55rem 1rem; transition:.2s;
  }
  .stButton > button:hover { filter:brightness(1.08); }

  /* ── Sidebar headings ── */
  .sidebar-section {
    font-size:.7rem; font-weight:700; color:#c74b00;
    text-transform:uppercase; letter-spacing:.1em;
    margin:1.2rem 0 .4rem;
  }

  /* ── Tab bar ── */
  [data-testid="stTabs"] [role="tab"] { font-weight:600; }
  [data-testid="stTabs"] [aria-selected="true"] { color:#c74b00 !important; }

  /* ── Progress bar ── */
  [data-testid="stProgressBar"] > div > div { background-color:#f97316; }

  /* ── Image captions ── */
  [data-testid="caption"] { color:#9ca3af; font-size:.78rem; }

  /* ── Expander ── */
  [data-testid="stExpander"] { background:#fafafa; border-radius:8px; border:1px solid #e5e7eb; }

  /* ── Misc ── */
  hr { border-color:#e5e7eb; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loader — supports:
#   YOLO Ultralytics  .pt / .onnx / .engine / .tflite
#   YOLO checkpoint   .pt  with {'model': ...} dict (standard training output)
#   DINOv3-LTDETR     .pt  state-dict (via lightly_train)
#   PyTorch full obj  .pt / .pth  (torch.nn.Module saved whole)
#   Generic ONNX      .onnx  (onnxruntime, non-YOLO)
#   Keras / TF        .h5 / .keras
# ─────────────────────────────────────────────
KERAS_EXTENSIONS = {".h5", ".keras"}

# LTDETR model IDs available in lightly_train
LTDETR_MODELS = [
    "dinov3/vitt16-ltdetr-coco",
    "dinov3/vits16-ltdetr-coco",
    "dinov3/convnext-tiny-ltdetr-coco",
]


class ModelWrapper:
    """Thin wrapper that normalises inference across model backends."""
    def __init__(self, model, backend: str, class_names: dict | None = None):
        self.model       = model
        self.backend     = backend   # 'yolo' | 'ltdetr' | 'onnx' | 'torch' | 'keras'
        self.class_names = class_names or {0: "fire", 1: "smoke"}

    def predict(self, image_bgr, conf: float = 0.45, iou: float = 0.7):
        """Returns (annotated_bgr, detections_list, inf_ms)."""
        t0        = time.time()
        annotated = image_bgr.copy()
        detections = []

        if self.backend == "yolo":
            results = self.model.predict(image_bgr, conf=conf, iou=iou, verbose=False)
            annotated = results[0].plot()
            for box in results[0].boxes:
                label = results[0].names[int(box.cls[0])]
                detections.append({
                    "label":      label,
                    "confidence": float(box.conf[0]),
                    "xyxy":       box.xyxy[0].tolist(),   # [x1, y1, x2, y2]
                })

        elif self.backend == "ltdetr":
            import torch
            h, w = image_bgr.shape[:2]
            img_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            tensor   = (
                torch.from_numpy(img_resized.astype("float32") / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            orig_sz  = torch.tensor([[h, w]], dtype=torch.long)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(tensor, orig_sz)
            # outputs: (labels, boxes, scores)  each shape [1, N]
            labels_out, boxes_out, scores_out = outputs
            for lbl, box, score in zip(
                labels_out[0].tolist(),
                boxes_out[0].tolist(),
                scores_out[0].tolist(),
            ):
                if score >= conf:
                    name = self.class_names.get(int(lbl), f"class_{int(lbl)}")
                    detections.append({"label": name, "confidence": float(score)})
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        annotated, f"{name} {score:.2f}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 165, 255), 2,
                    )

        elif self.backend == "onnx":
            h, w  = image_bgr.shape[:2]
            blob  = cv2.dnn.blobFromImage(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                1 / 255.0, (640, 640), swapRB=False, crop=False,
            )
            self.model.set_providers(["CPUExecutionProvider"])
            inp  = self.model.get_inputs()[0].name
            outs = self.model.run(None, {inp: blob})
            preds = np.squeeze(outs[0])
            if preds.ndim == 1:
                preds = preds[np.newaxis, :]
            for pred in preds:
                if len(pred) >= 6:
                    scores = pred[4:]
                    cls_id = int(np.argmax(scores))
                    score  = float(scores[cls_id])
                    if score >= conf:
                        label = self.class_names.get(cls_id, f"class_{cls_id}")
                        detections.append({"label": label, "confidence": score})
                        x1,y1,x2,y2 = (int(pred[0]*w/640), int(pred[1]*h/640),
                                        int(pred[2]*w/640), int(pred[3]*h/640))
                        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,165,255), 2)
                        cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0,165,255), 2)

        elif self.backend == "torch":
            import torch
            import torchvision.transforms as T
            device = "cuda" if torch.cuda.is_available() else "cpu"
            transform = T.Compose([
                T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([.485,.456,.406],[.229,.224,.225]),
            ])
            img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            tensor  = transform(img_pil).unsqueeze(0).to(device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(tensor)
            probs  = torch.softmax(out, dim=1)[0]
            cls_id = int(torch.argmax(probs))
            score  = float(probs[cls_id])
            if score >= conf:
                label = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append({"label": label, "confidence": score})
                cv2.putText(annotated, f"{label} {score:.2f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,165,255), 3)

        elif self.backend == "keras":
            img_rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            tensor      = np.expand_dims(img_resized.astype("float32") / 255.0, 0)
            preds       = self.model.predict(tensor, verbose=0)[0]
            cls_id      = int(np.argmax(preds))
            score       = float(preds[cls_id])
            if score >= conf:
                label = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append({"label": label, "confidence": score})
                cv2.putText(annotated, f"{label} {score:.2f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,165,255), 3)

        return annotated, detections, (time.time() - t0) * 1000


# ─────────────────────────────────────────────
# .pt inspection helpers (run outside cache)
# ─────────────────────────────────────────────
def _inspect_pt(path: str) -> str:
    """
    Return a string describing what kind of object is inside a .pt file:
    'yolo_full' | 'yolo_ckpt' | 'state_dict' | 'nn_module' | 'unknown'
    """
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return "unknown"

    if isinstance(obj, dict):
        # Standard YOLO training checkpoint: {'model': model_obj, 'epoch':...}
        if "model" in obj and hasattr(obj["model"], "predict"):
            return "yolo_ckpt"
        # Pure state dict: all values are tensors
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return "state_dict"
        # Some other dict
        return "yolo_ckpt"   # let YOLO try anyway

    if hasattr(obj, "predict"):
        return "yolo_full"

    import torch.nn as nn
    if isinstance(obj, nn.Module):
        return "nn_module"

    return "unknown"


# ─────────────────────────────────────────────
# Cached model loader
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(
    file_bytes: bytes,
    filename: str,
    ltdetr_arch: str = "",
) -> ModelWrapper | None:
    """
    ltdetr_arch: lightly_train model ID (e.g. 'dinov3/vits16-ltdetr-coco').
                 Required when uploading a DINOv3-LTDETR state-dict .pt file.
    """
    ext = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── ONNX ──────────────────────────────────────────────────────────
        if ext == ".onnx":
            # Try Ultralytics YOLO first (handles YOLO-exported ONNX natively)
            try:
                from ultralytics import YOLO
                m = YOLO(tmp_path)
                return ModelWrapper(m, "yolo")
            except Exception:
                pass
            # Fall back to raw onnxruntime session
            import onnxruntime as ort
            m = ort.InferenceSession(tmp_path)
            return ModelWrapper(m, "onnx")

        # ── TensorRT / TFLite / Saved model ───────────────────────────────
        if ext in {".engine", ".tflite", ".pb"}:
            from ultralytics import YOLO
            m = YOLO(tmp_path)
            return ModelWrapper(m, "yolo")

        # ── Keras ─────────────────────────────────────────────────────────
        if ext in KERAS_EXTENSIONS:
            import tensorflow as tf
            m = tf.keras.models.load_model(tmp_path)
            return ModelWrapper(m, "keras")

        # ── PyTorch checkpoint / weights (.pt or .pth) ────────────────────
        if ext in {".pt", ".pth"}:
            import torch

            # 1) Try Ultralytics YOLO (handles standard .pt and YOLO ckpts)
            try:
                from ultralytics import YOLO
                m = YOLO(tmp_path)
                return ModelWrapper(m, "yolo")
            except Exception as yolo_err:
                yolo_err_str = str(yolo_err)

            # 2) Inspect what's actually in the file
            kind = _inspect_pt(tmp_path)

            if kind == "yolo_ckpt":
                # Extract the model object from the checkpoint dict
                try:
                    obj = torch.load(tmp_path, map_location="cpu", weights_only=False)
                    m   = obj["model"]
                    m.eval()
                    # Wrap in a minimal YOLO-compatible shim so we get .predict()
                    from ultralytics import YOLO as _YOLO
                    import tempfile as _tf
                    _tmp = _tf.NamedTemporaryFile(suffix=".pt", delete=False)
                    torch.save({"model": m, "epoch": 0}, _tmp.name)
                    _tmp.close()
                    wrapped = _YOLO(_tmp.name)
                    os.unlink(_tmp.name)
                    return ModelWrapper(wrapped, "yolo")
                except Exception:
                    pass

            if kind == "nn_module":
                obj = torch.load(tmp_path, map_location="cpu", weights_only=False)
                obj.eval()
                return ModelWrapper(obj, "torch")

            if kind == "state_dict":
                # ── DINOv3-LTDETR state dict ──────────────────────────────
                if ltdetr_arch:
                    try:
                        import lightly_train
                        state_dict = torch.load(tmp_path, map_location="cpu",
                                                weights_only=True)
                        m = lightly_train.load_model(ltdetr_arch)
                        m.load_state_dict(state_dict)
                        m.eval()
                        return ModelWrapper(m, "ltdetr")
                    except Exception as e:
                        st.error(f"DINOv3-LTDETR load failed: {e}")
                        return None
                else:
                    st.error(
                        "**State-dict detected** — this file contains weights only, "
                        "not a full model. Select the **DINOv3-LTDETR architecture** "
                        "in the sidebar, then upload again."
                    )
                    return None

            # Fallback: surface the original YOLO error
            st.error(
                f"Could not load `{filename}` automatically.\n\n"
                f"YOLO error: `{yolo_err_str}`\n\n"
                "Try: export the model to ONNX, or select a model architecture hint "
                "in the sidebar."
            )
            return None

    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    st.error(f"Unsupported format: `{ext}`")
    return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;margin-bottom:8px'>
      <span style='font-size:2.8rem'>🔥</span><br>
      <span style='color:#ff8c42;font-weight:700;font-size:1.1rem'>FireGuard AI</span><br>
      <span style='color:#6b7280;font-size:.78rem'>Detection System v5.0 · RT-DETR + RectConv</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    uploaded_model = st.file_uploader(
        "Upload model weights",
        type=["pt", "pth", "onnx", "engine", "tflite", "pb", "h5", "keras"],
        help="Supports YOLO (.pt), ONNX (.onnx), PyTorch (.pth), Keras (.h5/.keras), TFLite (.tflite)",
    )

    st.markdown('<div class="sidebar-section">Architecture hint (state-dict .pt only)</div>',
                unsafe_allow_html=True)
    ltdetr_arch = st.selectbox(
        "DINOv3-LTDETR model ID",
        options=["— auto-detect —"] + LTDETR_MODELS,
        index=0,
        help=(
            "Only needed when uploading a **state-dict** .pt produced by "
            "`train_dinov3.py --mode ltdetr`. Leave on auto-detect for standard "
            "YOLO / distill-mode weights."
        ),
    )
    ltdetr_arch_val = "" if ltdetr_arch.startswith("—") else ltdetr_arch

    st.markdown('<div class="sidebar-section">Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.45, 0.05,
                               help="Minimum score to display a detection")
    iou_threshold  = st.slider("IOU / NMS", 0.0, 1.0, 0.70, 0.05,
                               help="Non-max suppression overlap threshold (YOLO only)")

    # ── Fisheye rectification ────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Fisheye Rectification</div>',
                unsafe_allow_html=True)
    if not _FISHEYE_AVAILABLE:
        st.warning("fisheye_rectifier.py not found in scripts/. Fisheye mode disabled.")
        fisheye_enabled = False
        rectifier = None
    else:
        fisheye_enabled = st.toggle(
            "Enable fisheye mode",
            value=False,
            help=(
                "Rectify the fisheye image before detection, then project "
                "bounding boxes back onto the original fisheye frame."
            ),
        )
        if fisheye_enabled:
            lens_type   = st.selectbox(
                "Lens projection model",
                ["equidistant", "equisolid", "orthographic", "stereographic"],
                index=0,
                help=(
                    "equidistant: r=f·θ  (most IP security cameras)\n"
                    "equisolid: r=2f·sin(θ/2)  (photographic fisheye)\n"
                    "orthographic: r=f·sin(θ)\n"
                    "stereographic: r=2f·tan(θ/2)"
                ),
            )
            fisheye_fov = st.slider(
                "Fisheye FOV (°)", 120, 220, 180, 5,
                help="Full field-of-view of your fisheye lens.",
            )
            output_fov  = st.slider(
                "Output perspective FOV (°)", 60, 160, 120, 5,
                help="Field-of-view of the rectified output window.",
            )
            rectifier = FisheyeRectifier(
                fov=fisheye_fov, pfov=output_fov, dtype=lens_type
            )
        else:
            rectifier = None

    # ── RectConv ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">RectConv (fisheye-native)</div>',
                unsafe_allow_html=True)
    if not _RECTCONV_AVAILABLE:
        st.warning("rectconv_adapter unavailable. Check third_party/RectConv exists.")
        rectconv_enabled = False
        _rc_distmap      = None
    else:
        rectconv_enabled = st.toggle(
            "Enable RectConv mode",
            value=False,
            help=(
                "Patches the model's Conv2d layers to sample locally-rectified "
                "patches. Detects directly on the fisheye image — no global "
                "warping, no back-projection. Requires one-time offset computation."
            ),
        )
        if rectconv_enabled:
            if fisheye_enabled:
                st.warning("RectConv and Fisheye Rectification modes are mutually "
                           "exclusive. Disable one before enabling the other.")
                rectconv_enabled = False
                _rc_distmap      = None
            else:
                rc_cam_mode = st.radio(
                    "Camera specification",
                    ["From FOV (approximate)", "From calibration JSON"],
                    horizontal=True,
                )
                if rc_cam_mode == "From FOV (approximate)":
                    rc_fov    = st.slider("Fisheye FOV (°)", 120, 220, 180, 5)
                    rc_imgsz  = st.number_input("Image size (px)", 320, 1280, 640, 32)
                    rc_cam_key = f"fov{rc_fov}_sz{rc_imgsz}"
                    _rc_cam_fn = lambda: make_camera_from_fov(  # noqa: E731
                        w=int(rc_imgsz), h=int(rc_imgsz), fov_deg=rc_fov
                    )
                    rc_cam_desc = f"FOV {rc_fov}°, {int(rc_imgsz)}×{int(rc_imgsz)}px"
                else:
                    rc_json_file = st.file_uploader(
                        "Upload camera JSON", type=["json"], key="rc_json"
                    )
                    if rc_json_file:
                        import tempfile as _tf, json as _json
                        _rc_tmp = _tf.NamedTemporaryFile(suffix=".json", delete=False)
                        _rc_tmp.write(rc_json_file.read())
                        _rc_tmp.close()
                        rc_cam_key  = rc_json_file.name
                        _rc_cam_fn  = lambda: make_camera_from_json(_rc_tmp.name)  # noqa: E731
                        rc_cam_desc = rc_json_file.name
                    else:
                        st.info("Upload a camera JSON, or switch to FOV mode.")
                        rectconv_enabled = False
                        _rc_distmap      = None

                if rectconv_enabled:
                    @st.cache_resource(show_spinner="Computing RectConv offset map…")
                    def _get_distmap(cam_key: str):
                        _cam = _rc_cam_fn()
                        return build_distortion_map(_cam, cache_path=None)

                    with st.spinner(f"Loading offset map for {rc_cam_desc}…"):
                        _rc_distmap = _get_distmap(rc_cam_key)
                    st.success(f"Offset map ready  ·  {rc_cam_desc}")
        else:
            _rc_distmap = None

    if uploaded_model:
        st.markdown('<div class="sidebar-section">Class Names</div>', unsafe_allow_html=True)
        raw = st.text_area("Custom class names (one per line)",
                           value="fire\nsmoke",
                           height=90,
                           help="Override default class names. First line = class 0, second = class 1, …")
        custom_classes = {i: n.strip() for i, n in enumerate(raw.splitlines()) if n.strip()}
    else:
        custom_classes = {0: "fire", 1: "smoke"}

    st.markdown("---")
    st.caption("💡 Tip: Upload your trained `best.pt` from `runs/` for best accuracy.")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style='font-size:2.4rem'>🚨</span>
  <div>
    <h1>FireGuard AI — Detection System</h1>
    <p>Real-time fire &amp; smoke analytics · Multi-model · Image · Video · Live camera</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model: ModelWrapper | None = None
if uploaded_model:
    with st.spinner("Loading model…"):
        model = load_model(uploaded_model.getvalue(), uploaded_model.name, ltdetr_arch_val)
    if model:
        model.class_names = custom_classes
        ext = os.path.splitext(uploaded_model.name)[1].upper()
        st.markdown(
            f'<div style="margin-bottom:.8rem">'
            f'<span class="badge-ready">✔ MODEL READY</span> &nbsp;'
            f'<span style="color:#6b7280;font-size:.85rem">{uploaded_model.name} &nbsp;·&nbsp; '
            f'backend: <b style="color:#ff8c42">{model.backend.upper()}</b> &nbsp;·&nbsp; '
            f'format: <b style="color:#ff8c42">{ext}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge-notready">✘ LOAD FAILED</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-notready">✘ NO MODEL LOADED</span>', unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

# Track which model instances have already been patched with RectConv
_patched_models: set = set()


def apply_rectconv_to_model(mdl: "ModelWrapper", distmap) -> None:
    """
    Patch all Conv2d layers in mdl with RectifyConv2d (in-place, once).
    Subsequent calls on the same model instance are no-ops.
    """
    model_id = id(mdl.model)
    if model_id in _patched_models:
        return
    if mdl.backend != "yolo":
        st.warning("RectConv patching is only supported for YOLO/RT-DETR backends.")
        return
    nn_module = mdl.model.model   # Ultralytics YOLO wraps the nn.Module here
    patch_model(nn_module, distmap)
    _patched_models.add(model_id)


def run_fisheye_pipeline(
    mdl: "ModelWrapper",
    fisheye_bgr: np.ndarray,
    rect: "FisheyeRectifier",
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray, list, float]:
    """
    Full fisheye detection pipeline:
      1. Rectify the fisheye image geometrically.
      2. Run the detector on the perspective image.
      3. Back-project bounding boxes to the original fisheye frame.

    Returns
    -------
    annotated_rect    : rectified image with standard detection boxes drawn
    annotated_fisheye : original fisheye image with polygon boxes drawn
    detections        : list of detection dicts (label, confidence, xyxy)
    inf_ms            : inference time in milliseconds
    """
    rectified_bgr = rect.rectify(fisheye_bgr)
    annotated_rect, detections, inf_ms = mdl.predict(
        rectified_bgr, conf=conf, iou=iou
    )

    boxes       = [d["xyxy"] for d in detections if "xyxy" in d]
    labels      = [d["label"]      for d in detections if "xyxy" in d]
    confidences = [d["confidence"] for d in detections if "xyxy" in d]

    if boxes:
        annotated_fisheye = rect.draw_fisheye_boxes(
            fisheye_bgr, boxes, labels, confidences
        )
    else:
        annotated_fisheye = fisheye_bgr.copy()

    return annotated_rect, annotated_fisheye, detections, inf_ms


def render_metrics(n_det: int, inf_ms: float, detections: list):
    top = detections[0]["label"].upper() if detections else "—"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="label">Detections</div>
        <div class="value">{n_det}</div>
      </div>
      <div class="metric-card">
        <div class="label">Inference</div>
        <div class="value">{inf_ms:.0f}<span class="unit"> ms</span></div>
      </div>
      <div class="metric-card">
        <div class="label">Top result</div>
        <div class="value" style="font-size:1.2rem">{top}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def detection_alert(detections: list):
    if not detections:
        st.markdown('<div class="alert-safe">✅ <b>CLEAR</b> — no threats detected.</div>',
                    unsafe_allow_html=True)
    else:
        labels = ", ".join(f'{d["label"]} ({d["confidence"]:.0%})' for d in detections)
        st.markdown(f'<div class="alert-danger">⚠️ <b>THREAT DETECTED</b> — {labels}</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────
if model:
    tab_img, tab_vid, tab_cam = st.tabs(["🖼️  Image Analysis", "🎬  Video Analysis", "📷  Live Camera"])

    # ══════════════════════════════════════════
    # TAB 1 — Image Analysis
    # ══════════════════════════════════════════
    with tab_img:
        fisheye_label = " · fisheye mode ON" if fisheye_enabled else ""
        st.subheader(f"Static Image Detection{fisheye_label}")
        img_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="img_uploader",
        )

        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            col_src, col_res = st.columns(2)
            col_src.image(image_rgb, caption="Source image", use_container_width=True)

            if st.button("▶  RUN DETECTION", key="btn_img"):
                with st.spinner("Running inference…"):
                    if rectconv_enabled and _rc_distmap is not None:
                        # ── RectConv path ──────────────────────────────────
                        # Patch model once (no-op on subsequent calls)
                        apply_rectconv_to_model(model, _rc_distmap)
                        # Run directly on the fisheye image — detections are
                        # already in fisheye pixel coordinates
                        annotated_bgr, detections, inf_ms = model.predict(
                            image_bgr, conf=conf_threshold, iou=iou_threshold
                        )
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        col_res.image(annotated_rgb,
                                      caption="RectConv — fisheye-native detection",
                                      use_container_width=True)
                    elif fisheye_enabled and rectifier is not None:
                        # ── Global rectification + back-projection path ────
                        ann_rect_bgr, ann_fish_bgr, detections, inf_ms = run_fisheye_pipeline(
                            model, image_bgr, rectifier, conf_threshold, iou_threshold
                        )
                        col_res.image(
                            cv2.cvtColor(ann_rect_bgr, cv2.COLOR_BGR2RGB),
                            caption="Rectified + detections",
                            use_container_width=True,
                        )
                        st.image(
                            cv2.cvtColor(ann_fish_bgr, cv2.COLOR_BGR2RGB),
                            caption="Original fisheye + projected bounding polygons",
                            use_container_width=True,
                        )
                        annotated_bgr = ann_fish_bgr
                    else:
                        # ── Standard path ──────────────────────────────────
                        annotated_bgr, detections, inf_ms = model.predict(
                            image_bgr, conf=conf_threshold, iou=iou_threshold
                        )
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        col_res.image(annotated_rgb, caption="Detection result",
                                      use_container_width=True)

                st.markdown("---")
                render_metrics(len(detections), inf_ms, detections)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                detection_alert(detections)

                if detections:
                    with st.expander("Raw detection log"):
                        for i, d in enumerate(detections, 1):
                            st.write(f"#{i}: **{d['label']}** — confidence {d['confidence']:.4f}")

                # Download annotated image
                _, enc = cv2.imencode(".jpg", annotated_bgr)
                st.download_button(
                    "⬇  Download annotated image",
                    data=enc.tobytes(),
                    file_name="fireguard_result.jpg",
                    mime="image/jpeg",
                )

    # ══════════════════════════════════════════
    # TAB 2 — Video Analysis
    # ══════════════════════════════════════════
    with tab_vid:
        st.subheader("Video File Detection")
        st.markdown(
            '<div class="alert-warning">⚡ Large videos are processed frame-by-frame. '
            'Use a shorter clip for faster results.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        vid_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv", "wmv", "flv"],
            key="vid_uploader",
        )

        frame_skip = st.slider(
            "Process every Nth frame",
            min_value=1, max_value=30, value=3,
            help="Skip frames to speed up processing (1 = every frame, 5 = every 5th frame)",
        )
        save_video = st.checkbox("Save annotated video for download", value=True)

        if vid_file and st.button("▶  ANALYSE VIDEO", key="btn_vid"):
            # Write uploaded video to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
                tmp_in.write(vid_file.read())
                tmp_in_path = tmp_in.name

            cap = cv2.VideoCapture(tmp_in_path)
            total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps           = cap.get(cv2.CAP_PROP_FPS) or 25
            width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            processed_fps = max(1, fps / frame_skip)

            tmp_out_path = None
            writer       = None
            if save_video:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
                    tmp_out_path = tmp_out.name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp_out_path, fourcc, processed_fps, (width, height))

            # UI placeholders
            progress_bar  = st.progress(0, text="Initialising…")
            status_text   = st.empty()
            preview_slot  = st.empty()

            # Stats
            total_det_frames = 0
            total_dets       = 0
            all_inf_ms       = []
            detection_timeline = []   # list of (frame_idx, n_detections)

            frame_idx  = 0
            proc_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    if rectconv_enabled and _rc_distmap is not None:
                        apply_rectconv_to_model(model, _rc_distmap)
                        annotated, detections, inf_ms = model.predict(
                            frame, conf=conf_threshold, iou=iou_threshold
                        )
                    elif fisheye_enabled and rectifier is not None:
                        _, annotated, detections, inf_ms = run_fisheye_pipeline(
                            model, frame, rectifier, conf_threshold, iou_threshold
                        )
                    else:
                        annotated, detections, inf_ms = model.predict(
                            frame, conf=conf_threshold, iou=iou_threshold
                        )
                    all_inf_ms.append(inf_ms)
                    detection_timeline.append((frame_idx, len(detections)))
                    if detections:
                        total_det_frames += 1
                        total_dets       += len(detections)

                    if writer:
                        writer.write(annotated)

                    proc_count += 1
                    # Refresh preview every ~10 processed frames
                    if proc_count % 10 == 0 or frame_idx == 0:
                        prev_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        preview_slot.image(prev_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

                pct = min(frame_idx / max(total_frames - 1, 1), 1.0)
                progress_bar.progress(pct, text=f"Processing frame {frame_idx}/{total_frames}…")
                frame_idx += 1

            cap.release()
            if writer:
                writer.release()
            os.unlink(tmp_in_path)

            progress_bar.progress(1.0, text="Done!")
            status_text.empty()

            # ── Summary ──────────────────────────────────────────────────────
            st.markdown("---")
            avg_inf = np.mean(all_inf_ms) if all_inf_ms else 0
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card">
                <div class="label">Frames processed</div>
                <div class="value">{proc_count}</div>
              </div>
              <div class="metric-card">
                <div class="label">Frames with detections</div>
                <div class="value">{total_det_frames}</div>
              </div>
              <div class="metric-card">
                <div class="label">Total detections</div>
                <div class="value">{total_dets}</div>
              </div>
              <div class="metric-card">
                <div class="label">Avg inference</div>
                <div class="value">{avg_inf:.0f}<span class="unit"> ms</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            threat_pct = (total_det_frames / proc_count * 100) if proc_count else 0
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if total_dets > 0:
                st.markdown(
                    f'<div class="alert-danger">⚠️ <b>THREAT DETECTED</b> in '
                    f'{total_det_frames} / {proc_count} frames ({threat_pct:.1f}%).</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="alert-safe">✅ <b>CLEAR</b> — no threats in video.</div>',
                            unsafe_allow_html=True)

            # ── Detection timeline chart ──────────────────────────────────────
            if detection_timeline:
                import pandas as pd
                df = pd.DataFrame(detection_timeline, columns=["frame", "detections"])
                st.markdown("**Detection timeline**")
                st.bar_chart(df.set_index("frame")["detections"], color="#ff6b1a")

            # ── Download annotated video ──────────────────────────────────────
            if save_video and tmp_out_path and os.path.exists(tmp_out_path):
                with open(tmp_out_path, "rb") as f:
                    vid_bytes = f.read()
                st.download_button(
                    "⬇  Download annotated video",
                    data=vid_bytes,
                    file_name="fireguard_video.mp4",
                    mime="video/mp4",
                )
                os.unlink(tmp_out_path)

    # ══════════════════════════════════════════
    # TAB 3 — Live Camera
    # ══════════════════════════════════════════
    with tab_cam:
        st.subheader("Live Camera Stream")
        st.write("Enable camera access in your browser, then capture frames below.")

        cam_active = st.toggle("Enable camera")
        cam_input  = st.camera_input("Capture frame", disabled=not cam_active)

        if cam_input:
            bytes_data = cam_input.getvalue()
            cv2_img    = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if rectconv_enabled and _rc_distmap is not None:
                apply_rectconv_to_model(model, _rc_distmap)
                annotated_bgr, detections, inf_ms = model.predict(
                    cv2_img, conf=conf_threshold, iou=iou_threshold
                )
                st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                         caption="RectConv — fisheye-native detection",
                         use_container_width=True)
            elif fisheye_enabled and rectifier is not None:
                ann_rect_bgr, ann_fish_bgr, detections, inf_ms = run_fisheye_pipeline(
                    model, cv2_img, rectifier, conf_threshold, iou_threshold
                )
                col_r, col_f = st.columns(2)
                col_r.image(cv2.cvtColor(ann_rect_bgr, cv2.COLOR_BGR2RGB),
                            caption="Rectified + detections", use_container_width=True)
                col_f.image(cv2.cvtColor(ann_fish_bgr, cv2.COLOR_BGR2RGB),
                            caption="Fisheye + projected boxes", use_container_width=True)
            else:
                annotated_bgr, detections, inf_ms = model.predict(
                    cv2_img, conf=conf_threshold, iou=iou_threshold
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Live frame detection", use_container_width=True)

            render_metrics(len(detections), inf_ms, detections)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            detection_alert(detections)

# ─────────────────────────────────────────────
# No-model landing page
# ─────────────────────────────────────────────
else:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem'>
          <div style='font-size:5rem'>🔥</div>
          <h2 style='color:#c74b00;margin:.5rem 0'>FireGuard AI</h2>
          <p style='color:#6b7280;margin-bottom:2rem'>
            Upload a model in the sidebar to begin detection.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:20px 24px'>
          <div style='color:#c74b00;font-weight:700;margin-bottom:12px'>Supported model formats</div>
          <table style='width:100%;border-collapse:collapse;color:#374151;font-size:.88rem'>
            <tr style='border-bottom:1px solid #fde8cc'>
              <td style='padding:7px 4px'><b>YOLO (Ultralytics)</b></td>
              <td style='padding:7px 4px;color:#6b7280'>.pt &nbsp;·&nbsp; .onnx &nbsp;·&nbsp; .engine &nbsp;·&nbsp; .tflite</td>
            </tr>
            <tr style='border-bottom:1px solid #fde8cc'>
              <td style='padding:7px 4px'><b>PyTorch</b></td>
              <td style='padding:7px 4px;color:#6b7280'>.pt &nbsp;·&nbsp; .pth</td>
            </tr>
            <tr style='border-bottom:1px solid #fde8cc'>
              <td style='padding:7px 4px'><b>ONNX Runtime</b></td>
              <td style='padding:7px 4px;color:#6b7280'>.onnx</td>
            </tr>
            <tr>
              <td style='padding:7px 4px'><b>Keras / TensorFlow</b></td>
              <td style='padding:7px 4px;color:#6b7280'>.h5 &nbsp;·&nbsp; .keras &nbsp;·&nbsp; .tflite &nbsp;·&nbsp; .pb</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
