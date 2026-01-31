import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time

# --- Page Config ---
st.set_page_config(
    page_title="VisionAI - Fire & Smoke Detection",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
    .metric-container {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4259;
    }
    .status-active {
        color: #00ff00;
        font-weight: bold;
    }
    .status-inactive {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Control Panel ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🔥</h1>", unsafe_allow_html=True)
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("🛠️ Model Selection")
    # File uploader for the model
    uploaded_model = st.file_uploader("Upload YOLO Model weights (.pt)", type=["pt"])
    
    st.markdown("---")
    st.subheader("⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05, help="Minimum confidence to show a detection")
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.70, 0.05, help="NMS IOU threshold")
    
    st.markdown("---")
    st.info("💡 **Tip:** Upload your trained `best.pt` file from the `runs/` directory for specific fire/smoke detection.")

# --- Model Loading Logic ---
@st.cache_resource
def load_uploaded_model(model_file):
    if model_file is None:
        return None

    # Save the uploaded file to a temporary location for YOLO to load
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "fire_detection_model.pt")
    with open(temp_path, "wb") as f:
        f.write(model_file.getbuffer())
    
    try:
        model = YOLO(temp_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main Dashboard ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🚨 VisionAI Detection System")
    st.markdown("*Real-time fire and smoke analytics platform.*")

with col2:
    if uploaded_model:
        st.markdown(f'<div class="metric-container">Model Status: <span class="status-active">READY</span><br>File: {uploaded_model.name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-container">Model Status: <span class="status-inactive">NOT LOADED</span><br>Please upload a .pt file</div>', unsafe_allow_html=True)

st.markdown("---")

# Load model if uploaded
model = load_uploaded_model(uploaded_model)

if model:
    tab1, tab2 = st.tabs(["�️ Image Analysis", "🔭 Live Stream"])

    with tab1:
        st.subheader("Static Image Detection")
        img_file = st.file_uploader("Upload Image for Analysis", type=["jpg", "jpeg", "png"], key="img_uploader")
        
        if img_file:
            # Pre-processing
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(image_rgb, caption="Source Image", use_container_width=True)
            
            if st.button("RUN ANALYTICS"):
                start_time = time.time()
                with st.spinner('Processing...'):
                    results = model.predict(image_bgr, conf=conf_threshold, iou=iou_threshold)
                    inf_time = (time.time() - start_time) * 1000 # ms
                    
                    annotated_img = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                with c2:
                    st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
                
                # Metrics Display
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Detections Found", len(results[0].boxes))
                m2.metric("Inference Time", f"{inf_time:.1f} ms")
                
                if len(results[0].boxes) > 0:
                    dominant_cls = results[0].names[int(results[0].boxes[0].cls[0])]
                    m3.metric("Top Result", dominant_cls.upper())
                    
                    # Detailed Logs
                    with st.expander("Show Raw Detection Logs"):
                        for i, box in enumerate(results[0].boxes):
                            cls_name = results[0].names[int(box.cls[0])]
                            conf = box.conf[0]
                            st.write(f"#{i+1}: {cls_name} - {conf:.4f}")
                else:
                    st.success("No threats detected in the scanned area.")

    with tab2:
        st.subheader("Camera Integration")
        st.write("Ensure your browser allows camera access.")
        
        cam_active = st.toggle("Enable Live Scanner")
        cam_input = st.camera_input("Capture frame for analysis", disabled=not cam_active)
        
        if cam_input:
            bytes_data = cam_input.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            results = model.predict(cv2_img, conf=conf_threshold, iou=iou_threshold)
            
            annotated_img = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_rgb, caption='Live Scanner Preview', use_container_width=True)
            
            if len(results[0].boxes) > 0:
                st.error(f"⚠️ THREAT DETECTED: {len(results[0].boxes)} signals identified.")
            else:
                st.success("Safe: Monitoring active.")

else:
    st.warning("⚠️ **Waiting for Model weights.** Please upload a YOLO weights file (.pt) in the sidebar to begin.")
    st.image("https://images.unsplash.com/photo-1542353436-312f0ebbf670?q=80&w=1000&auto=format&fit=crop", width=600) # Nice placeholder image (Fire station/Abstract)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("VisionAI v2.0 - Advanced Incident Detection")
