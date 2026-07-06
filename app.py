import streamlit as st
import os
import cv2
import numpy as np
import io
import pandas as pd
from PIL import Image
from datetime import datetime
import shutil

# Import utilities
from ultralytics import YOLO
from utils.detection_utils import load_model, run_detection, draw_detections, get_detection_summary

# PAGE CONFIG
st.set_page_config(
    page_title="Road Pothole Detection",
    page_icon="🕳️",
    layout="wide"
)

# COMPLETE CSS (custom styles for slate/dark-mode theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0F172A !important; color: #F1F5F9 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] {
    background: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stButton > button {
    background: linear-gradient(135deg, #EF4444, #DC2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(239,68,68,0.4) !important;
    color: white !important;
}
[data-testid="stFileUploader"] {
    background: #1E293B !important;
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
}
.stSlider > div > div > div > div { background: #EF4444 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# HEADER SECTION
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-left: 5px solid #EF4444;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
">
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
        <span style="font-size:36px;">🕳️</span>
        <h1 style="color:#F1F5F9; font-size:28px; font-weight:800; margin:0;">
            Road Pothole Detection System
        </h1>
    </div>
    <p style="color:#94A3B8; margin:0; font-size:14px; padding-left:52px;">
        AI-powered pothole detection using fine-tuned YOLOv8 model
    </p>
</div>
""", unsafe_allow_html=True)

# MODEL LOADING WITH CACHE
@st.cache_resource
def load_detection_model():
    """Cached loader for the YOLO weights."""
    return load_model('models/best.pt')

# SIDEBAR Setup & Threshold Controls
# Check if model exists locally or in parent cache directory and auto-copy it
local_model_path = 'models/best.pt'
parent_model_path = os.path.join("..", "Models", "best.pt")
if not os.path.exists(local_model_path) and os.path.exists(parent_model_path):
    os.makedirs('models', exist_ok=True)
    try:
        shutil.copy2(parent_model_path, local_model_path)
    except Exception as e:
        pass

model_exists = os.path.exists(local_model_path)

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0; border-bottom:1px solid #334155; margin-bottom:20px;">
        <p style="color:#F1F5F9; font-weight:800; font-size:18px; margin:0;">🕳️ PotholeAI</p>
        <p style="color:#64748B; font-size:11px; margin:4px 0 0 0;">Smart Road Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Detection Settings")

    high_recall = st.toggle(
        "High Recall Optimization",
        value=True,
        help="Optimizes settings to maximize recall and detect small, distant, and partially visible potholes."
    )

    if high_recall:
        st.info("ℹ️ **High Recall Mode Active**:\n- Conf Threshold: `0.15`\n- IOU Threshold: `0.45`\n- Resolution: `1024px`\n- Contrast Enhancer: `Enabled`")
        conf_val = 0.15
        iou_val = 0.45
        imgsz_val = 1024
        enhance_contrast_val = True
    else:
        conf_val = st.slider(
            "Confidence Threshold",
            min_value=0.10, max_value=0.90,
            value=0.25, step=0.05,
            help="Minimum confidence for detection"
        )

        iou_val = st.slider(
            "IOU Threshold",
            min_value=0.10, max_value=0.90,
            value=0.45, step=0.05,
            help="IOU threshold for NMS"
        )
        
        imgsz_val = st.selectbox(
            "Inference Resolution",
            options=[640, 800, 1024],
            index=1,
            help="Higher resolution improves detection of small/distant potholes but uses more memory/CPU."
        )
        
        enhance_contrast_val = st.checkbox(
            "Road Contrast Enhancement",
            value=True,
            help="Apply adaptive contrast correction (CLAHE) to make pothole edges more distinct."
        )

    # Model status display
    if model_exists:
        try:
            # Check model classes in session state
            if 'model_classes' not in st.session_state:
                test_model = YOLO(local_model_path)
                st.session_state.model_classes = list(test_model.names.values())
            
            classes = st.session_state.model_classes
            if any(cls in ['free', 'occupied', 'partially_occupied'] for cls in classes):
                st.markdown("""
                <div style="background:rgba(245,158,11,0.1); border:1px solid #F59E0B;
                    border-radius:8px; padding:10px; margin-top:16px; margin-bottom:12px;">
                    <p style="color:#F59E0B; margin:0; font-size:13px; font-weight:600;">
                        ⚠️ Parking Model Active
                    </p>
                    <p style="color:#94A3B8; margin:4px 0 0 0; font-size:11px;">
                        The current weights are configured for parking spaces. You can train the model or click download below to load pre-trained pothole weights.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(16,185,129,0.1); border:1px solid #10B981;
                    border-radius:8px; padding:10px; margin-top:16px; margin-bottom:12px;">
                    <p style="color:#10B981; margin:0; font-size:13px; font-weight:600;">
                        ✅ Pothole Model Loaded
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.1); border:1px solid #10B981;
                border-radius:8px; padding:10px; margin-top:16px; margin-bottom:12px;">
                <p style="color:#10B981; margin:0; font-size:13px; font-weight:600;">
                    ✅ Model Loaded
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.1); border:1px solid #EF4444;
            border-radius:8px; padding:10px; margin-top:16px; margin-bottom:12px;">
            <p style="color:#EF4444; margin:0; font-size:13px; font-weight:600;">
                ❌ Model Not Found
            </p>
            <p style="color:#94A3B8; margin:4px 0 0 0; font-size:11px;">
                Run: python train.py
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Download Button for pre-trained weights
    if st.button("📥 Load Pre-trained Pothole Weights", help="Click to download ready-to-use fine-tuned YOLOv8 pothole weights from HuggingFace."):
        with st.spinner("Downloading weights (6MB)... Please wait."):
            try:
                import urllib.request
                url = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
                os.makedirs("models", exist_ok=True)
                urllib.request.urlretrieve(url, "models/best.pt")
                # Clear session state cache for model classes
                if 'model_classes' in st.session_state:
                    del st.session_state['model_classes']
                st.success("✅ Weights downloaded! Reloading...")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")

    st.markdown("---")
    st.markdown("""
    <div style="color:#64748B; font-size:12px;">
        <p><b style="color:#94A3B8;">How to use:</b></p>
        <p>1. Upload a road image</p>
        <p>2. Click Detect Potholes</p>
        <p>3. View results and analysis</p>
    </div>
    """, unsafe_allow_html=True)

# MAIN CONTENT AREA

# If model doesn't exist, show error and stop
if not model_exists:
    st.error("""
❌ Model not found at models/best.pt

Please follow these steps:
1. Download dataset from Kaggle
2. Run: python prepare_dataset.py
3. Run: python train.py
4. Then restart this app
""")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a road image to detect potholes"
)

if uploaded_file is not None:
    try:
        # Read uploaded file as Image
        image_pil = Image.open(uploaded_file)
        
        # Convert to numpy array (RGB)
        image_np_rgb = np.array(image_pil)
        
        # Convert RGB to BGR for OpenCV functions
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        
        # Render a layout with Centered Detection button
        col_btn_left, col_btn_center, col_btn_right = st.columns([2, 1, 2])
        with col_btn_center:
            detect_clicked = st.button("🔍 Detect Potholes", width='stretch')
            
        if detect_clicked:
            with st.spinner("🔍 Analyzing road image for potholes..."):
                # Load the model
                model = load_detection_model()
                
                # Run detection
                detections = run_detection(model, image_np_bgr, conf_val, iou_val, imgsz_val, enhance_contrast_val)
                
                # Draw detections on image copy (returns BGR)
                annotated_image = draw_detections(image_np_bgr, detections)
                
                # Get summary stats
                summary = get_detection_summary(detections, image_np_bgr.shape)
                
                # Split-screen columns for image comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_np_rgb, caption="📷 Original Image", width='stretch')
                with col2:
                    # Convert BGR back to RGB for streamlit rendering
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, caption="🎯 Detection Results", width='stretch')
                
                st.markdown("---")
                
                # Summary cards (4 Columns)
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("🕳️ Potholes Detected", f"{summary['total_potholes']}")
                with sc2:
                    st.metric("📊 Avg Confidence", f"{summary['avg_confidence']:.1f}%")
                with sc3:
                    st.metric("⚠️ Severity", f"{summary['severity']}")
                with sc4:
                    st.metric("🗺️ Area Coverage", f"{summary['coverage_percent']:.2f}%")
                
                # Severity alert display
                if summary['severity'] == 'High':
                    st.error("🚨 HIGH SEVERITY — Immediate road repair required!")
                elif summary['severity'] == 'Medium':
                    st.warning("⚠️ MEDIUM SEVERITY — Road maintenance recommended")
                elif summary['severity'] == 'Low':
                    st.info("ℹ️ LOW SEVERITY — Monitor road condition")
                else:
                    st.success("✅ NO POTHOLES DETECTED — Road condition good")
                
                # Detailed analysis section
                if summary['total_potholes'] > 0:
                    st.markdown("### 📊 Detection Details")
                    
                    df_data = []
                    for idx, det in enumerate(detections, 1):
                        x1, y1, x2, y2 = det['bbox']
                        width_px = int(x2 - x1)
                        height_px = int(y2 - y1)
                        area_px = int(width_px * height_px)
                        
                        df_data.append({
                            "Pothole #": idx,
                            "Confidence %": f"{det['confidence']*100:.1f}%",
                            "Width px": width_px,
                            "Height px": height_px,
                            "Area px²": area_px
                        })
                        
                    df = pd.DataFrame(df_data)
                    
                    # Custom CSS styling applied internally
                    st.dataframe(
                        df.set_index("Pothole #"), 
                        width='stretch'
                    )
                    
                    # Download result action
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    buf = io.BytesIO()
                    annotated_pil.save(buf, format="JPEG", quality=95)
                    
                    st.download_button(
                        label="⬇️ Download Result Image",
                        data=buf.getvalue(),
                        file_name=f"pothole_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                        mime="image/jpeg",
                        width='content'
                    )
                else:
                    st.success("✅ No potholes detected in this image!")
                    st.info("Try lowering the confidence threshold if you think potholes exist.")
                    
    except Exception as ex:
        st.error(f"[-] An unexpected error occurred while processing the image: {ex}")
