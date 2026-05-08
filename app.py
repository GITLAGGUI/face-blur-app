import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64
import io

# --- SVG Icons (Inline for st.markdown) ---
SVG_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
SVG_SHIELD_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>'
SVG_ROCKET = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"></path><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"></path></svg>'
SVG_SPARKLES = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path><path d="M5 3v4"></path><path d="M19 17v4"></path><path d="M3 5h4"></path><path d="M17 19h4"></path></svg>'
SVG_COG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
SVG_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>'
SVG_IMAGE = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect><circle cx="9" cy="9" r="2"></circle><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path></svg>'
SVG_LOCK = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>'
SVG_EYE = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>'
SVG_USER = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
SVG_TARGET = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
SVG_CODE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>'
SVG_LOCK_SM = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>'

# --- Page Config ---
st.set_page_config(
    page_title="Neural Network Privacy System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Theme / CSS Overhaul ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Hide Streamlit Default Elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #f4f7f9 !important;
            color: #1e293b;
        }

        .stApp {
            background-color: #f4f7f9;
        }

        /* Hero Section */
        .hero-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            margin-bottom: 4rem;
            width: 100%;
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.3;
            margin-bottom: 1rem;
            text-align: center;
        }
        .hero-title span {
            color: #6366f1;
        }
        .hero-subtitle {
            color: #64748b;
            font-size: 1.1rem;
            max-width: 700px;
            text-align: center !important;
            margin: 0 auto;
        }

        /* Style Streamlit Columns as Cards */
        div[data-testid="column"] {
            background-color: white;
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 1.5rem;
        }
        .card-icon {
            background-color: #6366f1;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .card-subtitle {
            font-size: 0.875rem;
            color: #64748b;
            margin: 0;
        }

        /* File Uploader Customization */
        .stFileUploader > div > div {
            background-color: #f8fafc !important;
            border: 2px dashed #cbd5e1 !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            transition: all 0.3s !important;
        }
        .stFileUploader > div > div:hover {
            border-color: #6366f1 !important;
            background-color: #eff6ff !important;
        }

        /* Custom Button */
        div.stButton > button {
            background-color: #eff6ff;
            color: #6366f1;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #e0e7ff;
            color: #4f46e5;
            border: none;
        }
        
        /* Secondary Primary Button Styling via type="primary" */
        div.stButton > button[data-testid="baseButton-primary"] {
            background-color: #eff6ff;
            color: #6366f1;
            border: none;
        }
        div.stButton > button[data-testid="baseButton-primary"]:hover {
            background-color: #e0e7ff;
            color: #4f46e5;
        }

        /* Footer */
        .footer-container {
            text-align: center;
            color: #64748b;
            font-size: 0.875rem;
            margin-top: 4rem;
            padding-bottom: 2rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        
        /* Metric Box */
        .info-box {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 1rem;
        }
        .info-icon {
            color: #6366f1;
            display: flex;
            align-items: center;
        }
        .info-text-sm {
            font-size: 0.75rem;
            color: #64748b;
            margin: 0;
            font-weight: 500;
        }
        .info-value {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        
        /* Slider Customization */
        .stSlider > div > div > div > div {
            background-color: #6366f1;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- Hero Section ---
st.markdown(f"""
<div class="hero-section">
    <h1 class="hero-title">Privacy-first facial detection and anonymization<br>using <span>YOLOv8</span> Neural Networks</h1>
    <p class="hero-subtitle">Detect faces accurately and automatically blur them to protect identity and ensure privacy.</p>
</div>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = YOLO("face_detector_yolov8.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Main Interface ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem;">
            <div class="card-header" style="margin-bottom: 0;">
                <div class="card-icon" style="background-color: #8b5cf6;">{SVG_UPLOAD}</div>
                <div>
                    <h3 class="card-title">1. Input Source</h3>
                    <p class="card-subtitle">Upload an image to start processing</p>
                </div>
            </div>
            <div style="background-color: #f1f5f9; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; color: #64748b;">
                <span style="color: #3b82f6; font-weight: 600;">Supports</span> JPG, JPEG, PNG, WEBP
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    def clear_sample():
        st.session_state.selected_sample = None

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", on_change=clear_sample)
    
    st.markdown("<p style='font-size: 0.875rem; color: #64748b; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem;'>Or select a sample image:</p>", unsafe_allow_html=True)
    
    sample_cols = st.columns(4)
    sample_paths = ["sample_face.jpg", "sample_face1.jpg", "sample_face2.jpg", "sample_face3.jpg"]
    
    for i, col in enumerate(sample_cols):
        with col:
            st.image(sample_paths[i], use_container_width=True)
            if st.button(f"Sample {i+1}", key=f"btn_sample_{i}", use_container_width=True):
                st.session_state.selected_sample = sample_paths[i]

    image_to_process = None
    if st.session_state.selected_sample is not None:
        image_to_process = Image.open(st.session_state.selected_sample).convert('RGB')
    elif uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert('RGB')

    if image_to_process is not None:
        image = image_to_process
        
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-top: 1rem; margin-bottom: 1.5rem;">
            <div class="info-box" style="flex: 1; margin: 0;">
                <div class="info-icon" style="color: #3b82f6;">{SVG_IMAGE}</div>
                <div>
                    <p class="info-text-sm">Max File Size</p>
                    <p class="info-value">200MB</p>
                </div>
            </div>
            <div class="info-box" style="flex: 1; margin: 0;">
                <div class="info-icon" style="color: #8b5cf6;">{SVG_LOCK}</div>
                <div>
                    <p class="info-text-sm">Your Data is Safe</p>
                    <p class="info-value">Processed securely</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='display: flex; align-items: center; gap: 6px; font-size: 0.875rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;'><span style='color: #64748b;'>{SVG_COG}</span> Tuning Settings</div>", unsafe_allow_html=True)
        conf_threshold = st.slider("Detection Confidence", 0.05, 1.0, 0.50, 0.05)
        blur_strength = st.select_slider("Blur Intensity", options=[15, 31, 51, 71, 99, 151], value=15)
        
        st.markdown("""
        <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 8px; margin-top: 1rem; font-size: 0.85rem; color: #1e3a8a; line-height: 1.4;">
            <strong style="color: #1d4ed8; display: flex; align-items: center; gap: 4px; margin-bottom: 4px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.9 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg> Pro Tip
            </strong>
            Lower the <b>Detection Confidence</b> if some faces aren't being detected. Increase it if non-faces are accidentally blurred.
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem;">
            <div class="card-header" style="margin-bottom: 0;">
                <div class="card-icon" style="background-color: #f1f5f9; color: #0f172a;">{SVG_EYE}</div>
                <div>
                    <h3 class="card-title">2. Output Preview</h3>
                    <p class="card-subtitle">Processed image with blurred faces</p>
                </div>
            </div>
            <div style="background-color: #ecfdf5; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; color: #059669; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                {SVG_SHIELD_CHECK} Face Blur Enabled
            </div>
        </div>
    """, unsafe_allow_html=True)

    if image_to_process is not None and model:
        if st.button("Run Detection & Blur", type="primary", use_container_width=True):
            with st.spinner('Analyzing image frames...'):
                img_array = np.array(image)
                results = model.predict(img_array, conf=conf_threshold, verbose=False)
                
                # Processing
                img_processed = img_array.copy()
                face_count = 0
                total_conf = 0.0
                
                for r in results:
                    boxes = r.boxes
                    face_count += len(boxes)
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        total_conf += float(box.conf[0])
                        
                        # Apply specialized blur
                        face_roi = img_processed[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            ksize = (blur_strength, blur_strength)
                            blurred_roi = cv2.GaussianBlur(face_roi, ksize, 30)
                            img_processed[y1:y2, x1:x2] = blurred_roi
                
                avg_conf = (total_conf / face_count) * 100 if face_count > 0 else 0
                avg_conf_str = f"{avg_conf:.0f}%" if face_count > 0 else "N/A"

                st.image(img_processed, use_container_width=True, clamp=True)
                
                # Metrics Display
                st.markdown(f"""
                <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                    <div class="info-box" style="flex: 1; margin: 0;">
                        <div class="info-icon" style="color: #3b82f6;">{SVG_USER}</div>
                        <div>
                            <p class="info-text-sm">Faces Detected</p>
                            <p class="info-value">{face_count}</p>
                        </div>
                    </div>
                    <div class="info-box" style="flex: 1; margin: 0;">
                        <div class="info-icon" style="color: #059669;">{SVG_SHIELD_CHECK}</div>
                        <div>
                            <p class="info-text-sm">Status</p>
                            <p class="info-value">Anonymized</p>
                        </div>
                    </div>
                    <div class="info-box" style="flex: 1; margin: 0;">
                        <div class="info-icon" style="color: #64748b;">{SVG_TARGET}</div>
                        <div>
                            <p class="info-text-sm">Avg. Confidence</p>
                            <p class="info-value">{avg_conf_str}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Download Button
                result_img = Image.fromarray(img_processed)
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="Download Processed Image",
                    data=byte_im,
                    file_name="anonymized_face.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.image(image, use_container_width=True, clamp=True)
            st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 1rem; color: #64748b; font-size: 0.875rem;">
                    {SVG_LOCK_SM} Face blur is <strong>enabled</strong> by default to protect privacy. Click the button to process.
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #f8fafc; border-radius: 16px; height: 350px; display: flex; align-items: center; justify-content: center; border: 2px dashed #e2e8f0;">
            <p style="color: #94a3b8; font-weight: 500;">Image preview will appear here</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f'<div class="footer-container"><span style="color: #6366f1;">{SVG_CODE}</span> Developed by joel | Neural Network Privacy System v1.0</div>', unsafe_allow_html=True)

