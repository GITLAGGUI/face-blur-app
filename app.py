import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="Face Detection & Blurring", page_icon="🎭", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.2em;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Face Detection & Blurring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by YOLOv8</div>', unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Load the best.pt model. If not found, fallback to yolov8n-face.pt
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.warning(f"Could not load 'best.pt'. Please ensure it's in the same folder. Falling back to default YOLOv8n... (Error: {e})")
        # For demonstration purposes, if the face model isn't available, we fallback to yolov8n
        return YOLO("yolov8n.pt") 

model = load_model()

# Image Upload
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    st.markdown("### 📸 Uploaded Image")
    st.image(image, use_container_width=True)
    
    # Confidence slider
    conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    
    if st.button("Detect and Blur Faces 🚀"):
        with st.spinner('Processing...'):
            # Run inference
            results = model.predict(img_array, conf=conf_threshold)
            
            # Create copies for displaying bounding boxes and blurred faces
            img_with_boxes = img_array.copy()
            img_blurred = img_array.copy()
            
            # Iterate through detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box on the first copy
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    
                    # Add label
                    conf = float(box.conf[0])
                    label = f"Face {conf:.2f}"
                    cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    # Apply Gaussian Blur to the face region on the second copy
                    face_roi = img_blurred[y1:y2, x1:x2]
                    
                    # Ensure the ROI is valid
                    if face_roi.size > 0:
                        # Use a large kernel size for strong blurring
                        blurred_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                        img_blurred[y1:y2, x1:x2] = blurred_roi

            st.markdown("### 🎯 Prediction (Bounding Boxes)")
            st.image(img_with_boxes, use_container_width=True)
            
            st.markdown("### 🕵️ Processed Image (Faces Blurred)")
            st.image(img_blurred, use_container_width=True)
            
            st.success("Processing Complete!")
