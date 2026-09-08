import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PIL import Image
from detector import PlateDetector
import torch
from ultralytics.nn.tasks import DetectionModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Fix PyTorch 2.6+ unpickling error for Ultralytics models
torch.serialization.add_safe_globals([DetectionModel])
# Set page config
st.set_page_config(
    page_title="Egyptian License Plate Detector",
    page_icon="🚗",
    layout="wide",
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .ocr-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .ocr-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0F172A;
        direction: rtl;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(model_path: str):
    return PlateDetector(model_path=model_path)


def main():
    st.markdown('<div class="main-title">🚗 Egyptian License Plate Detector & OCR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an image to detect license plates and extract character/number sequences.</div>', unsafe_allow_html=True)

    # Sidebar configurations
    st.sidebar.header("Settings")
    confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
    model_path = st.sidebar.text_input("YOLO Model Weight Path", value="best.pt")

    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found! Please ensure your trained weights (`.pt`) are placed in the root directory.")
        return

    # Load Model
    with st.spinner("Loading YOLO & OCR Model..."):
        detector = load_detector(model_path)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Processing image..."):
            annotated_img, detections = detector.detect_and_ocr(image, conf_threshold=confidence_thresh)

        with col2:
            st.subheader("Detection Result")
            st.image(annotated_img, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 OCR Results & Extracted Plates")

        if len(detections) == 0:
            st.info("No license plates detected above the confidence threshold.")
        else:
            for det in detections:
                st.markdown(f"### Plate #{det['id']} (Confidence: `{det['confidence']:.2%}`)")
                crop_col, text_col = st.columns([1, 2])

                with crop_col:
                    st.image(det["crop"], caption=f"Cropped Plate #{det['id']}", width=250)

                with text_col:
                    st.markdown('<div class="ocr-card">', unsafe_allow_html=True)
                    st.markdown("**Recognized Characters / Text:**")
                    st.markdown(f'<div class="ocr-text">{det["text"]}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()