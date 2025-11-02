import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64, io
from st_clickable_images import clickable_images

# --- your modules ---
from tools.detector import PlateDetector        # YOLO detector
from tools.processor import ImagePreprocessor   # OCR preprocessor / OCR model

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Automatic Number Plate Recognition (ANPR)", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px;'>
        🚗 Automatic Number Plate Recognition (ANPR)
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar Config
# ---------------------------
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.7)
weights_path = "v3_best.pt"

detector = PlateDetector(weights=weights_path, conf=confidence)
processor = ImagePreprocessor(min_width=300, min_height=200, ocr_langs=['en'])

selected_image = None

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        selected_image = image

# ---------------------------
# Clickable Image Grid (sample images)
# ---------------------------
st.subheader("Or select an image from the gallery")

@st.cache_data
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# example local images (adjust to your dataset)
local_images = [f"./images/{i}.jpg" for i in range(1, 11)]
base64_images = [f"data:image/png;base64,{get_img_as_base64(img)}" for img in local_images]

clicked = clickable_images(
    base64_images,
    titles=[f"Image #{i+1}" for i in range(len(local_images))],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "gap": "10px"},
    img_style={"margin": "5px", "height": "200px", "border-radius": "10px", "object-fit": "cover", "box-shadow": "0 2px 6px rgba(0,0,0,0.2)"},
)

# if user clicked on gallery image
if uploaded_file is None and clicked > -1:
    image_bytes = base64.b64decode(get_img_as_base64(local_images[clicked]))
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    selected_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------------
# Detection + OCR Pipeline
# ---------------------------
if selected_image is not None:
    st.write("---")
    st.subheader("🔍 Detection Results")

    # Detect license plates with YOLO
    crops = detector.crop(selected_image, original_name="selected_image", save_dir=None)
    annotated_image = selected_image.copy()

    if crops:
        for item in crops:
            x1, y1, x2, y2, conf = item["bbox"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        st.success(f"✅ Detected {len(crops)} license plate(s)")
    else:
        st.warning("No license plates detected.")

    # Display original and annotated side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Plates", use_container_width=True)

    # If any plate detected → run OCR
    if crops:
        st.write("---")
        st.subheader("🧠 OCR Recognition Results")

        for i, item in enumerate(crops):
            crop_img = item["crop"]
            bbox = item["bbox"]
            conf = bbox[4]

            # Run OCR preprocessing + recognition
            ocr_results, processed_img = processor.adaptive_ocr(crop_img, return_image=True)

            # Display results
            with st.container():
                st.markdown(f"**Plate {i+1} (confidence: {conf:.2f})**")
                col_a, col_b, col_c = st.columns([1, 1, 2])

                with col_a:
                    st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), caption="Detected Plate", use_container_width=True)
                with col_b:
                    st.image(processed_img, caption="Processed for OCR", use_container_width=True)
                with col_c:
                    if ocr_results:
                        st.markdown("**Detected Text:**")
                        for bbox, text, score in ocr_results:
                            st.write(f"🅿️ `{text}` (conf: {score:.2f})")
                    else:
                        st.warning("No text detected.")
else:
    st.info("📤 Upload or select an image to start license plate detection and OCR.")
