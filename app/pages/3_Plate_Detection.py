import streamlit as st
import cv2
import numpy as np
from tools.detector import PlateDetector
from st_clickable_images import clickable_images
from PIL import Image
import base64
import io

st.set_page_config(page_title="YOLO License Plate Detector", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px;'>
        Object Detection Module
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar settings
# ---------------------------
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.7)
weights_path = "v3_best.pt"
detector = PlateDetector(weights=weights_path, conf=confidence)

selected_image = None  # The image that YOLO will run on

# ---------------------------
# Image uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        selected_image = image

# ---------------------------
# Clickable image grid
# ---------------------------
st.subheader("Or click an image from the grid")

@st.cache_data
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

local_images = ["./images/images_original/1.jpg", "./images/images_original/2.jpg", "./images/images_original/3.jpg", "./images/images_original/4.jpg", "./images/images_original/5.jpg", "./images/images_original/6.jpg", "./images/images_original/7.jpg", "./images/images_original/8.jpg", "./images/images_original/9.jpg", "./images/images_original/10.jpg"]  # add more if needed
base64_images = [f"data:image/png;base64,{get_img_as_base64(img)}" for img in local_images]

clicked = clickable_images(
    base64_images,
    titles=[f"Image #{i+1}" for i in range(len(local_images))],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)

# ---------------------------
# Handle clicked image (only if no upload)
# ---------------------------
if uploaded_file is None and clicked > -1:
    image_bytes = base64.b64decode(get_img_as_base64(local_images[clicked]))
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    selected_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------------
# Run YOLO detection
# ---------------------------
if selected_image is not None:
    st.write("---")
    st.subheader('Detection Results') 
    # Detect plates
    crops = detector.crop(selected_image, original_name="selected_image", save_dir=None)
    annotated_image = selected_image.copy()

    # Draw bounding boxes if any plates detected
    if crops:
        for item in crops:
            x1, y1, x2, y2, conf = item["bbox"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.success(f"Detected {len(crops)} plate(s)")
    else:
        st.warning("No license plates detected.")

    # Display original and annotated side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB),
                 caption="Original Image", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                 caption="Annotated Image", use_container_width=True)

    # Display cropped plates below
    if crops:
        st.subheader("Cropped Plates")
        for i, item in enumerate(crops):
            crop_img = item["crop"]
            bbox = item["bbox"]
            st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB),
                     caption=f"Plate {i+1} (conf: {bbox[4]:.2f})",
                     width=200)  # optional fixed width
else:
    st.info("Please upload a JPG/PNG image or click an image from the grid to start detection.")