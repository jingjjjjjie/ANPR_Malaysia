import streamlit as st
import cv2
import numpy as np
from tools.processor import ImagePreprocessor  # your class
from st_clickable_images import clickable_images
from PIL import Image
import base64
import io

st.set_page_config(page_title="OCR and Image Preprocessor Module", layout="wide")
st.markdown(
    "<h1 style='text-align: center; font-size: 48px;'>OCR Image Preprocessor</h1>",
    unsafe_allow_html=True
)

# ---------------------------
# Image uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
selected_image = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    selected_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ---------------------------
# Clickable image grid
# ---------------------------
st.subheader("Or select an image from the grid")

@st.cache_data
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# List your local images here
local_images = ["./images/images_cropped/1_0.jpg","./images/images_cropped/2_0.jpg","./images/images_cropped/3_0.jpg","./images/images_cropped/4_0.jpg","./images/images_cropped/5_0.jpg","./images/images_cropped/6_0.jpg","./images/images_cropped/7_0.jpg","./images/images_cropped/8_2.jpg","./images/images_cropped/9_0.jpg","./images/images_cropped/10_0.jpg"]

base64_images = [f"data:image/png;base64,{get_img_as_base64(img)}" for img in local_images]



clicked = clickable_images(
    base64_images,
    titles=[f"Image #{i+1}" for i in range(len(local_images))],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "75px"},
)

# ---------------------------
# Handle clicked image (only if no upload)
# ---------------------------
if uploaded_file is None and clicked > -1:
    image_bytes = base64.b64decode(get_img_as_base64(local_images[clicked]))
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    selected_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------------
# Run OCR preprocessing
# ---------------------------

st.write("---")  # Horizontal rule
if selected_image is not None:
    st.subheader('Image Processing Results')
    processor = ImagePreprocessor(min_width=300, min_height=200, ocr_langs=['en'])
    ocr_results, processed_img = processor.adaptive_ocr(selected_image, return_image=True)

    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    with col2:
        st.image(processed_img, caption="Processed Image", use_container_width=True)

    # Display OCR results
    st.subheader("OCR Results")
    if ocr_results:
        for bbox, text, conf in ocr_results:
            st.write(f"{text} ({conf:.2f})")
    else:
        st.warning("No text detected.")
else:
    st.info("Please upload an image or click an image from the grid to start OCR processing.")
