import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("../runs/plates/v3_RainyKlNight_SupplimentaryMorning_TaxiEV/weights/best.pt") # Replace with your model path

# Confidence threshold slider
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Image uploader (only jpg/png)
uploaded_file = st.file_uploader(
    "Upload an Image", 
    type=["jpg", "jpeg", "png"]
)

# Run inference
if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model(image, conf=confidence)

    # Plot and display annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, channels="BGR", caption="Detections")

else:
    st.info("Please upload a JPG or PNG image to start detection.")