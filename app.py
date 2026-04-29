# app.py
# Clean Streamlit code to connect your CNN model (cnn_model.h5)

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pneumonia Detection using CNN")
st.write("Upload a Chest X-ray image to predict whether it is NORMAL or PNEUMONIA.")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("cnn_model.h5")
    return model

model = load_cnn_model()

# -------------------------------
# Class Labels
# IMPORTANT:
# Change according to your train_generator.class_indices
# Example:
# {'NORMAL': 0, 'PNEUMONIA': 1}
# -------------------------------
class_names = ['NORMAL', 'PNEUMONIA']

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))   # same as training size
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # normalization
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    display_image, processed_image = preprocess_image(uploaded_file)

    st.image(display_image, caption="Uploaded Image", use_container_width=True)

    # Predict Button
    if st.button("Predict"):
        with st.spinner("Analyzing Image..."):
            prediction = model.predict(processed_image)

            # For multi-class / categorical output
            predicted_index = np.argmax(prediction)
            predicted_label = class_names[predicted_index]
            confidence = float(np.max(prediction)) * 100

        st.success("Prediction Complete ✅")

        st.subheader(f"Prediction: {predicted_label}")
        st.write(f"Confidence Score: **{confidence:.2f}%**")

        # Optional result message
        if predicted_label == "PNEUMONIA":
            st.error("⚠️ Signs of Pneumonia Detected")
        else:
            st.success("✅ Chest X-ray appears Normal")
