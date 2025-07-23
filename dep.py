import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
 
# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mask_model.h5")
    return model
    
 
model = load_model()
 
# Set Streamlit page settings
st.set_page_config(page_title="Mask Detection", layout="centered")
st.title("😷 Mask Detection CNN")
 
# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
 
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

 
    # Preprocess image
    img_size = (128, 128)  # Ensure this matches your model input size
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # RGBA to RGB
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
 
    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "❌ No Mask" if prediction >= 0.5 else "😷 Mask"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
 
    # Show result
    st.subheader("Prediction:")
    st.markdown(f"**{label}** (Confidence: `{confidence:.2%}`)")
 
 