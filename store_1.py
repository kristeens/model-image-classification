
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🍔 Fast Food Classifier", layout="centered")

st.title("🍕 Fast Food Image Classifier")
st.write("Upload a food image and the model will predict its class!")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("FineTuned-DenseNet.h5")
    return model

model = load_model()

# -------------------- CLASS LABELS --------------------
# ⚠️ Replace these with your actual class names from the training directory
class_names = [
    "Burger", "Pizza", "Hotdog", "Fries", "Sandwich",
    "Taco", "Nuggets", "Donut", "Pasta", "Salad"
]

# -------------------- UPLOAD IMAGE --------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # same rescaling as training

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display result
    st.markdown(f"### 🍟 Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
