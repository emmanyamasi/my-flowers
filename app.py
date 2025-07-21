import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load model
model = keras.models.load_model('my_modela.keras')

# Class names (adjust to your training labels)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

st.title("🌸 Flower Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_resized = image.resize((180, 180))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🌼 Predicted Flower: **{predicted_class}**")
