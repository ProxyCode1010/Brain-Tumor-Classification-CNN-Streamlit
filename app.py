import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
model = tf.keras.models.load_model('model/brain_tumor_model.h5')

st.title("🧠 Brain Tumor Classifier")
st.markdown("Upload a brain MRI image to predict tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.convert("RGB")  
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.success(f"🧠 Predicted Tumor Type: *{predicted_class.upper()}*")
