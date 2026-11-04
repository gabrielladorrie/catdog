
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Title
st.title("Cat vs Dog Classifier 🐱🐶")

# Sidebar for image upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a cat or dog image...", type=["jpg", "png", "jpeg"])

# Load pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_model.h5")  # You must save your trained model as 'cnn_model.h5'
    return model

model = load_model()

# Preprocess image for prediction
def preprocess_image(image: Image.Image):
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed)[0][0]
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.2%}")

        # Confidence bar chart
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(["Cat", "Dog"], [1 - prediction, prediction], color=["blue", "orange"])
        ax.set_ylim([0, 1])
        st.pyplot(fig)

# Optionally add model summary
with st.expander("Show Model Summary"):
    model.summary(print_fn=lambda x: st.text(x))
