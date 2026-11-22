"""Streamlit interface for handwritten digit prediction."""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.keras"

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write(
    "Upload an image containing one handwritten digit (0–9). "
    "The CNN will process the image and predict the digit."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to the MNIST-like 28x28 grayscale format."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    if np.asarray(image, dtype=np.float32).mean() > 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, 28, 28, 1)


if not MODEL_PATH.exists():
    st.warning(
        "The trained model has not been created yet. In your project terminal, "
        "run python train.py, then refresh this page."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a digit image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Uploaded image", width=250)

        if st.button("Predict digit", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                processed = preprocess_image(original_image)
                probabilities = model.predict(processed, verbose=0)[0]
                predicted_digit = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_digit])

            st.success(f"Predicted digit: **{predicted_digit}**")
            st.metric("Model confidence", f"{confidence:.2%}")

            st.subheader("Class probabilities")
            st.bar_chart(
                {str(i): float(probabilities[i]) for i in range(10)}
            )
    except Exception as error:
        st.error(f"Could not process this image: {error}")
