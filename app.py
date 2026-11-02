
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="FoodVision AI",
    page_icon="🍛",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        BASE_DIR / "fine_tuned_best_food_model.keras",
        compile=False
    )

    with open(BASE_DIR / "class_names.json") as f:
        classes = json.load(f)

    return model, classes

st.title("🍛 FoodVision AI")
st.subheader("Indian Food Identification & Classification")

st.write(
    "Upload a food photograph to identify it "
    "using our MobileNetV2 deep learning model."
)

try:
    with st.spinner("Loading AI model..."):
        model, class_names = load_model()

    st.success("AI model ready! 80 food categories supported.")

    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded food image", use_container_width=True)

        if st.button("Identify Food", type="primary"):

            with st.spinner("Analyzing image..."):

                resized = image.resize((224, 224))

                arr = np.array(resized, dtype=np.float32)
                arr = np.expand_dims(arr, axis=0)

                predictions = model.predict(arr, verbose=0)[0]

                top_indices = np.argsort(predictions)[::-1][:3]

            st.subheader("Prediction Results")

            best_idx = int(top_indices[0])

            st.success(
                f"Predicted food: {class_names[best_idx]}"
            )

            for idx in top_indices:
                idx = int(idx)
                confidence = float(predictions[idx]) * 100

                st.write(
                    f"**{class_names[idx]} — {confidence:.2f}%**"
                )

                st.progress(float(predictions[idx]))

            st.caption(
                "Confidence scores are model outputs, not "
                "guarantees of correctness. The model supports "
                "the 80 classes in its training dataset."
            )

except Exception as e:
    st.error(f"Unable to load the model: {e}")
