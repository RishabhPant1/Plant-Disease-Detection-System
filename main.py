import os
import json
import time
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

from image_validator import load_validator, validate_colored_leaf

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

HOME_IMAGE = os.path.join(
    BASE_DIR,
    "home_page.JPG"
)

# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    return model
# =====================================================
# Load Class Names
# =====================================================

@st.cache_resource
def load_class_names():

    with open(CLASS_PATH, "r") as f:

        class_names = json.load(f)

    return class_names


# =====================================================
# Load Image Validator (Zero-Shot CLIP Gating)
# =====================================================

@st.cache_resource
def get_validator():

    return load_validator()


# =====================================================
# Prediction Function
# =====================================================

def predict_image(image, model):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(image)

    img_array = np.expand_dims(img_array, axis=0)

    # Note: plant_disease_model.keras already includes TrueDivide and Subtract
    # layers in its graph for MobileNetV2 normalization. Do not apply preprocess_input here.
    prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    probabilities = prediction[0]

    return predicted_index, confidence, probabilities


# =====================================================
# Load Resources
# =====================================================

try:

    model = load_model()

    class_names = load_class_names()

    validator_model, validator_processor = get_validator()

except Exception as e:

    st.error(f"❌ Failed to load application resources.\n\n{e}")

    st.stop()
# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("🌿 Dashboard")

st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "ℹ️ About",
        "🔍 Disease Recognition"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.info(
    """
**Model:** MobileNetV2

**Input Size:** 224 × 224

**Classes:** 3

**Validation Accuracy:** 98.16%
"""
)


# =====================================================
# Home Page
# =====================================================

if app_mode == "🏠 Home":

    st.title("🌿 Plant Disease Detection System")

    st.image(
        HOME_IMAGE,
        use_container_width=True
    )

    st.markdown(
        """
            Welcome to the **Plant Disease Detection System**.

            This application uses **Transfer Learning with MobileNetV2**
            to identify diseases in Black Pepper leaves.

            The model can classify images into:

            - 🌱 Healthy
            - 🍂 Leaf Blight
            - 🟡 Yellow Mottle Virus

            Navigate to **Disease Recognition** from the sidebar
            to upload a leaf image and receive predictions.
        """
    )

# =====================================================
# About Page
# =====================================================

elif app_mode == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown("""
        ### 🌿 Plant Disease Detection using Deep Learning

        This project is a deep learning-based web application that detects diseases in **Black Pepper leaves** using **Transfer Learning** with **MobileNetV2**.

        The application allows users to upload an image of a leaf and predicts whether it belongs to one of the following classes:

        - 🌱 Healthy
        - 🍂 Leaf Blight
        - 🟡 Yellow Mottle Virus

        ---

        ### 🧠 Technologies Used

        - Python
        - TensorFlow & Keras
        - MobileNetV2 (Transfer Learning)
        - Streamlit
        - NumPy
        - Pillow

        ---

        ### 📂 Dataset

        - Total Images: **819**
        - Classes: **3**
        - Images per Class: **273**

        The dataset was trained using data augmentation techniques including:

        - Horizontal Flip
        - Random Rotation
        - Random Zoom
        - Random Contrast

        ---

        ### 📊 Model Performance

        - Validation Accuracy: **98.16%**
        - Precision: **98%**
        - Recall: **98%**
        - F1-Score: **98%**

        The model was evaluated using:

        - Confusion Matrix
        - Classification Report
        - Validation Accuracy & Loss Curves

        ---

        This project was developed as a Machine Learning portfolio project demonstrating image classification using Transfer Learning.
        """)
    
# =====================================================
# Disease Recognition Page
# =====================================================

elif app_mode == "🔍 Disease Recognition":

    st.title("🔍 Plant Disease Recognition")

    st.write(
        "Upload a clear image of a **Black Pepper leaf** for disease detection."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file)
        except Exception:
            st.error("Please upload a valid colored leaf photo or image")
            st.stop()

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Prediction")

            if st.button(
                "Analyze Leaf",
                use_container_width=True
            ):

                with st.spinner("Analyzing image..."):

                    is_valid, validation_msg = validate_colored_leaf(
                        image,
                        validator_model,
                        validator_processor
                    )

                    if not is_valid:
                        st.error(validation_msg)
                    else:
                        start_time = time.time()

                        predicted_index, confidence, probabilities = predict_image(
                            image,
                            model
                        )

                        inference_time = time.time() - start_time

                        predicted_class = class_names[predicted_index]

                        display_names = {
                            "healthy": "🌱 Healthy",
                            "leaf_blight": "🍂 Leaf Blight",
                            "yellow_mottle_virus": "🟡 Yellow Mottle Virus"
                        }

                        st.success(
                            f"### Prediction: {display_names[predicted_class]}"
                        )

                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.2f}%"
                        )

                        st.metric(
                            label="Inference Time",
                            value=f"{inference_time:.3f} sec"
                        )

                        st.markdown("---")

                        st.subheader("Prediction Probabilities")

                        for i, probability in enumerate(probabilities):

                            st.write(
                                f"**{display_names[class_names[i]]}**"
                            )

                            st.progress(
                                int(probability * 100)
                            )

                            st.caption(
                                f"{probability*100:.2f}%"
                            )

                        st.markdown("---")
                        # ==========================================
                        # Recommendations
                        # ==========================================

                        if predicted_class == "healthy":

                            st.success(
                                """
                                    ### 🌱 Healthy Plant

                                    The leaf appears to be healthy.

                                    **Recommendations**

                                    - Continue regular watering.
                                    - Apply fertilizers as required.
                                    - Monitor the plant regularly.
                                    - Maintain proper sunlight and airflow.
                                """
                            )

                        elif predicted_class == "leaf_blight":

                            st.warning(
                                """
                                    ### 🍂 Leaf Blight Detected

                                    **Recommended Actions**

                                    - Remove infected leaves immediately.
                                    - Avoid overhead watering.
                                    - Improve air circulation.
                                    - Apply a suitable fungicide if required.
                                    - Monitor nearby plants for symptoms.
                                """
                            )

                        elif predicted_class == "yellow_mottle_virus":

                            st.error(
                                """
                                    ### 🟡 Yellow Mottle Virus Detected

                                    **Recommended Actions**

                                    - Remove infected leaves.
                                    - Isolate infected plants if possible.
                                    - Control insect vectors.
                                    - Keep farming tools sanitized.
                                    - Consult a local agricultural expert if infection spreads.
                                """
                            )

                            st.info(
                                """
                                    **Note**

                                    This prediction is generated using a deep learning model.
                                    Always consult an agricultural expert before making major treatment decisions.
                                """
                            )

            